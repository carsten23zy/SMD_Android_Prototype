package com.edgy.privacy.vocoder

import android.util.Log
import com.edgy.privacy.ml.ModelConfig
import com.edgy.privacy.util.DSP
import org.jtransforms.fft.DoubleFFT_1D
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Griffin-Lim vocoder: reconstructs time-domain audio directly from a
 * normalized log-mel spectrogram — the same representation produced by
 * [com.edgy.privacy.audio.MelSpectrogramExtractor] and consumed by the encoder.
 *
 * This mirrors `reconstruct_wav_griffinlim` in `EDGY_ExportedModels_Inference.ipynb`,
 * which feeds the original log-mel directly into Griffin-Lim — the decoder does
 * not need any encoder output (VQ embeddings or codebook indices).
 *
 * Pipeline:
 *   normalized log-mel [nMels, T]
 *     → denormalize to dB:  mel_db = (mel_norm - 1) * top_db
 *     → dB → amplitude:     mel_amp = 10^(mel_db / 20)
 *     → mel pseudo-inverse: linear_amp = max(0, M⁺ · mel_amp)
 *     → Griffin-Lim phase reconstruction (with momentum)
 *     → de-emphasis (inverse of analysis pre-emphasis)
 *     → peak-normalize to 0.98
 */
class GriffinLimVocoder(
    private val config: ModelConfig,
    melFilterbank: Array<DoubleArray>,
    private val iterations: Int = DEFAULT_ITERATIONS,
    private val momentum: Double = DEFAULT_MOMENTUM
) {
    companion object {
        private const val TAG = "GriffinLimVocoder"
        const val DEFAULT_ITERATIONS = 32
        const val DEFAULT_MOMENTUM = 0.99
    }

    private val sampleRate = config.preprocessing.sampleRate
    private val nFft = config.preprocessing.nFft
    private val nMels = config.preprocessing.nMels
    private val hopLength = config.preprocessing.hopLength
    private val winLength = config.preprocessing.winLength
    private val topDb = config.preprocessing.topDb.toDouble()
    private val preemphCoeff = config.preprocessing.preemph
    private val numBins = nFft / 2 + 1

    // Mel pseudo-inverse [numBins, nMels], computed once from the analysis filterbank.
    private val melPseudoInverse: Array<FloatArray> = computeMelPseudoInverse(melFilterbank)

    // Hann window of length winLength, zero-padded to nFft and centered — matches
    // librosa.stft when n_fft > win_length.
    private val paddedWindow: DoubleArray = DoubleArray(nFft).also { buf ->
        val winOffset = (nFft - winLength) / 2
        for (i in 0 until winLength) {
            buf[winOffset + i] = 0.5 * (1.0 - cos(2.0 * PI * i / winLength))
        }
    }

    private val fft = DoubleFFT_1D(nFft.toLong())

    /** Vocoder is always available — no external decoder weights required. */
    val isAvailable: Boolean get() = true

    /**
     * Reconstruct PCM audio from a normalized log-mel spectrogram.
     *
     * @param logMelNorm Normalized log-mel of shape [nMels, T] in roughly [0, 1],
     *                   matching [com.edgy.privacy.audio.MelSpectrogramExtractor.extract].
     * @return PCM float samples in [-1, 1] (peak-normalized to 0.98).
     */
    fun synthesize(logMelNorm: Array<FloatArray>): FloatArray {
        if (logMelNorm.isEmpty() || logMelNorm[0].isEmpty()) return FloatArray(0)
        require(logMelNorm.size == nMels) {
            "Expected log-mel with $nMels rows, got ${logMelNorm.size}"
        }

        val numFrames = logMelNorm[0].size

        // 1. Denormalize to amplitude mel.
        //    notebook: logmel_db = (logmel_norm - 1) * top_db; mel_amp = db_to_amplitude(db)
        val melAmp = Array(nMels) { m ->
            DoubleArray(numFrames) { t ->
                val db = (logMelNorm[m][t].toDouble() - 1.0) * topDb
                10.0.pow(db / 20.0)
            }
        }

        // 2. Mel → linear magnitude via pseudo-inverse (clamped to non-negative).
        val linearSpec = Array(numBins) { k ->
            DoubleArray(numFrames) { t ->
                var sum = 0.0
                val row = melPseudoInverse[k]
                for (m in 0 until nMels) {
                    sum += row[m].toDouble() * melAmp[m][t]
                }
                if (sum > 0.0) sum else 0.0
            }
        }

        // 3. Griffin-Lim phase reconstruction.
        val signal = griffinLim(linearSpec)

        // 4. De-emphasis (inverse of analysis pre-emphasis).
        val deemphasized = DSP.applyDeemphasis(signal, preemphCoeff)

        // 5. Peak-normalize to 0.98.
        return peakNormalize(deemphasized, 0.98f)
    }

    /**
     * Griffin-Lim with momentum, matching librosa.griffinlim defaults.
     */
    private fun griffinLim(magnitudes: Array<DoubleArray>): FloatArray {
        val numFrames = magnitudes[0].size
        val signalLen = (numFrames - 1) * hopLength + nFft

        // Initialize phase randomly (deterministic seed for reproducibility).
        val random = java.util.Random(0L)
        var angleRe = Array(numBins) { DoubleArray(numFrames) }
        var angleIm = Array(numBins) { DoubleArray(numFrames) }
        for (k in 0 until numBins) {
            for (t in 0 until numFrames) {
                val phi = 2.0 * PI * random.nextDouble()
                angleRe[k][t] = cos(phi)
                angleIm[k][t] = sin(phi)
            }
        }

        // Previous STFT estimate, used by momentum update.
        var prevRe = Array(numBins) { DoubleArray(numFrames) }
        var prevIm = Array(numBins) { DoubleArray(numFrames) }

        val signalScratch = DoubleArray(signalLen)
        val windowSumScratch = DoubleArray(signalLen)
        val newRe = Array(numBins) { DoubleArray(numFrames) }
        val newIm = Array(numBins) { DoubleArray(numFrames) }

        val mFactor = momentum / (1.0 + momentum)

        for (iter in 0 until iterations) {
            // y = istft(|S| * angles)
            istftInto(magnitudes, angleRe, angleIm, signalScratch, windowSumScratch, signalLen)

            // newC = stft(y)
            stftInto(signalScratch, signalLen, newRe, newIm)

            // angles = newC - mFactor * prev; normalize to unit modulus.
            for (k in 0 until numBins) {
                val nrK = newRe[k]; val niK = newIm[k]
                val prK = prevRe[k]; val piK = prevIm[k]
                val arK = angleRe[k]; val aiK = angleIm[k]
                for (t in 0 until numFrames) {
                    val re = nrK[t] - mFactor * prK[t]
                    val im = niK[t] - mFactor * piK[t]
                    val mag = sqrt(re * re + im * im) + 1e-16
                    arK[t] = re / mag
                    aiK[t] = im / mag
                    // Save current STFT estimate for next iteration's momentum term.
                    prK[t] = nrK[t]
                    piK[t] = niK[t]
                }
            }
        }

        // Final synthesis with the last phase estimate.
        istftInto(magnitudes, angleRe, angleIm, signalScratch, windowSumScratch, signalLen)

        // Trim center-padding (librosa center=True): drop nFft/2 from each side.
        val pad = nFft / 2
        val outLen = (signalLen - 2 * pad).coerceAtLeast(0)
        val output = FloatArray(outLen)
        for (i in 0 until outLen) {
            output[i] = signalScratch[i + pad].toFloat()
        }
        return output
    }

    /**
     * Inverse STFT via overlap-add. Writes into [signalOut].
     */
    private fun istftInto(
        magnitudes: Array<DoubleArray>,
        angleRe: Array<DoubleArray>,
        angleIm: Array<DoubleArray>,
        signalOut: DoubleArray,
        windowSumOut: DoubleArray,
        signalLen: Int
    ) {
        java.util.Arrays.fill(signalOut, 0.0)
        java.util.Arrays.fill(windowSumOut, 0.0)
        val numFrames = magnitudes[0].size
        val complexFrame = DoubleArray(nFft * 2)

        for (t in 0 until numFrames) {
            java.util.Arrays.fill(complexFrame, 0.0)
            for (k in 0 until numBins) {
                val mag = magnitudes[k][t]
                val re = mag * angleRe[k][t]
                val im = mag * angleIm[k][t]
                if (k == 0) {
                    complexFrame[0] = re
                    complexFrame[1] = 0.0
                } else if (k == numBins - 1) {
                    complexFrame[2 * k] = re
                    complexFrame[2 * k + 1] = 0.0
                } else {
                    complexFrame[2 * k] = re
                    complexFrame[2 * k + 1] = im
                    // Hermitian symmetric mirror for negative frequencies.
                    complexFrame[2 * (nFft - k)] = re
                    complexFrame[2 * (nFft - k) + 1] = -im
                }
            }

            fft.complexInverse(complexFrame, true)

            val frameStart = t * hopLength
            for (n in 0 until nFft) {
                val idx = frameStart + n
                if (idx < signalLen) {
                    val w = paddedWindow[n]
                    signalOut[idx] += complexFrame[2 * n] * w
                    windowSumOut[idx] += w * w
                }
            }
        }

        for (i in 0 until signalLen) {
            if (windowSumOut[i] > 1e-8) signalOut[i] /= windowSumOut[i]
        }
    }

    /**
     * Forward STFT into preallocated complex buffers.
     */
    private fun stftInto(
        signal: DoubleArray,
        signalLen: Int,
        outRe: Array<DoubleArray>,
        outIm: Array<DoubleArray>
    ) {
        val numFrames = outRe[0].size
        val frame = DoubleArray(nFft)

        for (t in 0 until numFrames) {
            val frameStart = t * hopLength
            for (n in 0 until nFft) {
                val idx = frameStart + n
                frame[n] = if (idx < signalLen) signal[idx] * paddedWindow[n] else 0.0
            }

            fft.realForward(frame)

            // JTransforms packs realForward as: [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ...]
            outRe[0][t] = frame[0]
            outIm[0][t] = 0.0
            outRe[numBins - 1][t] = frame[1]
            outIm[numBins - 1][t] = 0.0
            for (k in 1 until numBins - 1) {
                outRe[k][t] = frame[2 * k]
                outIm[k][t] = frame[2 * k + 1]
            }
        }
    }

    private fun peakNormalize(signal: FloatArray, target: Float): FloatArray {
        if (signal.isEmpty()) return signal
        var peak = 0f
        for (s in signal) {
            val a = abs(s)
            if (a > peak) peak = a
        }
        if (peak < 1e-8f) return signal
        val scale = target / peak
        val out = FloatArray(signal.size)
        for (i in signal.indices) {
            out[i] = (signal[i] * scale).coerceIn(-1f, 1f)
        }
        return out
    }

    /**
     * Compute the mel pseudo-inverse [numBins, nMels] from a mel filterbank
     * [nMels, numBins] as P = M^T * (M*M^T + eps*I)^{-1}.
     */
    private fun computeMelPseudoInverse(melFilterbank: Array<DoubleArray>): Array<FloatArray> {
        val rows = melFilterbank.size        // nMels
        val cols = melFilterbank[0].size     // numBins
        require(rows == nMels && cols == numBins) {
            "Mel filterbank shape [$rows, $cols] does not match [$nMels, $numBins]"
        }
        val eps = 1e-8

        Log.i(TAG, "Computing mel pseudo-inverse from filterbank [$rows, $cols]")

        // G = M * M^T  [nMels, nMels]
        val g = Array(rows) { DoubleArray(rows) }
        for (i in 0 until rows) {
            for (j in i until rows) {
                var sum = 0.0
                for (k in 0 until cols) {
                    sum += melFilterbank[i][k] * melFilterbank[j][k]
                }
                g[i][j] = sum + if (i == j) eps else 0.0
                g[j][i] = g[i][j]
            }
        }

        // Invert G via Gauss-Jordan on augmented [G | I].
        val aug = Array(rows) { i ->
            DoubleArray(2 * rows).also { row ->
                for (j in 0 until rows) row[j] = g[i][j]
                row[rows + i] = 1.0
            }
        }

        for (col in 0 until rows) {
            var maxVal = abs(aug[col][col])
            var maxRow = col
            for (row in col + 1 until rows) {
                val v = abs(aug[row][col])
                if (v > maxVal) { maxVal = v; maxRow = row }
            }
            if (maxRow != col) {
                val tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp
            }

            val pivot = aug[col][col]
            if (abs(pivot) < 1e-15) continue

            for (j in 0 until 2 * rows) aug[col][j] /= pivot

            for (row in 0 until rows) {
                if (row == col) continue
                val factor = aug[row][col]
                for (j in 0 until 2 * rows) {
                    aug[row][j] -= factor * aug[col][j]
                }
            }
        }

        val gInv = Array(rows) { i -> DoubleArray(rows) { j -> aug[i][rows + j] } }

        // P = M^T * G^{-1}  [numBins, nMels]
        val result = Array(cols) { FloatArray(rows) }
        for (k in 0 until cols) {
            for (j in 0 until rows) {
                var sum = 0.0
                for (i in 0 until rows) {
                    sum += melFilterbank[i][k] * gInv[i][j]
                }
                result[k][j] = sum.toFloat()
            }
        }

        Log.i(TAG, "Mel pseudo-inverse computed: [$cols, $rows]")
        return result
    }
}
