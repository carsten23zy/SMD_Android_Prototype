package com.edgy.privacy.vocoder

import android.util.Log
import com.edgy.privacy.ml.ModelConfig
import com.edgy.privacy.util.DSP
import org.jtransforms.fft.DoubleFFT_1D
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Griffin-Lim vocoder: reconstructs time-domain audio from VQ embeddings.
 *
 * Pipeline:
 *   VQ embedding [T', 64] → projection → mel spectrogram [nMels, T']
 *   → pseudo-inverse → linear spectrogram [nFft/2+1, T']
 *   → Griffin-Lim iterative phase estimation → PCM audio
 *
 * The pseudo-inverse matrix converts mel spectrograms back to approximate
 * linear-frequency spectrograms. It is precomputed as:
 *   P = M^T * (M*M^T + eps*I)^{-1}
 * where M is the [nMels, nFft/2+1] mel filterbank.
 *
 * If the projection matrix is not available, the vocoder returns silence
 * (graceful fallback).
 */
class GriffinLimVocoder(
    private val config: ModelConfig,
    projectionMatrix: Array<FloatArray>?,
    private val codebook: Array<FloatArray>?,
    melFilterbank: Array<DoubleArray>? = null
) {
    companion object {
        private const val TAG = "GriffinLimVocoder"
        private const val DEFAULT_ITERATIONS = 30
    }

    private val sampleRate = config.preprocessing.sampleRate
    private val nFft = config.preprocessing.nFft
    private val nMels = config.preprocessing.nMels
    private val hopLength = config.preprocessing.hopLength
    private val winLength = config.preprocessing.winLength
    private val numBins = nFft / 2 + 1

    private val iterations: Int = DEFAULT_ITERATIONS

    // Precomputed Hann window for overlap-add synthesis
    private val hannWindow = DoubleArray(winLength) { n ->
        0.5 * (1.0 - cos(2.0 * PI * n / winLength))
    }

    // FFT instance
    private val fft = DoubleFFT_1D(nFft.toLong())

    // Mel pseudo-inverse: [numBins, nMels] — loaded from file or computed from filterbank
    private val melPseudoInverse: Array<FloatArray>? = projectionMatrix
        ?: melFilterbank?.let { computeMelPseudoInverse(it) }

    val isAvailable: Boolean get() = melPseudoInverse != null && codebook != null

    /**
     * Reconstruct PCM audio from VQ embedding and codebook indices.
     *
     * @param vqEmbedding [T', embeddingDim] VQ-quantized embeddings
     * @param codebookIndices [T'] indices into the codebook (unused if vqEmbedding provided directly)
     * @return PCM float samples in [-1, 1] range, or empty array if vocoder unavailable
     */
    fun synthesize(
        vqEmbedding: Array<FloatArray>,
        codebookIndices: IntArray? = null
    ): FloatArray {
        if (!isAvailable) {
            Log.w(TAG, "Vocoder unavailable (missing mel pseudo-inverse or codebook)")
            return FloatArray(0)
        }

        val proj = melPseudoInverse!!

        // Step 1: VQ embedding → approximate mel spectrogram
        // vqEmbedding is [T', 64], we need to map it back to [nMels, T']
        // The VQ embedding IS the encoder's quantized representation of the mel.
        // We project it back: mel_approx[m, t] = sum_d(embedding[t, d] * W[m, d])
        // But we don't have a trained decoder — so we use codebook indices to
        // look up codebook vectors, which approximate the encoded mel frames.
        val numFrames = vqEmbedding.size

        // Step 1a: Reconstruct mel spectrogram from VQ embeddings
        // Each VQ embedding row [64] is projected to mel space [nMels]
        // We need a simple linear projection: since the encoder compressed mel→64d,
        // we estimate mel by treating the 64d embedding as a compressed mel frame.
        // For now, use direct projection: mel ≈ vqEmb * projWeight
        // Since we don't have the decoder weights, we use a simpler approach:
        // reconstruct mel directly from the codebook lookup.
        val melSpec = reconstructMelFromEmbedding(vqEmbedding)

        // Step 2: Mel spectrogram → linear spectrogram via pseudo-inverse
        // proj is [numBins, nMels], melSpec is [nMels, numFrames]
        // linear[k, t] = sum_m(proj[k, m] * mel[m, t])
        val linearSpec = Array(numBins) { DoubleArray(numFrames) }
        for (k in 0 until numBins) {
            for (t in 0 until numFrames) {
                var sum = 0.0
                for (m in 0 until nMels) {
                    sum += proj[k][m].toDouble() * melSpec[m][t]
                }
                // Clamp to non-negative (magnitudes can't be negative)
                linearSpec[k][t] = max(0.0, sum)
            }
        }

        // Step 3: Griffin-Lim phase reconstruction
        return griffinLim(linearSpec, numFrames)
    }

    /**
     * Reconstruct mel spectrogram from VQ embedding vectors.
     *
     * The VQ embedding [T', 64] represents compressed mel frames. Since we don't
     * have a learned decoder, we use a simple approach: normalize the embedding
     * energy and spread it across mel bins proportionally.
     *
     * The embedding dimension (64) is close to nMels (80), so we can do a
     * straightforward zero-padded mapping with interpolation.
     */
    private fun reconstructMelFromEmbedding(vqEmbedding: Array<FloatArray>): Array<DoubleArray> {
        val numFrames = vqEmbedding.size
        val embDim = if (numFrames > 0) vqEmbedding[0].size else 0
        val melSpec = Array(nMels) { DoubleArray(numFrames) }

        for (t in 0 until numFrames) {
            val emb = vqEmbedding[t]

            // Linear interpolation from embDim → nMels
            for (m in 0 until nMels) {
                val srcIdx = m.toDouble() * (embDim - 1) / (nMels - 1)
                val lo = srcIdx.toInt().coerceIn(0, embDim - 2)
                val hi = lo + 1
                val frac = srcIdx - lo
                val value = emb[lo] * (1.0 - frac) + emb[hi] * frac

                // Convert from embedding space back to power: exp(value) approximation
                // The mel values were originally in [0, 1] normalized dB range
                // We reconstruct approximate power: 10^((value * topDb + clipMin) / 10)
                // Simplified: treat embedding values as normalized mel magnitudes
                melSpec[m][t] = max(0.0, value.toDouble())
            }
        }

        return melSpec
    }

    /**
     * Griffin-Lim iterative phase estimation.
     *
     * Given magnitude-only spectrogram |S|[numBins, T'], estimates phase
     * by alternating between time and frequency domains.
     *
     * @param magnitudes [numBins][numFrames] magnitude spectrogram
     * @param numFrames number of STFT frames
     * @return PCM samples
     */
    private fun griffinLim(magnitudes: Array<DoubleArray>, numFrames: Int): FloatArray {
        // Output signal length
        val signalLen = (numFrames - 1) * hopLength + winLength

        // Initialize with random phase
        val random = java.util.Random(42)
        var phase = Array(numBins) { DoubleArray(numFrames) { random.nextDouble() * 2 * PI } }

        // Iterative Griffin-Lim
        for (iter in 0 until iterations) {
            // Construct complex STFT: S = |S| * exp(j*phase)
            // Inverse STFT → signal
            val signal = istft(magnitudes, phase, signalLen)

            // If last iteration, return the signal
            if (iter == iterations - 1) {
                return normalizeAndConvert(signal)
            }

            // Forward STFT to get new phase estimate
            phase = stftPhase(signal)
        }

        // Should not reach here
        return FloatArray(signalLen)
    }

    /**
     * Inverse STFT: complex spectrogram → time-domain signal via overlap-add.
     */
    private fun istft(
        magnitudes: Array<DoubleArray>,
        phase: Array<DoubleArray>,
        signalLen: Int
    ): DoubleArray {
        val signal = DoubleArray(signalLen)
        val windowSum = DoubleArray(signalLen)
        val numFrames = magnitudes[0].size

        for (t in 0 until numFrames) {
            // Build complex spectrum for this frame
            val complexFrame = DoubleArray(nFft * 2)
            for (k in 0 until numBins) {
                val mag = magnitudes[k][t]
                val phi = phase[k][t]
                val re = mag * cos(phi)
                val im = mag * kotlin.math.sin(phi)

                if (k == 0) {
                    complexFrame[0] = re
                    complexFrame[1] = 0.0
                } else if (k == numBins - 1) {
                    complexFrame[2 * k] = re
                    complexFrame[2 * k + 1] = 0.0
                } else {
                    complexFrame[2 * k] = re
                    complexFrame[2 * k + 1] = im
                    // Conjugate symmetric for negative frequencies
                    complexFrame[2 * (nFft - k)] = re
                    complexFrame[2 * (nFft - k) + 1] = -im
                }
            }

            // Inverse FFT
            fft.complexInverse(complexFrame, true)

            // Extract real part, apply window, overlap-add
            val frameStart = t * hopLength
            for (n in 0 until winLength) {
                val idx = frameStart + n
                if (idx < signalLen) {
                    signal[idx] += complexFrame[2 * n] * hannWindow[n]
                    windowSum[idx] += hannWindow[n] * hannWindow[n]
                }
            }
        }

        // Normalize by window sum (avoid division by zero)
        for (i in signal.indices) {
            if (windowSum[i] > 1e-8) {
                signal[i] /= windowSum[i]
            }
        }

        return signal
    }

    /**
     * Forward STFT to extract phase from a time-domain signal.
     * Returns phase[numBins][numFrames].
     */
    private fun stftPhase(signal: DoubleArray): Array<DoubleArray> {
        val numFrames = (signal.size - winLength) / hopLength + 1
        val phase = Array(numBins) { DoubleArray(numFrames) }

        for (t in 0 until numFrames) {
            val frameStart = t * hopLength
            val windowed = DoubleArray(nFft)
            for (n in 0 until winLength) {
                val idx = frameStart + n
                if (idx < signal.size) {
                    windowed[n] = signal[idx] * hannWindow[n]
                }
            }

            // Forward FFT using realForward
            fft.realForward(windowed)

            // Extract phase from JTransforms packed format:
            // [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ...]
            // Bin 0: DC
            phase[0][t] = if (windowed[0] >= 0) 0.0 else PI

            // Bin N/2: Nyquist
            phase[numBins - 1][t] = if (windowed[1] >= 0) 0.0 else PI

            // Bins 1..N/2-1
            for (k in 1 until numBins - 1) {
                val re = windowed[2 * k]
                val im = windowed[2 * k + 1]
                phase[k][t] = kotlin.math.atan2(im, re)
            }
        }

        return phase
    }

    /**
     * Normalize signal to [-1, 1] range and convert to FloatArray.
     */
    private fun normalizeAndConvert(signal: DoubleArray): FloatArray {
        if (signal.isEmpty()) return FloatArray(0)

        var maxAbs = 0.0
        for (s in signal) {
            val abs = kotlin.math.abs(s)
            if (abs > maxAbs) maxAbs = abs
        }

        val scale = if (maxAbs > 1e-8) 0.95 / maxAbs else 1.0
        return FloatArray(signal.size) { (signal[it] * scale).toFloat().coerceIn(-1f, 1f) }
    }

    /**
     * Compute the mel pseudo-inverse matrix from the mel filterbank.
     *
     * Given mel filterbank M [nMels, numBins], computes:
     *   P = M^T * (M * M^T + eps * I)^{-1}
     * Result is [numBins, nMels].
     *
     * This allows the vocoder to work without a pre-exported projection matrix file.
     */
    private fun computeMelPseudoInverse(melFilterbank: Array<DoubleArray>): Array<FloatArray> {
        val rows = melFilterbank.size        // nMels
        val cols = melFilterbank[0].size     // numBins
        val eps = 1e-8

        Log.i(TAG, "Computing mel pseudo-inverse from filterbank [$rows, $cols]")

        // Compute G = M * M^T  [nMels, nMels]
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

        // Invert G using Gauss-Jordan elimination
        // Augmented matrix [G | I]
        val aug = Array(rows) { i ->
            DoubleArray(2 * rows).also { row ->
                for (j in 0 until rows) row[j] = g[i][j]
                row[rows + i] = 1.0
            }
        }

        for (col in 0 until rows) {
            // Find pivot
            var maxVal = kotlin.math.abs(aug[col][col])
            var maxRow = col
            for (row in col + 1 until rows) {
                val v = kotlin.math.abs(aug[row][col])
                if (v > maxVal) { maxVal = v; maxRow = row }
            }
            if (maxRow != col) {
                val tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp
            }

            val pivot = aug[col][col]
            if (kotlin.math.abs(pivot) < 1e-15) continue

            // Scale pivot row
            for (j in 0 until 2 * rows) aug[col][j] /= pivot

            // Eliminate column in all other rows
            for (row in 0 until rows) {
                if (row == col) continue
                val factor = aug[row][col]
                for (j in 0 until 2 * rows) {
                    aug[row][j] -= factor * aug[col][j]
                }
            }
        }

        // Extract G^{-1} from augmented matrix
        val gInv = Array(rows) { i -> DoubleArray(rows) { j -> aug[i][rows + j] } }

        // Compute P = M^T * G^{-1}  [numBins, nMels]
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

    /**
     * Synthesize from codebook indices only (looks up embeddings from codebook).
     */
    fun synthesizeFromIndices(codebookIndices: IntArray): FloatArray {
        val cb = codebook ?: return FloatArray(0)
        val vqEmbedding = Array(codebookIndices.size) { i ->
            val idx = codebookIndices[i].coerceIn(0, cb.size - 1)
            cb[idx]
        }
        return synthesize(vqEmbedding, codebookIndices)
    }
}
