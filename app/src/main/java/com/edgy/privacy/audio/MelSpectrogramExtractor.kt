package com.edgy.privacy.audio

import com.edgy.privacy.ml.ModelConfig
import com.edgy.privacy.util.DSP
import org.jtransforms.fft.DoubleFFT_1D
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow

/**
 * Converts raw 16kHz PCM into log-mel spectrograms matching EDGY's Python preprocessing.
 *
 * This component must produce numerically identical output to Python's librosa.
 * If it doesn't, the ONNX encoder receives out-of-distribution input and all
 * downstream output is invalid.
 *
 * Pipeline: PCM → preemphasis → frame → window → FFT → magnitude → mel filterbank → log → normalize
 */
class MelSpectrogramExtractor(private val config: ModelConfig) {

    private val sampleRate = config.preprocessing.sampleRate
    private val nFft = config.preprocessing.nFft
    private val nMels = config.preprocessing.nMels
    private val hopLength = config.preprocessing.hopLength
    private val winLength = config.preprocessing.winLength
    private val fmin = config.preprocessing.fmin.toDouble()
    private val preemphCoeff = config.preprocessing.preemph
    private val topDb = config.preprocessing.topDb.toDouble()

    // Precomputed mel filterbank [nMels, nFft/2 + 1]
    private val melFilterbank: Array<DoubleArray> = buildMelFilterbank()

    // Precomputed Hann window
    private val hannWindow: DoubleArray = DoubleArray(winLength) { n ->
        0.5 * (1.0 - kotlin.math.cos(2.0 * PI * n / winLength))
    }

    // FFT instance (reusable, thread-local for safety)
    private val fft = DoubleFFT_1D(nFft.toLong())

    // Streaming state
    private var overlapBuffer = FloatArray(0)
    private var lastPreemphSample = 0f

    /**
     * Extract mel spectrogram from a complete PCM signal.
     *
     * @param pcm Float PCM samples in [-1, 1] range
     * @return Mel spectrogram as [nMels][T] array (transposed from [T, nMels])
     */
    fun extract(pcm: FloatArray): Array<FloatArray> {
        if (pcm.isEmpty()) return Array(nMels) { FloatArray(0) }

        // 1. Pre-emphasis
        val (preemph, _) = DSP.applyPreemphasis(pcm, preemphCoeff)

        // 2. Center-pad the signal (librosa default: center=True with reflect padding)
        val padSize = nFft / 2
        val padded = DSP.padReflect(preemph, padSize)

        // 3. Frame the signal (use nFft as frame length, matching librosa.stft)
        val frames = DSP.frameSignal(padded, nFft, hopLength)
        if (frames.isEmpty()) return Array(nMels) { FloatArray(0) }

        val numFrames = frames.size
        val melSpec = Array(nMels) { DoubleArray(numFrames) }
        val winOffset = (nFft - winLength) / 2

        for (t in frames.indices) {
            // 4. Apply centered Hann window (librosa center-pads the window to nFft)
            val windowed = DoubleArray(nFft)
            for (i in 0 until winLength) {
                windowed[winOffset + i] = frames[t][winOffset + i].toDouble() * hannWindow[i]
            }
            // Zero-pad to nFft (already done by allocating nFft size)

            // 5. FFT → magnitude spectrum
            fft.realForward(windowed)
            val magnitudes = fftToMagnitude(windowed)

            // 6. Apply mel filterbank
            for (m in 0 until nMels) {
                var sum = 0.0
                val filter = melFilterbank[m]
                for (k in filter.indices) {
                    sum += filter[k] * magnitudes[k]
                }
                melSpec[m][t] = sum
            }
        }

        // 7. Convert to dB: 10 * log10(mel + 1e-10)
        val melDb = Array(nMels) { DoubleArray(numFrames) }
        var globalMax = Double.NEGATIVE_INFINITY
        for (m in 0 until nMels) {
            for (t in 0 until numFrames) {
                melDb[m][t] = 10.0 * log10(melSpec[m][t] + 1e-10)
                if (melDb[m][t] > globalMax) globalMax = melDb[m][t]
            }
        }

        // 8. If the signal has no meaningful energy (silence), return zeros
        //    to avoid normalization artifact where silence maps to all-ones
        if (globalMax < -99.0) {
            return Array(nMels) { FloatArray(numFrames) }
        }

        // 9. Clip: max(mel_db, globalMax - top_db)
        val clipMin = globalMax - topDb
        for (m in 0 until nMels) {
            for (t in 0 until numFrames) {
                melDb[m][t] = max(melDb[m][t], clipMin)
            }
        }

        // 10. Normalize: mel_db / top_db + 1
        val result = Array(nMels) { m ->
            FloatArray(numFrames) { t ->
                ((melDb[m][t] - clipMin) / topDb).toFloat()
            }
        }

        return result
    }

    /**
     * Extract mel spectrogram in streaming mode.
     * Maintains overlap buffer across calls for continuity.
     *
     * @param pcmChunk PCM chunk (e.g., 1600 samples for 100ms at 16kHz)
     * @return Mel spectrogram for this chunk [nMels][T']
     */
    fun extractStreaming(pcmChunk: FloatArray): Array<FloatArray> {
        // Concatenate overlap buffer with new chunk
        val combined = FloatArray(overlapBuffer.size + pcmChunk.size)
        System.arraycopy(overlapBuffer, 0, combined, 0, overlapBuffer.size)
        System.arraycopy(pcmChunk, 0, combined, overlapBuffer.size, pcmChunk.size)

        // Apply preemphasis with state
        val (preemph, newLastSample) = DSP.applyPreemphasis(combined, preemphCoeff, lastPreemphSample)
        lastPreemphSample = newLastSample

        // Center-pad
        val padSize = nFft / 2
        val padded = DSP.padReflect(preemph, padSize)

        // Frame (use nFft as frame length, matching librosa.stft)
        val frames = DSP.frameSignal(padded, nFft, hopLength)
        if (frames.isEmpty()) {
            overlapBuffer = combined
            return Array(nMels) { FloatArray(0) }
        }

        val numFrames = frames.size
        val melSpec = Array(nMels) { DoubleArray(numFrames) }
        val winOffset = (nFft - winLength) / 2

        for (t in frames.indices) {
            val windowed = DoubleArray(nFft)
            for (i in 0 until winLength) {
                windowed[winOffset + i] = frames[t][winOffset + i].toDouble() * hannWindow[i]
            }

            fft.realForward(windowed)
            val magnitudes = fftToMagnitude(windowed)

            for (m in 0 until nMels) {
                var sum = 0.0
                val filter = melFilterbank[m]
                for (k in filter.indices) {
                    sum += filter[k] * magnitudes[k]
                }
                melSpec[m][t] = sum
            }
        }

        // Convert to dB, clip, normalize
        var globalMax = Double.NEGATIVE_INFINITY
        val melDb = Array(nMels) { DoubleArray(numFrames) }
        for (m in 0 until nMels) {
            for (t in 0 until numFrames) {
                melDb[m][t] = 10.0 * log10(melSpec[m][t] + 1e-10)
                if (melDb[m][t] > globalMax) globalMax = melDb[m][t]
            }
        }

        // Silence threshold
        if (globalMax < -99.0) {
            val overlapSize = nFft - hopLength
            if (combined.size > overlapSize) {
                overlapBuffer = combined.copyOfRange(combined.size - overlapSize, combined.size)
            } else {
                overlapBuffer = combined
            }
            return Array(nMels) { FloatArray(numFrames) }
        }

        val clipMin = globalMax - topDb
        val result = Array(nMels) { m ->
            FloatArray(numFrames) { t ->
                val clipped = max(melDb[m][t], clipMin)
                ((clipped - clipMin) / topDb).toFloat()
            }
        }

        // Save overlap: keep last (nFft - hopLength) samples from the original combined
        val overlapSize = nFft - hopLength
        if (combined.size > overlapSize) {
            overlapBuffer = combined.copyOfRange(combined.size - overlapSize, combined.size)
        } else {
            overlapBuffer = combined
        }

        return result
    }

    /**
     * Reset streaming state.
     */
    fun resetStreaming() {
        overlapBuffer = FloatArray(0)
        lastPreemphSample = 0f
    }

    /**
     * Convert JTransforms realForward output to magnitude spectrum.
     * JTransforms packs the result as: [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ...]
     * We need magnitudes for bins 0..N/2 (inclusive), giving N/2+1 = 1025 bins for N=2048.
     */
    private fun fftToMagnitude(fftData: DoubleArray): DoubleArray {
        val numBins = nFft / 2 + 1
        val magnitudes = DoubleArray(numBins)

        // Bin 0: DC component (purely real)
        magnitudes[0] = abs(fftData[0])

        // Bin N/2: Nyquist (purely real)
        magnitudes[numBins - 1] = abs(fftData[1])

        // Bins 1..N/2-1: complex values
        for (k in 1 until numBins - 1) {
            val re = fftData[2 * k]
            val im = fftData[2 * k + 1]
            magnitudes[k] = kotlin.math.sqrt(re * re + im * im)
        }

        return magnitudes
    }

    /**
     * Build mel filterbank matrix [nMels, nFft/2 + 1].
     * Replicates librosa.filters.mel() with HTK mel scale.
     */
    private fun buildMelFilterbank(): Array<DoubleArray> {
        val fmax = sampleRate / 2.0
        val numBins = nFft / 2 + 1

        // Mel scale boundaries
        val melMin = DSP.hzToMel(fmin)
        val melMax = DSP.hzToMel(fmax)

        // nMels + 2 equally spaced points in mel scale
        val melPoints = DoubleArray(nMels + 2) { i ->
            melMin + i * (melMax - melMin) / (nMels + 1)
        }

        // Convert back to Hz
        val hzPoints = DoubleArray(melPoints.size) { DSP.melToHz(melPoints[it]) }

        // Convert to FFT bin indices (continuous, not rounded)
        val binPoints = DoubleArray(hzPoints.size) { i ->
            hzPoints[i] * nFft / sampleRate
        }

        // Build triangular filters
        val filterbank = Array(nMels) { DoubleArray(numBins) }

        for (m in 0 until nMels) {
            val left = binPoints[m]
            val center = binPoints[m + 1]
            val right = binPoints[m + 2]

            for (k in 0 until numBins) {
                val kd = k.toDouble()
                filterbank[m][k] = when {
                    kd < left -> 0.0
                    kd <= center -> (kd - left) / (center - left)
                    kd <= right -> (right - kd) / (right - center)
                    else -> 0.0
                }
            }

            // Normalize: slaney normalization (librosa default)
            val enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m])
            for (k in 0 until numBins) {
                filterbank[m][k] *= enorm
            }
        }

        return filterbank
    }
}
