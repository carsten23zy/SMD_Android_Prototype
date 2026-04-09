package com.edgy.privacy.util

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Signal processing utilities for audio preprocessing.
 * These functions replicate librosa's behavior for exact parity with the Python pipeline.
 */
object DSP {

    /**
     * Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
     * Returns the filtered signal and the last sample (for streaming continuity).
     */
    fun applyPreemphasis(
        pcm: FloatArray,
        coeff: Float,
        lastSample: Float = 0f
    ): Pair<FloatArray, Float> {
        if (pcm.isEmpty()) return Pair(FloatArray(0), lastSample)

        val result = FloatArray(pcm.size)
        result[0] = pcm[0] - coeff * lastSample
        for (i in 1 until pcm.size) {
            result[i] = pcm[i] - coeff * pcm[i - 1]
        }
        return Pair(result, pcm.last())
    }

    /**
     * Generate a Hann window of the given size.
     * Matches numpy.hanning() / scipy.signal.hann(): periodic Hann window.
     */
    fun hannWindow(size: Int): FloatArray {
        return FloatArray(size) { n ->
            (0.5 * (1.0 - cos(2.0 * PI * n / size))).toFloat()
        }
    }

    /**
     * Frame a signal into overlapping frames.
     * Each frame has length frameLen, with hopLen samples between frame starts.
     * Matches librosa.util.frame() behavior (center=False).
     */
    fun frameSignal(pcm: FloatArray, frameLen: Int, hopLen: Int): Array<FloatArray> {
        if (pcm.size < frameLen) return emptyArray()

        val numFrames = 1 + (pcm.size - frameLen) / hopLen
        return Array(numFrames) { i ->
            val start = i * hopLen
            pcm.copyOfRange(start, start + frameLen)
        }
    }

    /**
     * Convert frequency in Hz to mel scale.
     * Uses the HTK formula: mel = 2595 * log10(1 + hz / 700)
     */
    fun hzToMel(hz: Double): Double {
        return 2595.0 * Math.log10(1.0 + hz / 700.0)
    }

    /**
     * Convert mel scale to frequency in Hz.
     * Inverse of hzToMel.
     */
    fun melToHz(mel: Double): Double {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0)
    }

    /**
     * Convert 16-bit PCM short samples to float [-1.0, 1.0].
     */
    fun shortToFloat(pcm: ShortArray): FloatArray {
        return FloatArray(pcm.size) { pcm[it].toFloat() / 32768f }
    }

    /**
     * Convert float [-1.0, 1.0] PCM to 16-bit short samples.
     */
    fun floatToShort(pcm: FloatArray): ShortArray {
        return ShortArray(pcm.size) {
            val clamped = pcm[it].coerceIn(-1f, 1f)
            (clamped * 32767f).toInt().toShort()
        }
    }

    /**
     * Resample audio from one sample rate to another using windowed sinc interpolation.
     *
     * For downsampling (targetRate < sourceRate), applies a low-pass filter at
     * targetRate/2 to prevent aliasing. Uses a sinc kernel with Hann window.
     *
     * @param pcm Input samples
     * @param sourceRate Original sample rate (e.g. 96000)
     * @param targetRate Desired sample rate (e.g. 16000)
     * @return Resampled audio
     */
    fun resample(pcm: FloatArray, sourceRate: Int, targetRate: Int): FloatArray {
        if (sourceRate == targetRate) return pcm
        if (pcm.isEmpty()) return pcm

        val ratio = targetRate.toDouble() / sourceRate
        val outputLen = (pcm.size * ratio).toInt()
        val result = FloatArray(outputLen)

        // Low-pass cutoff: min of the two Nyquist frequencies
        val cutoff = minOf(0.5, ratio / 2.0)
        // Sinc kernel half-width in input samples
        val halfWidth = 16
        val scale = 2.0 * cutoff

        for (i in 0 until outputLen) {
            val srcPos = i / ratio
            val srcCenter = srcPos.toInt()
            var sum = 0.0
            var weightSum = 0.0

            val jStart = maxOf(0, srcCenter - halfWidth)
            val jEnd = minOf(pcm.size - 1, srcCenter + halfWidth)

            for (j in jStart..jEnd) {
                val x = srcPos - j
                // Windowed sinc
                val sincVal = if (kotlin.math.abs(x) < 1e-8) {
                    scale
                } else {
                    val piX = PI * x
                    scale * sin(scale * piX) / (scale * piX)
                }
                // Hann window over the kernel
                val winPos = (j - srcCenter + halfWidth).toDouble() / (2 * halfWidth)
                val window = 0.5 * (1.0 - cos(2.0 * PI * winPos))
                val w = sincVal * window
                sum += pcm[j] * w
                weightSum += w
            }

            result[i] = if (weightSum > 1e-8) (sum / weightSum).toFloat() else 0f
        }
        return result
    }

    /**
     * Pad signal with reflection padding (matching numpy's reflect mode).
     * Pads padSize samples on each side. Handles padSize >= signal.size
     * by wrapping reflections multiple times.
     */
    fun padReflect(signal: FloatArray, padSize: Int): FloatArray {
        if (signal.isEmpty()) return FloatArray(0)
        if (signal.size == 1) return FloatArray(1 + 2 * padSize) { signal[0] }

        val result = FloatArray(signal.size + 2 * padSize)
        for (i in result.indices) {
            result[i] = signal[reflectIndex(i - padSize, signal.size)]
        }
        return result
    }

    /**
     * Map an index into [0, size-1] using reflection at boundaries.
     */
    private fun reflectIndex(idx: Int, size: Int): Int {
        if (size <= 1) return 0
        val period = 2 * (size - 1)
        var i = idx % period
        if (i < 0) i += period
        return if (i < size) i else period - i
    }
}
