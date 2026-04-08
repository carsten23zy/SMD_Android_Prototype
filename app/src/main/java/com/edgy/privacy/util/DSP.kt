package com.edgy.privacy.util

import kotlin.math.PI
import kotlin.math.cos

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
