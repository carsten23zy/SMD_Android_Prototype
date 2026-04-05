package com.edgy.privacy

import com.edgy.privacy.util.DSP
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

/**
 * Unit tests for DSP utilities.
 */
class DSPTest {

    @Test
    fun testPreemphasis() {
        val pcm = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f)
        val coeff = 0.97f

        val (result, lastSample) = DSP.applyPreemphasis(pcm, coeff)

        // y[0] = x[0] - 0.97 * 0 = 0.1
        assertEquals(0.1f, result[0], 1e-6f)
        // y[1] = x[1] - 0.97 * x[0] = 0.2 - 0.097 = 0.103
        assertEquals(0.103f, result[1], 1e-3f)
        // Last sample should be the last input sample
        assertEquals(0.5f, lastSample, 1e-6f)
        assertEquals(pcm.size, result.size)
    }

    @Test
    fun testPreemphasisStreaming() {
        val chunk1 = floatArrayOf(0.1f, 0.2f, 0.3f)
        val chunk2 = floatArrayOf(0.4f, 0.5f)
        val coeff = 0.97f

        val (result1, last1) = DSP.applyPreemphasis(chunk1, coeff, 0f)
        val (result2, last2) = DSP.applyPreemphasis(chunk2, coeff, last1)

        // Verify continuity: first sample of chunk2 uses last sample of chunk1
        val expected = 0.4f - coeff * 0.3f
        assertEquals(expected, result2[0], 1e-6f)
    }

    @Test
    fun testHannWindow() {
        val window = DSP.hannWindow(4)
        assertEquals(4, window.size)
        // Hann window: at n=0, value should be 0 (periodic)
        assertEquals(0f, window[0], 1e-6f)
        // At n=2 (halfway), should be max
        assertTrue(window[2] < 1e-6f) // periodic Hann goes back to ~0
    }

    @Test
    fun testHannWindowSymmetry() {
        val size = 400
        val window = DSP.hannWindow(size)
        // Check that it's roughly symmetric (periodic Hann)
        for (i in 0 until size / 2) {
            val diff = abs(window[i] - window[size - i - 1])
            // Periodic window won't be exactly symmetric, but close for large sizes
            assertTrue("Diff at $i too large: $diff", diff < 0.05f)
        }
    }

    @Test
    fun testFrameSignal() {
        val pcm = FloatArray(10) { it.toFloat() }
        val frames = DSP.frameSignal(pcm, frameLen = 4, hopLen = 2)

        // Number of frames: 1 + (10 - 4) / 2 = 4
        assertEquals(4, frames.size)
        assertArrayEquals(floatArrayOf(0f, 1f, 2f, 3f), frames[0], 1e-6f)
        assertArrayEquals(floatArrayOf(2f, 3f, 4f, 5f), frames[1], 1e-6f)
        assertArrayEquals(floatArrayOf(4f, 5f, 6f, 7f), frames[2], 1e-6f)
        assertArrayEquals(floatArrayOf(6f, 7f, 8f, 9f), frames[3], 1e-6f)
    }

    @Test
    fun testFrameSignalTooShort() {
        val pcm = FloatArray(3) { it.toFloat() }
        val frames = DSP.frameSignal(pcm, frameLen = 4, hopLen = 2)
        assertEquals(0, frames.size)
    }

    @Test
    fun testHzMelConversion() {
        // 1000 Hz should be approximately 1000 mel (HTK scale)
        val mel = DSP.hzToMel(1000.0)
        assertTrue("1000Hz should be ~1000 mel, got $mel", abs(mel - 999.985) < 1.0)

        // Round-trip
        val hz = DSP.melToHz(mel)
        assertEquals(1000.0, hz, 0.1)
    }

    @Test
    fun testShortFloatConversion() {
        val shorts = shortArrayOf(0, 16384, -16384, 32767, -32768)
        val floats = DSP.shortToFloat(shorts)

        assertEquals(0f, floats[0], 1e-4f)
        assertEquals(0.5f, floats[1], 1e-3f)
        assertEquals(-0.5f, floats[2], 1e-3f)

        // Round-trip
        val backShorts = DSP.floatToShort(floats)
        for (i in shorts.indices) {
            assertTrue("Mismatch at $i", abs(shorts[i] - backShorts[i]) <= 1)
        }
    }

    @Test
    fun testPadReflect() {
        val signal = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val padded = DSP.padReflect(signal, 2)

        // Expected: [3, 2, 1, 2, 3, 4, 5, 4, 3]
        assertEquals(9, padded.size)
        assertEquals(3f, padded[0], 1e-6f) // Reflect: signal[2]
        assertEquals(2f, padded[1], 1e-6f) // Reflect: signal[1]
        assertEquals(1f, padded[2], 1e-6f) // Original start
        assertEquals(5f, padded[6], 1e-6f) // Original end
        assertEquals(4f, padded[7], 1e-6f) // Reflect: signal[3]
        assertEquals(3f, padded[8], 1e-6f) // Reflect: signal[2]
    }
}
