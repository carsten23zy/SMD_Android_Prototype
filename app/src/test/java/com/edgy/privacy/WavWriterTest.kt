package com.edgy.privacy

import com.edgy.privacy.util.WavWriter
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.math.abs
import kotlin.math.sin

/**
 * Unit tests for WavWriter.
 */
class WavWriterTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testWriteAndReadRoundTrip() {
        val sampleRate = 16000
        val duration = 0.5
        val numSamples = (sampleRate * duration).toInt()
        val pcm = FloatArray(numSamples) { i ->
            (0.5 * sin(2.0 * Math.PI * 440.0 * i / sampleRate)).toFloat()
        }

        val file = tempFolder.newFile("test.wav")
        WavWriter.writeWav(file.absolutePath, pcm, sampleRate)

        assertTrue("WAV file should exist", file.exists())
        assertTrue("WAV file should be > 44 bytes (header)", file.length() > 44)

        // Read back
        val (readSampleRate, readPcm) = WavWriter.readWav(file.absolutePath)
        assertEquals(sampleRate, readSampleRate)
        assertEquals(numSamples, readPcm.size)

        // Check values are close (16-bit quantization introduces small error)
        for (i in pcm.indices) {
            assertTrue(
                "Sample $i: expected ${pcm[i]}, got ${readPcm[i]}",
                abs(pcm[i] - readPcm[i]) < 0.001f
            )
        }
    }

    @Test
    fun testWriteEmptyAudio() {
        val file = tempFolder.newFile("empty.wav")
        WavWriter.writeWav(file.absolutePath, FloatArray(0))
        assertTrue(file.exists())
        assertEquals(44L, file.length()) // Just the header
    }

    @Test
    fun testBitIdenticalPassthrough() {
        // Stage 2 verification: LOW tier output should be bit-identical to input
        val pcm = FloatArray(1600) { i ->
            (0.3 * sin(2.0 * Math.PI * 440.0 * i / 16000)).toFloat()
        }

        val file1 = tempFolder.newFile("input.wav")
        val file2 = tempFolder.newFile("output.wav")

        WavWriter.writeWav(file1.absolutePath, pcm, 16000)
        val (_, readPcm) = WavWriter.readWav(file1.absolutePath)

        // Write the read PCM back (simulating LOW tier passthrough)
        WavWriter.writeWav(file2.absolutePath, readPcm, 16000)
        val (_, readPcm2) = WavWriter.readWav(file2.absolutePath)

        // Should be bit-identical
        assertArrayEquals(readPcm, readPcm2, 0f)
    }
}
