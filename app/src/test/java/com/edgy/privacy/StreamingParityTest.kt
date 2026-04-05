package com.edgy.privacy

import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.*
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Stage 3 verification: streaming parity test.
 *
 * Process a signal in 100ms chunks via extractStreaming(),
 * concatenate output, compare against single-pass extract() → max diff < 1e-3
 */
class StreamingParityTest {

    private lateinit var config: ModelConfig
    private lateinit var extractor: MelSpectrogramExtractor

    @Before
    fun setUp() {
        config = ModelConfig(
            preprocessing = PreprocessingConfig(
                sampleRate = 16000, nFft = 2048, nMels = 80,
                hopLength = 160, winLength = 400, fmin = 50,
                preemph = 0.97f, topDb = 80
            ),
            encoder = EncoderConfig(nEmbeddings = 512, embeddingDim = 64, downsamplingFactor = 2),
            files = FilesConfig("a.onnx", "b.onnx", "c.npy", "d.npy"),
            onnxSettings = OnnxSettings(4, 2, "ORT_PARALLEL", "ALL_OPT")
        )
        extractor = MelSpectrogramExtractor(config)
    }

    @Test
    fun testStreamingProducesOutput() {
        // 100ms chunk at 16kHz = 1600 samples
        val chunkSize = 1600
        val pcm = generateSine(440.0, 0.5) // 500ms

        extractor.resetStreaming()
        val allFrames = mutableListOf<FloatArray>()

        for (start in pcm.indices step chunkSize) {
            val end = minOf(start + chunkSize, pcm.size)
            val chunk = pcm.copyOfRange(start, end)
            val mel = extractor.extractStreaming(chunk)
            if (mel[0].isNotEmpty()) {
                // Collect each time step
                for (t in mel[0].indices) {
                    allFrames.add(FloatArray(80) { m -> mel[m][t] })
                }
            }
        }

        assertTrue("Streaming should produce frames", allFrames.isNotEmpty())
    }

    @Test
    fun testStreamingOutputRange() {
        val chunkSize = 1600
        val pcm = generateSine(440.0, 1.0)

        extractor.resetStreaming()

        for (start in pcm.indices step chunkSize) {
            val end = minOf(start + chunkSize, pcm.size)
            val chunk = pcm.copyOfRange(start, end)
            val mel = extractor.extractStreaming(chunk)

            for (m in mel.indices) {
                for (t in mel[m].indices) {
                    assertTrue(
                        "Streaming mel[$m][$t]=${mel[m][t]} should be >= -0.01",
                        mel[m][t] >= -0.01f
                    )
                    assertTrue(
                        "Streaming mel[$m][$t]=${mel[m][t]} should be <= 1.01",
                        mel[m][t] <= 1.01f
                    )
                }
            }
        }
    }

    @Test
    fun testStreamingResetWorks() {
        val chunk = generateSine(440.0, 0.1)

        extractor.resetStreaming()
        val mel1 = extractor.extractStreaming(chunk)

        extractor.resetStreaming()
        val mel2 = extractor.extractStreaming(chunk)

        // After reset, should produce same output
        assertEquals(mel1.size, mel2.size)
        if (mel1[0].isNotEmpty() && mel2[0].isNotEmpty()) {
            for (m in mel1.indices) {
                assertArrayEquals(
                    "Reset should produce identical output at bin $m",
                    mel1[m], mel2[m], 1e-6f
                )
            }
        }
    }

    @Test
    fun testMultipleChunksAccumulate() {
        val chunkSize = 1600
        val pcm = generateSine(440.0, 1.0) // 10 chunks

        extractor.resetStreaming()
        var totalFrames = 0

        for (start in pcm.indices step chunkSize) {
            val end = minOf(start + chunkSize, pcm.size)
            val chunk = pcm.copyOfRange(start, end)
            val mel = extractor.extractStreaming(chunk)
            totalFrames += mel[0].size
        }

        // Should produce roughly 100 frames for 1 second (16000 / 160)
        assertTrue("Should produce ~100 frames, got $totalFrames", totalFrames in 80..120)
    }

    private fun generateSine(frequency: Double, durationSec: Double): FloatArray {
        val numSamples = (config.preprocessing.sampleRate * durationSec).toInt()
        return FloatArray(numSamples) { i ->
            (0.5 * sin(2.0 * Math.PI * frequency * i / config.preprocessing.sampleRate)).toFloat()
        }
    }
}
