package com.edgy.privacy

import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.*
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Unit tests for MelSpectrogramExtractor.
 * Tests basic functionality and properties that must hold for librosa parity.
 */
class MelSpectrogramExtractorTest {

    private lateinit var extractor: MelSpectrogramExtractor
    private lateinit var config: ModelConfig

    @Before
    fun setUp() {
        config = ModelConfig(
            preprocessing = PreprocessingConfig(
                sampleRate = 16000,
                nFft = 2048,
                nMels = 80,
                hopLength = 160,
                winLength = 400,
                fmin = 50,
                preemph = 0.97f,
                topDb = 80
            ),
            encoder = EncoderConfig(
                nEmbeddings = 512,
                embeddingDim = 64,
                downsamplingFactor = 2
            ),
            files = FilesConfig(
                encoderFp32 = "edgy_encoder_fp32.onnx",
                encoderInt8 = "edgy_encoder_int8.onnx",
                codebook = "vq_codebook.npy",
                speakerEmbeddings = "speaker_embeddings.npy"
            ),
            onnxSettings = OnnxSettings(
                intraOpNumThreads = 4,
                interOpNumThreads = 2,
                executionMode = "ORT_PARALLEL",
                optimizationLevel = "ALL_OPT"
            )
        )
        extractor = MelSpectrogramExtractor(config)
    }

    @Test
    fun testOutputShape() {
        // 1 second of audio at 16kHz = 16000 samples
        // With center padding and hop_length=160:
        // T = ceil(16000 / 160) = 100 frames
        val pcm = generateSine(440.0, 1.0)
        val mel = extractor.extract(pcm)

        assertEquals("Should have 80 mel bins", 80, mel.size)
        assertTrue("Should have ~100 frames", mel[0].size in 95..105)
    }

    @Test
    fun testOutputRange() {
        // After normalization: values should be in [0, 1]
        val pcm = generateSine(440.0, 1.0)
        val mel = extractor.extract(pcm)

        for (m in mel.indices) {
            for (t in mel[m].indices) {
                assertTrue(
                    "Mel[$m][$t] = ${mel[m][t]} should be >= 0",
                    mel[m][t] >= -0.01f
                )
                assertTrue(
                    "Mel[$m][$t] = ${mel[m][t]} should be <= 1",
                    mel[m][t] <= 1.01f
                )
            }
        }
    }

    @Test
    fun testDeterministic() {
        // Same input should produce same output
        val pcm = generateSine(440.0, 0.5)
        val mel1 = extractor.extract(pcm)
        val mel2 = extractor.extract(pcm)

        assertEquals(mel1.size, mel2.size)
        for (m in mel1.indices) {
            assertArrayEquals(
                "Mel bin $m should be identical",
                mel1[m], mel2[m], 0f
            )
        }
    }

    @Test
    fun testEmptyInput() {
        val mel = extractor.extract(FloatArray(0))
        assertEquals(80, mel.size)
        assertEquals(0, mel[0].size)
    }

    @Test
    fun testShortInput() {
        // Very short input (less than one window)
        val pcm = FloatArray(100) { 0.1f }
        val mel = extractor.extract(pcm)
        assertEquals(80, mel.size)
        // Should still produce some frames due to center padding
    }

    @Test
    fun testSilenceVsTone() {
        // Silence should have lower energy than a tone
        val silence = FloatArray(16000) { 0f }
        val tone = generateSine(440.0, 1.0)

        val melSilence = extractor.extract(silence)
        val melTone = extractor.extract(tone)

        // Sum all mel values
        var sumSilence = 0.0
        var sumTone = 0.0
        for (m in melSilence.indices) {
            for (t in melSilence[m].indices) {
                sumSilence += melSilence[m][t]
            }
        }
        for (m in melTone.indices) {
            for (t in melTone[m].indices) {
                sumTone += melTone[m][t]
            }
        }

        assertTrue(
            "Tone should have higher mel energy than silence",
            sumTone > sumSilence
        )
    }

    @Test
    fun testFrequencyDiscrimination() {
        // Low frequency tone should have more energy in lower mel bins
        val lowTone = generateSine(200.0, 1.0)
        val highTone = generateSine(4000.0, 1.0)

        val melLow = extractor.extract(lowTone)
        val melHigh = extractor.extract(highTone)

        // Sum energy in lower quarter of mel bins
        var lowBinEnergyLow = 0.0
        var lowBinEnergyHigh = 0.0
        val quarterBins = 80 / 4
        for (m in 0 until quarterBins) {
            for (t in melLow[m].indices) {
                lowBinEnergyLow += melLow[m][t]
            }
            for (t in melHigh[m].indices) {
                lowBinEnergyHigh += melHigh[m][t]
            }
        }

        assertTrue(
            "Low tone should have more energy in low mel bins",
            lowBinEnergyLow > lowBinEnergyHigh
        )
    }

    @Test
    fun testStreamingResetState() {
        val chunk = generateSine(440.0, 0.1) // 100ms
        extractor.extractStreaming(chunk)
        extractor.resetStreaming()
        // After reset, should work like fresh extractor
        val mel = extractor.extractStreaming(chunk)
        assertEquals(80, mel.size)
    }

    /**
     * Generate a sine wave for testing.
     */
    private fun generateSine(frequency: Double, durationSec: Double): FloatArray {
        val sampleRate = config.preprocessing.sampleRate
        val numSamples = (sampleRate * durationSec).toInt()
        return FloatArray(numSamples) { i ->
            (0.5 * sin(2.0 * Math.PI * frequency * i / sampleRate)).toFloat()
        }
    }
}
