package com.edgy.privacy

import com.edgy.privacy.ml.EncoderConfig
import com.edgy.privacy.ml.FilesConfig
import com.edgy.privacy.ml.ModelConfig
import com.edgy.privacy.ml.OnnxSettings
import com.edgy.privacy.ml.PreprocessingConfig
import com.edgy.privacy.vocoder.GriffinLimVocoder
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for Griffin-Lim vocoder.
 */
class GriffinLimVocoderTest {

    private lateinit var config: ModelConfig

    // Simple projection matrix [numBins=1025, nMels=80]
    private lateinit var projectionMatrix: Array<FloatArray>

    // Simple codebook [512, 64]
    private lateinit var codebook: Array<FloatArray>

    @Before
    fun setup() {
        config = ModelConfig(
            preprocessing = PreprocessingConfig(
                sampleRate = 16000,
                nFft = 2048,
                nMels = 80,
                hopLength = 160,
                winLength = 400,
                fmin = 0,
                preemph = 0.97f,
                topDb = 80
            ),
            encoder = EncoderConfig(
                nEmbeddings = 512,
                embeddingDim = 64,
                downsamplingFactor = 2
            ),
            files = FilesConfig(
                encoderFp32 = "encoder.onnx",
                encoderInt8 = "encoder_int8.onnx",
                codebook = "codebook.npy",
                speakerEmbeddings = "speaker_embeddings.npy",
                projectionMatrix = "mel_pseudo_inverse.npy"
            ),
            onnxSettings = OnnxSettings(
                intraOpNumThreads = 2,
                interOpNumThreads = 1,
                executionMode = "SEQUENTIAL",
                optimizationLevel = "ALL_OPT"
            )
        )

        val numBins = 1025
        val nMels = 80

        // Identity-like projection (scaled down for stability)
        projectionMatrix = Array(numBins) { k ->
            FloatArray(nMels) { m ->
                if (k % (numBins / nMels) == m) 0.01f else 0.0f
            }
        }

        codebook = Array(512) { i ->
            FloatArray(64) { d -> ((i * 64 + d) % 100) / 100f }
        }
    }

    @Test
    fun `vocoder isAvailable when projection and codebook present`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        assertTrue(vocoder.isAvailable)
    }

    @Test
    fun `vocoder unavailable without projection matrix`() {
        val vocoder = GriffinLimVocoder(config, null, codebook)
        assertFalse(vocoder.isAvailable)
    }

    @Test
    fun `vocoder unavailable without codebook`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, null)
        assertFalse(vocoder.isAvailable)
    }

    @Test
    fun `synthesize returns empty when unavailable`() {
        val vocoder = GriffinLimVocoder(config, null, null)
        val emb = Array(10) { FloatArray(64) { 0.5f } }
        val result = vocoder.synthesize(emb)
        assertTrue(result.isEmpty())
    }

    @Test
    fun `synthesize returns non-empty audio from valid embedding`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val numFrames = 20
        val embedding = Array(numFrames) { FloatArray(64) { 0.3f } }
        val audio = vocoder.synthesize(embedding)
        assertTrue("Expected non-empty audio, got ${audio.size} samples", audio.isNotEmpty())
    }

    @Test
    fun `synthesized audio is within valid range`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val embedding = Array(10) { FloatArray(64) { 0.5f } }
        val audio = vocoder.synthesize(embedding)
        for (sample in audio) {
            assertTrue("Sample $sample out of range", sample in -1f..1f)
        }
    }

    @Test
    fun `synthesizeFromIndices uses codebook lookup`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val indices = IntArray(10) { it % 512 }
        val audio = vocoder.synthesizeFromIndices(indices)
        assertTrue("Expected non-empty audio from indices", audio.isNotEmpty())
        for (sample in audio) {
            assertTrue("Sample $sample out of range", sample in -1f..1f)
        }
    }

    @Test
    fun `synthesizeFromIndices clamps out-of-range indices`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val indices = intArrayOf(-1, 0, 511, 999, 5)
        // Should not throw — indices are clamped
        val audio = vocoder.synthesizeFromIndices(indices)
        assertTrue(audio.isNotEmpty())
    }

    @Test
    fun `output length scales with number of frames`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val small = vocoder.synthesize(Array(5) { FloatArray(64) { 0.3f } })
        val large = vocoder.synthesize(Array(50) { FloatArray(64) { 0.3f } })
        assertTrue("Larger input should produce longer audio", large.size > small.size)
    }

    @Test
    fun `different embeddings produce different audio`() {
        val vocoder = GriffinLimVocoder(config, projectionMatrix, codebook)
        val emb1 = Array(10) { FloatArray(64) { 0.2f } }
        val emb2 = Array(10) { FloatArray(64) { 0.8f } }
        val audio1 = vocoder.synthesize(emb1)
        val audio2 = vocoder.synthesize(emb2)

        // They should differ (not all zeros or identical)
        val minLen = minOf(audio1.size, audio2.size)
        var maxDiff = 0f
        for (i in 0 until minLen) {
            val diff = kotlin.math.abs(audio1[i] - audio2[i])
            if (diff > maxDiff) maxDiff = diff
        }
        assertTrue("Different embeddings should produce different audio (maxDiff=$maxDiff)", maxDiff > 0.001f)
    }
}
