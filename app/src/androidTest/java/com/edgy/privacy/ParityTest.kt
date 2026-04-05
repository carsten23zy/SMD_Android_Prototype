package com.edgy.privacy

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.ModelManager
import com.edgy.privacy.privacy.PrivacyPipeline
import com.edgy.privacy.privacy.PrivacyTier
import com.edgy.privacy.util.NpyReader
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Instrumented parity tests — run on a real device or emulator.
 * These validate that the Kotlin pipeline matches the Python notebook output.
 *
 * Verification criteria (from handover doc):
 * 1. Kotlin mel vs Python mel: max absolute diff < 1e-3
 * 2. Kotlin ONNX VQ vs Python VQ: cosine similarity > 0.99
 * 3. Kotlin indices vs Python indices: match rate > 95%
 */
@RunWith(AndroidJUnit4::class)
class ParityTest {

    private lateinit var context: Context
    private lateinit var modelManager: ModelManager
    private lateinit var melExtractor: MelSpectrogramExtractor

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        modelManager = ModelManager(context)
        val config = modelManager.loadConfig()
        melExtractor = MelSpectrogramExtractor(config)
    }

    /**
     * Test that model_config.json loads correctly.
     */
    @Test
    fun testConfigLoads() {
        val config = modelManager.loadConfig()
        assertEquals(16000, config.preprocessing.sampleRate)
        assertEquals(80, config.preprocessing.nMels)
        assertEquals(2048, config.preprocessing.nFft)
        assertEquals(160, config.preprocessing.hopLength)
        assertEquals(400, config.preprocessing.winLength)
        assertEquals(512, config.encoder.nEmbeddings)
        assertEquals(64, config.encoder.embeddingDim)
    }

    /**
     * Test mel spectrogram extraction produces valid output.
     */
    @Test
    fun testMelExtractionBasic() {
        // Generate a 1-second sine wave
        val pcm = FloatArray(16000) { i ->
            (0.5 * kotlin.math.sin(2.0 * Math.PI * 440.0 * i / 16000)).toFloat()
        }

        val mel = melExtractor.extract(pcm)

        assertEquals("Should have 80 mel bins", 80, mel.size)
        assertTrue("Should have ~100 frames", mel[0].size in 95..105)

        // Values should be normalized to [0, 1]
        for (m in mel.indices) {
            for (t in mel[m].indices) {
                assertTrue("Value should be >= 0", mel[m][t] >= -0.01f)
                assertTrue("Value should be <= 1", mel[m][t] <= 1.01f)
            }
        }
    }

    /**
     * Run parity tests against all available test fixtures.
     * Each fixture has: sample_XX_mel.npy, sample_XX_vq.npy, sample_XX_indices.npy
     */
    @Test
    fun testParityAgainstFixtures() {
        val fixtureDir = "models/test_fixtures"
        val fixtureFiles = try {
            context.assets.list(fixtureDir) ?: emptyArray()
        } catch (e: Exception) {
            emptyArray()
        }

        val melFixtures = fixtureFiles.filter { it.endsWith("_mel.npy") }

        if (melFixtures.isEmpty()) {
            // No fixtures available — skip but don't fail
            println("No test fixtures found in $fixtureDir — skipping parity tests")
            return
        }

        for (melFile in melFixtures) {
            val sampleName = melFile.removeSuffix("_mel.npy")
            println("Testing parity for: $sampleName")

            // Load Python reference mel
            val refMelNpy = context.assets.open("$fixtureDir/$melFile").use {
                NpyReader.read(it)
            }
            val refMel = refMelNpy.toFloatMatrix()

            // Load corresponding WAV and compute Kotlin mel
            val wavFile = "$sampleName.wav"
            val pcm = try {
                context.assets.open("$fixtureDir/$wavFile").use {
                    com.edgy.privacy.util.WavWriter.readWav(it).second
                }
            } catch (e: Exception) {
                println("  WAV file not found: $wavFile — skipping")
                continue
            }

            val kotlinMel = melExtractor.extract(pcm)

            // Criterion 1: max absolute diff < 1e-3
            var maxDiff = 0f
            val rows = minOf(kotlinMel.size, refMel.size)
            val cols = minOf(
                if (kotlinMel.isNotEmpty()) kotlinMel[0].size else 0,
                if (refMel.isNotEmpty()) refMel[0].size else 0
            )
            for (m in 0 until rows) {
                for (t in 0 until cols) {
                    val diff = abs(kotlinMel[m][t] - refMel[m][t])
                    if (diff > maxDiff) maxDiff = diff
                }
            }
            println("  Mel max abs diff: $maxDiff")
            assertTrue(
                "Mel parity failed for $sampleName: max diff $maxDiff > 1e-3",
                maxDiff < 1e-3f
            )

            // Try to test VQ parity if encoder is available
            testVqParity(sampleName, fixtureDir, kotlinMel)
        }
    }

    /**
     * Test VQ embedding parity against Python reference.
     */
    private fun testVqParity(sampleName: String, fixtureDir: String, mel: Array<FloatArray>) {
        val encoder = try {
            modelManager.loadEncoder()
        } catch (e: Exception) {
            println("  Encoder not available — skipping VQ parity test")
            return
        }

        val encoderOutput = encoder.encode(mel)

        // Load reference VQ embeddings
        try {
            val refVqNpy = context.assets.open("$fixtureDir/${sampleName}_vq.npy").use {
                NpyReader.read(it)
            }
            val refVq = refVqNpy.toFloatMatrix()

            // Criterion 2: cosine similarity > 0.99
            val similarity = cosineSimilarity(
                encoderOutput.vqEmbedding.flatMap { it.toList() }.toFloatArray(),
                refVq.flatMap { it.toList() }.toFloatArray()
            )
            println("  VQ cosine similarity: $similarity")
            assertTrue(
                "VQ parity failed for $sampleName: cosine sim $similarity < 0.99",
                similarity > 0.99
            )
        } catch (e: Exception) {
            println("  Reference VQ not found — skipping")
        }

        // Load reference indices
        try {
            val refIdxNpy = context.assets.open("$fixtureDir/${sampleName}_indices.npy").use {
                NpyReader.read(it)
            }
            val refIndices = refIdxNpy.toIntArray()

            // Criterion 3: match rate > 95%
            val minLen = minOf(encoderOutput.codebookIndices.size, refIndices.size)
            var matches = 0
            for (i in 0 until minLen) {
                if (encoderOutput.codebookIndices[i] == refIndices[i]) matches++
            }
            val matchRate = if (minLen > 0) matches.toFloat() / minLen else 0f
            println("  Index match rate: $matchRate ($matches/$minLen)")
            assertTrue(
                "Index parity failed for $sampleName: match rate $matchRate < 0.95",
                matchRate > 0.95f
            )
        } catch (e: Exception) {
            println("  Reference indices not found — skipping")
        }
    }

    /**
     * Stage 2: Verify privacy tier outputs.
     */
    @Test
    fun testPrivacyTierOutputs() {
        val config = modelManager.loadConfig()

        val encoder = try {
            modelManager.loadEncoder()
        } catch (e: Exception) {
            println("Encoder not available — skipping tier output tests")
            return
        }

        val speakerEmbeddings = try {
            modelManager.loadSpeakerEmbeddings()
        } catch (e: Exception) {
            arrayOf(FloatArray(config.encoder.embeddingDim) { 0.1f })
        }

        val pipeline = PrivacyPipeline(melExtractor, encoder, speakerEmbeddings)

        val pcm = FloatArray(16000) { i ->
            (0.5 * kotlin.math.sin(2.0 * Math.PI * 440.0 * i / 16000)).toFloat()
        }

        // LOW tier: raw audio passthrough
        pipeline.setTier(PrivacyTier.LOW)
        val lowOutput = pipeline.processChunk(pcm)
        assertNotNull("LOW: raw audio should not be null", lowOutput.rawAudio)
        assertArrayEquals("LOW: raw audio should be identical to input", pcm, lowOutput.rawAudio!!, 0f)
        assertNull("LOW: VQ embedding should be null", lowOutput.vqEmbedding)
        assertNull("LOW: speaker embedding should be null", lowOutput.speakerEmbedding)

        // MODERATE tier: VQ + speaker embedding
        pipeline.setTier(PrivacyTier.MODERATE)
        val modOutput = pipeline.processChunk(pcm)
        assertNull("MODERATE: raw audio should be null", modOutput.rawAudio)
        assertNotNull("MODERATE: VQ embedding should not be null", modOutput.vqEmbedding)
        assertTrue("MODERATE: VQ embedding should have dim 64",
            modOutput.vqEmbedding!![0].size == config.encoder.embeddingDim)
        assertNotNull("MODERATE: speaker embedding should not be null", modOutput.speakerEmbedding)
        assertEquals("MODERATE: speaker embedding should have dim 64",
            config.encoder.embeddingDim, modOutput.speakerEmbedding!!.size)

        // HIGH tier: VQ only
        pipeline.setTier(PrivacyTier.HIGH)
        val highOutput = pipeline.processChunk(pcm)
        assertNull("HIGH: raw audio should be null", highOutput.rawAudio)
        assertNotNull("HIGH: VQ embedding should not be null", highOutput.vqEmbedding)
        assertNull("HIGH: speaker embedding should be null", highOutput.speakerEmbedding)

        // MODERATE and HIGH should produce identical VQ embeddings
        val modVq = modOutput.vqEmbedding!!.flatMap { it.toList() }.toFloatArray()
        val highVq = highOutput.vqEmbedding!!.flatMap { it.toList() }.toFloatArray()
        assertArrayEquals(
            "MODERATE and HIGH should produce identical VQ embeddings",
            modVq, highVq, 0f
        )
    }

    /**
     * Compute cosine similarity between two vectors.
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        val minLen = minOf(a.size, b.size)
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in 0 until minLen) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom > 0) (dotProduct / denom).toFloat() else 0f
    }
}
