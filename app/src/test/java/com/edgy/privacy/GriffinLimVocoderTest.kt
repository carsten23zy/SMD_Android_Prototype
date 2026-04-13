package com.edgy.privacy

import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.ml.EncoderConfig
import com.edgy.privacy.ml.FilesConfig
import com.edgy.privacy.ml.ModelConfig
import com.edgy.privacy.ml.OnnxSettings
import com.edgy.privacy.ml.PreprocessingConfig
import com.edgy.privacy.vocoder.GriffinLimVocoder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

/**
 * Unit tests for [GriffinLimVocoder].
 *
 * The vocoder consumes the same normalized log-mel that the encoder receives,
 * so these tests build mels via [MelSpectrogramExtractor] and verify that
 * synthesis returns plausible audio.
 */
class GriffinLimVocoderTest {

    private lateinit var config: ModelConfig
    private lateinit var melExtractor: MelSpectrogramExtractor
    private lateinit var vocoder: GriffinLimVocoder

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
                speakerEmbeddings = "speaker_embeddings.npy"
            ),
            onnxSettings = OnnxSettings(
                intraOpNumThreads = 2,
                interOpNumThreads = 1,
                executionMode = "SEQUENTIAL",
                optimizationLevel = "ALL_OPT"
            )
        )

        melExtractor = MelSpectrogramExtractor(config)
        // Use few iterations to keep tests fast; momentum=0 disables fast-Griffin-Lim.
        vocoder = GriffinLimVocoder(
            config,
            melExtractor.getMelFilterbank(),
            iterations = 4,
            momentum = 0.0
        )
    }

    private fun sineWave(durationSec: Float, freqHz: Float): FloatArray {
        val sr = config.preprocessing.sampleRate
        val n = (durationSec * sr).toInt()
        return FloatArray(n) { i ->
            (0.5 * sin(2.0 * PI * freqHz * i / sr)).toFloat()
        }
    }

    @Test
    fun `vocoder is always available`() {
        assertTrue(vocoder.isAvailable)
    }

    @Test
    fun `synthesize returns empty for empty mel`() {
        val empty = Array(config.preprocessing.nMels) { FloatArray(0) }
        assertEquals(0, vocoder.synthesize(empty).size)
    }

    @Test
    fun `synthesize round-trips a sine wave to non-empty audio in range`() {
        val pcm = sineWave(0.25f, 440f)
        val mel = melExtractor.extract(pcm)

        val audio = vocoder.synthesize(mel)

        assertTrue("Expected non-empty reconstructed audio", audio.isNotEmpty())
        for (sample in audio) {
            assertTrue("Sample $sample out of range", sample in -1f..1f)
        }
    }

    @Test
    fun `output length scales with number of mel frames`() {
        val short = melExtractor.extract(sineWave(0.1f, 440f))
        val long = melExtractor.extract(sineWave(0.4f, 440f))

        val shortAudio = vocoder.synthesize(short)
        val longAudio = vocoder.synthesize(long)

        assertTrue(
            "Longer mel should yield longer audio (${shortAudio.size} vs ${longAudio.size})",
            longAudio.size > shortAudio.size
        )
    }

    @Test
    fun `different mels produce different audio`() {
        val melLow = melExtractor.extract(sineWave(0.25f, 220f))
        val melHigh = melExtractor.extract(sineWave(0.25f, 880f))

        val a1 = vocoder.synthesize(melLow)
        val a2 = vocoder.synthesize(melHigh)

        val n = minOf(a1.size, a2.size)
        var maxDiff = 0f
        for (i in 0 until n) {
            val d = kotlin.math.abs(a1[i] - a2[i])
            if (d > maxDiff) maxDiff = d
        }
        assertTrue(
            "Different mels should yield different audio (maxDiff=$maxDiff)",
            maxDiff > 0.001f
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `synthesize rejects mismatched mel dimensions`() {
        val wrong = Array(config.preprocessing.nMels - 1) { FloatArray(10) }
        vocoder.synthesize(wrong)
    }
}
