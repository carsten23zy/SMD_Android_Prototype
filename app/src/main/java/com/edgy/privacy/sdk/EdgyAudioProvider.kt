package com.edgy.privacy.sdk

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.util.Log
import com.edgy.privacy.audio.AudioCaptureService
import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.audio.RealtimePipelineManager
import com.edgy.privacy.ml.EdgyEncoder
import com.edgy.privacy.ml.InferenceStats
import com.edgy.privacy.ml.ModelManager
import com.edgy.privacy.privacy.PrivacyPipeline
import com.edgy.privacy.privacy.PrivacyTier
import com.edgy.privacy.vocoder.GriffinLimVocoder

/**
 * AIDL service that exposes privacy-filtered audio to client apps.
 *
 * Client apps bind to this service and call openStream() to receive a
 * ParcelFileDescriptor that delivers privacy-filtered PCM audio (16kHz, mono, 16-bit).
 *
 * For MODERATE/HIGH tiers, the vocoder reconstructs audio from VQ embeddings.
 * For LOW tier, raw audio passes through.
 *
 * Usage from client app:
 * ```
 * val intent = Intent("com.edgy.privacy.AUDIO_SOURCE")
 * intent.setPackage("com.edgy.privacy")
 * bindService(intent, connection, Context.BIND_AUTO_CREATE)
 * // In onServiceConnected:
 * val source = IEdgyAudioSource.Stub.asInterface(binder)
 * val pfd = source.openStream(1) // MODERATE tier
 * val inputStream = ParcelFileDescriptor.AutoCloseInputStream(pfd)
 * // Read 16-bit PCM from inputStream...
 * ```
 */
class EdgyAudioProvider : Service() {

    companion object {
        private const val TAG = "EdgyAudioProvider"
        const val ACTION_BIND = "com.edgy.privacy.AUDIO_SOURCE"
    }

    private var modelManager: ModelManager? = null
    private var pipeline: PrivacyPipeline? = null
    private var pipelineManager: RealtimePipelineManager? = null
    private var currentTier: PrivacyTier = PrivacyTier.MODERATE
    private val inferenceStats = InferenceStats()
    private var sampleRate = AudioCaptureService.DEFAULT_SAMPLE_RATE

    // Active client pipe
    private var activePipe: AudioStreamPipe? = null

    private val binder = object : IEdgyAudioSource.Stub() {

        override fun openStream(privacyTier: Int): ParcelFileDescriptor? {
            val tier = when (privacyTier) {
                0 -> PrivacyTier.LOW
                1 -> PrivacyTier.MODERATE
                2 -> PrivacyTier.HIGH
                else -> PrivacyTier.MODERATE
            }
            this@EdgyAudioProvider.currentTier = tier
            pipeline?.setTier(tier)

            // Close any existing pipe
            activePipe?.close()

            val pipe = AudioStreamPipe()
            val readEnd = pipe.open()
            activePipe = pipe

            // Wire pipeline output to the pipe
            pipelineManager?.setOnOutputReady { output ->
                val audio = when {
                    output.tier == PrivacyTier.LOW -> output.rawAudio
                    output.reconstructedAudio != null && output.reconstructedAudio.isNotEmpty() ->
                        output.reconstructedAudio
                    else -> null
                }
                if (audio != null) {
                    if (!pipe.write(audio)) {
                        Log.w(TAG, "Client pipe broken, closing stream")
                    }
                }
            }

            Log.i(TAG, "Stream opened for client at tier ${tier.name}")
            return readEnd
        }

        override fun closeStream() {
            activePipe?.close()
            activePipe = null
            pipelineManager?.setOnOutputReady(null)
            Log.i(TAG, "Stream closed by client")
        }

        override fun isCapturing(): Boolean {
            return pipelineManager?.isRunning() == true
        }

        override fun getCurrentTier(): Int {
            return when (this@EdgyAudioProvider.currentTier) {
                PrivacyTier.LOW -> 0
                PrivacyTier.MODERATE -> 1
                PrivacyTier.HIGH -> 2
            }
        }

        override fun getSampleRate(): Int {
            return sampleRate
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "EdgyAudioProvider created")
        initializePipeline()
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "Client binding to EdgyAudioProvider")
        return binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        activePipe?.close()
        activePipe = null
        pipelineManager?.setOnOutputReady(null)
        Log.i(TAG, "Client unbound from EdgyAudioProvider")
        return true // allow rebind
    }

    override fun onDestroy() {
        activePipe?.close()
        pipelineManager?.stop()
        modelManager?.close()
        Log.i(TAG, "EdgyAudioProvider destroyed")
        super.onDestroy()
    }

    private fun initializePipeline() {
        try {
            val manager = ModelManager(this)
            val config = manager.loadConfig()
            sampleRate = config.preprocessing.sampleRate
            val melExtractor = MelSpectrogramExtractor(config)

            val encoder: EdgyEncoder
            try {
                encoder = manager.loadEncoder()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load encoder: ${e.message}")
                return
            }

            val speakerEmbeddings = try {
                manager.loadSpeakerEmbeddings()
            } catch (e: Exception) {
                emptyArray<FloatArray>()
            }

            val projMatrix = manager.loadProjectionMatrix()
            val codebook = try { manager.loadCodebook() } catch (e: Exception) { null }
            val vocoder = GriffinLimVocoder(config, projMatrix, codebook, melExtractor.getMelFilterbank())

            val pl = PrivacyPipeline(melExtractor, encoder, speakerEmbeddings, vocoder = vocoder)
            pl.setTier(currentTier)

            pipeline = pl
            pipelineManager = RealtimePipelineManager(pl, inferenceStats)
            modelManager = manager

            Log.i(TAG, "Pipeline initialized. Vocoder available: ${vocoder.isAvailable}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize pipeline", e)
        }
    }
}
