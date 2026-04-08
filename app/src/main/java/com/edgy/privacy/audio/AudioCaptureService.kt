package com.edgy.privacy.audio

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.edgy.privacy.MainActivity
import com.edgy.privacy.R
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Foreground service capturing microphone via AudioRecord.
 *
 * Thread 1 (Capture): AudioRecord read loop on a dedicated thread.
 * Reads 16kHz mono PCM 16-bit, converts to Float32, pushes to AudioChunkBuffer.
 *
 * Architecture:
 *   AudioRecord → Int16 → Float32 → AudioChunkBuffer → [Processing Thread]
 */
class AudioCaptureService : Service() {

    companion object {
        private const val TAG = "AudioCaptureService"
        private const val CHANNEL_ID = "edgy_capture_channel"
        private const val NOTIFICATION_ID = 1
        const val DEFAULT_SAMPLE_RATE = 16000
        const val DEFAULT_CHUNK_SIZE_MS = 100

        const val ACTION_START = "com.edgy.privacy.action.START_CAPTURE"
        const val ACTION_STOP = "com.edgy.privacy.action.STOP_CAPTURE"
    }

    inner class LocalBinder : Binder() {
        fun getService(): AudioCaptureService = this@AudioCaptureService
    }

    private val binder = LocalBinder()
    private var audioRecord: AudioRecord? = null
    private var captureThread: Thread? = null
    private val isCapturing = AtomicBoolean(false)

    // Configurable params
    private var sampleRate = DEFAULT_SAMPLE_RATE
    private var chunkSizeMs = DEFAULT_CHUNK_SIZE_MS
    private val chunkSizeSamples: Int get() = sampleRate * chunkSizeMs / 1000

    // Output buffer — processing thread reads from here
    private val outputBuffer = AudioChunkBuffer(capacity = 16)

    // Stats
    private val totalChunksCaptured = AtomicLong(0)
    private val droppedChunks = AtomicLong(0)
    private var captureStartTimeMs = 0L

    // Callback for chunk-ready events
    private var onChunkReadyListener: ((FloatArray) -> Unit)? = null

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                sampleRate = intent.getIntExtra("sample_rate", DEFAULT_SAMPLE_RATE)
                chunkSizeMs = intent.getIntExtra("chunk_size_ms", DEFAULT_CHUNK_SIZE_MS)
                startForegroundCapture()
            }
            ACTION_STOP -> {
                stopCapture()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopCapture()
        super.onDestroy()
    }

    /**
     * Start capturing audio from the microphone.
     */
    fun startCapture(sampleRate: Int = DEFAULT_SAMPLE_RATE, chunkSizeMs: Int = DEFAULT_CHUNK_SIZE_MS) {
        this.sampleRate = sampleRate
        this.chunkSizeMs = chunkSizeMs
        startForegroundCapture()
    }

    /**
     * Stop audio capture.
     */
    fun stopCapture() {
        isCapturing.set(false)
        captureThread?.join(2000)
        captureThread = null

        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioRecord", e)
        }
        audioRecord?.release()
        audioRecord = null
    }

    /**
     * Set callback for when a new chunk is ready.
     */
    fun setOnChunkReadyListener(callback: ((FloatArray) -> Unit)?) {
        onChunkReadyListener = callback
    }

    /**
     * Get the output buffer for the processing thread to read from.
     */
    fun getOutputBuffer(): AudioChunkBuffer = outputBuffer

    /**
     * Whether capture is currently active.
     */
    fun isCapturing(): Boolean = isCapturing.get()

    /**
     * Get capture statistics.
     */
    fun getStats(): CaptureStats {
        val elapsedMs = if (captureStartTimeMs > 0) {
            System.currentTimeMillis() - captureStartTimeMs
        } else 0L

        return CaptureStats(
            totalChunks = totalChunksCaptured.get(),
            droppedChunks = droppedChunks.get(),
            elapsedMs = elapsedMs,
            sampleRate = sampleRate,
            chunkSizeMs = chunkSizeMs
        )
    }

    private fun startForegroundCapture() {
        if (isCapturing.get()) return

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "RECORD_AUDIO permission not granted")
            return
        }

        val notification = createNotification("Capturing audio...")
        startForeground(NOTIFICATION_ID, notification)

        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        // Use at least 4x the chunk size for the internal buffer
        val bufferSize = maxOf(minBufferSize, chunkSizeSamples * 2 * 4)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord initialization failed")
            audioRecord?.release()
            audioRecord = null
            return
        }

        isCapturing.set(true)
        totalChunksCaptured.set(0)
        droppedChunks.set(0)
        captureStartTimeMs = System.currentTimeMillis()
        outputBuffer.clear()

        audioRecord?.startRecording()

        captureThread = Thread({
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO)
            captureLoop()
        }, "EDGY-Capture").apply {
            start()
        }

        Log.i(TAG, "Capture started: ${sampleRate}Hz, ${chunkSizeMs}ms chunks (${chunkSizeSamples} samples)")
    }

    /**
     * Main capture loop — runs on dedicated thread.
     * Reads Int16 PCM, converts to Float32, pushes to ring buffer.
     */
    private fun captureLoop() {
        val shortBuffer = ShortArray(chunkSizeSamples)

        while (isCapturing.get()) {
            val shortsRead = audioRecord?.read(shortBuffer, 0, chunkSizeSamples) ?: -1

            if (shortsRead > 0) {
                // Convert Int16 → Float32 [-1.0, 1.0]
                val floatChunk = FloatArray(shortsRead) { i ->
                    shortBuffer[i].toFloat() / 32768f
                }

                val pushed = outputBuffer.push(floatChunk)
                if (pushed) {
                    totalChunksCaptured.incrementAndGet()
                    onChunkReadyListener?.invoke(floatChunk)
                } else {
                    droppedChunks.incrementAndGet()
                    Log.w(TAG, "Chunk dropped — buffer full (total dropped: ${droppedChunks.get()})")
                }
            } else if (shortsRead < 0) {
                Log.e(TAG, "AudioRecord.read error: $shortsRead")
                break
            }
        }

        Log.i(TAG, "Capture loop ended. Total: ${totalChunksCaptured.get()}, Dropped: ${droppedChunks.get()}")
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "EDGY Audio Capture",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Privacy-filtered audio capture"
            setShowBadge(false)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }

    private fun createNotification(contentText: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("EDGY Privacy Filter")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    /**
     * Update the notification text (e.g., with current tier info).
     */
    fun updateNotification(text: String) {
        val notification = createNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    data class CaptureStats(
        val totalChunks: Long,
        val droppedChunks: Long,
        val elapsedMs: Long,
        val sampleRate: Int,
        val chunkSizeMs: Int
    ) {
        val elapsedSeconds: Double get() = elapsedMs / 1000.0
        val expectedChunks: Long get() = if (chunkSizeMs > 0) elapsedMs / chunkSizeMs else 0
        val dropRate: Double get() = if (totalChunks + droppedChunks > 0) {
            droppedChunks.toDouble() / (totalChunks + droppedChunks)
        } else 0.0

        fun summary(): String = buildString {
            append("Duration: ${"%.1f".format(elapsedSeconds)}s\n")
            append("Chunks: $totalChunks captured, $droppedChunks dropped\n")
            append("Expected: $expectedChunks chunks\n")
            append("Drop rate: ${"%.2f".format(dropRate * 100)}%\n")
        }
    }
}