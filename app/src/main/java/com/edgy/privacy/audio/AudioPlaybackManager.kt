package com.edgy.privacy.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.edgy.privacy.util.DSP
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Manages audio playback via AudioTrack for human listening tests.
 *
 * Supports two modes:
 * - Streaming: continuous real-time playback of processed chunks during live capture
 * - One-shot: playback of a complete processed audio buffer (from file processing)
 */
class AudioPlaybackManager(private val sampleRate: Int = 16000) {

    companion object {
        private const val TAG = "AudioPlayback"
    }

    private var audioTrack: AudioTrack? = null
    private val isPlaying = AtomicBoolean(false)
    private var playbackThread: Thread? = null

    /**
     * Start streaming playback mode. Subsequent calls to [writeChunk] feed audio
     * to the speaker in real time.
     */
    fun startStreaming() {
        if (isPlaying.get()) return

        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        // Use at least 4x min buffer to absorb processing jitter
        val trackBuffer = maxOf(bufferSize, sampleRate * 2) // ~1 second buffer

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(trackBuffer)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack?.play()
        isPlaying.set(true)
        Log.i(TAG, "Streaming playback started (buffer=${trackBuffer}B)")
    }

    /**
     * Write a PCM float chunk to the audio output during streaming playback.
     * Call from the processing/output thread.
     */
    fun writeChunk(pcm: FloatArray) {
        if (!isPlaying.get()) return
        val track = audioTrack ?: return

        val shorts = DSP.floatToShort(pcm)
        track.write(shorts, 0, shorts.size)
    }

    /**
     * Stop streaming playback and release AudioTrack.
     */
    fun stopStreaming() {
        isPlaying.set(false)
        try {
            audioTrack?.stop()
        } catch (e: IllegalStateException) {
            Log.w(TAG, "AudioTrack stop error: ${e.message}")
        }
        audioTrack?.release()
        audioTrack = null
        Log.i(TAG, "Streaming playback stopped")
    }

    /**
     * Play a complete PCM float buffer (one-shot mode for file processing results).
     * Playback runs on a background thread. Call [stopOneShot] to cancel.
     *
     * @param pcm PCM float samples in [-1, 1]
     * @param onComplete Callback when playback finishes (called on playback thread)
     */
    fun playOneShot(pcm: FloatArray, onComplete: (() -> Unit)? = null) {
        stopOneShot()

        playbackThread = Thread({
            try {
                val bufferSize = AudioTrack.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )

                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build()
                    )
                    .setBufferSizeInBytes(maxOf(bufferSize, sampleRate))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()

                audioTrack = track
                isPlaying.set(true)
                track.play()

                val shorts = DSP.floatToShort(pcm)
                var offset = 0
                val chunkSize = 1600 // 100ms at 16kHz

                while (offset < shorts.size && isPlaying.get()) {
                    val remaining = shorts.size - offset
                    val writeSize = minOf(chunkSize, remaining)
                    track.write(shorts, offset, writeSize)
                    offset += writeSize
                }

                // Wait for AudioTrack to finish playing buffered data
                if (isPlaying.get()) {
                    // Estimate remaining playback time
                    val remainingMs = (shorts.size.toLong() * 1000) / sampleRate
                    // AudioTrack.stop() will play remaining buffer
                    track.stop()
                    // Small grace period for final samples
                    Thread.sleep(minOf(remainingMs, 200))
                }

                isPlaying.set(false)
                track.release()
                audioTrack = null
                onComplete?.invoke()
                Log.i(TAG, "One-shot playback complete (${pcm.size} samples)")
            } catch (e: Exception) {
                isPlaying.set(false)
                Log.e(TAG, "One-shot playback error", e)
                onComplete?.invoke()
            }
        }, "EDGY-Playback").apply { start() }
    }

    /**
     * Stop one-shot playback if running.
     */
    fun stopOneShot() {
        if (!isPlaying.get() && playbackThread == null) return
        isPlaying.set(false)
        try {
            audioTrack?.stop()
        } catch (_: IllegalStateException) {}
        playbackThread?.join(1000)
        audioTrack?.release()
        audioTrack = null
        playbackThread = null
    }

    /**
     * Whether audio is currently playing (streaming or one-shot).
     */
    fun isPlaying(): Boolean = isPlaying.get()

    /**
     * Release all resources.
     */
    fun release() {
        stopStreaming()
        stopOneShot()
    }
}
