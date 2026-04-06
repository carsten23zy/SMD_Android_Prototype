package com.edgy.privacy.audio

import android.util.Log
import com.edgy.privacy.ml.InferenceStats
import com.edgy.privacy.privacy.PrivacyOutput
import com.edgy.privacy.privacy.PrivacyPipeline
import com.edgy.privacy.privacy.PrivacyTier
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Three-thread real-time processing pipeline.
 *
 * Thread 1 (Capture):    AudioCaptureService → RingBuffer A (captureBuffer)
 * Thread 2 (Processing): RingBuffer A → Mel+ONNX → RingBuffer B (outputBuffer)
 * Thread 3 (Output):     RingBuffer B → File/AudioTrack
 *
 * Latency budget per 100ms chunk:
 *   Capture:     ~1ms
 *   Mel + ONNX:  ~15-20ms (FP32 on phone CPU)
 *   Output:      ~1ms
 *   Headroom:    ~78ms
 */
class RealtimePipelineManager(
    private val pipeline: PrivacyPipeline,
    private val inferenceStats: InferenceStats
) {

    companion object {
        private const val TAG = "RealtimePipeline"
    }

    // Buffers connecting the three threads
    private val captureBuffer = AudioChunkBuffer(capacity = 16)
    private val outputBuffer = AudioChunkBuffer(capacity = 16)

    // Thread 2: processing thread
    private var processingThread: Thread? = null
    private val isRunning = AtomicBoolean(false)

    // Thread 3: output thread
    private var outputThread: Thread? = null

    // Callback for processed output
    private var onOutputReady: ((PrivacyOutput) -> Unit)? = null

    // Output file writer (optional)
    private var audioOutputService: AudioOutputService? = null
    private var outputDirectory: String? = null
    private var outputChunkIndex = 0

    /**
     * Get the capture buffer (AudioCaptureService pushes here).
     */
    fun getCaptureBuffer(): AudioChunkBuffer = captureBuffer

    /**
     * Set callback for processed output chunks.
     */
    fun setOnOutputReady(callback: ((PrivacyOutput) -> Unit)?) {
        onOutputReady = callback
    }

    /**
     * Configure file output for processed chunks.
     */
    fun setFileOutput(service: AudioOutputService, directory: String) {
        audioOutputService = service
        outputDirectory = directory
    }

    /**
     * Start the processing and output threads.
     */
    fun start() {
        if (isRunning.get()) return
        isRunning.set(true)
        outputChunkIndex = 0
        inferenceStats.reset()

        // Thread 2: Processing — reads from captureBuffer, writes to outputBuffer
        processingThread = Thread({
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_MORE_FAVORABLE)
            processingLoop()
        }, "EDGY-Processing").apply { start() }

        // Thread 3: Output — reads from outputBuffer, writes to file/callback
        outputThread = Thread({
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            outputLoop()
        }, "EDGY-Output").apply { start() }

        Log.i(TAG, "Pipeline started (processing + output threads)")
    }

    /**
     * Stop the processing and output threads.
     */
    fun stop() {
        isRunning.set(false)
        processingThread?.join(3000)
        outputThread?.join(3000)
        processingThread = null
        outputThread = null
        captureBuffer.clear()
        outputBuffer.clear()
        Log.i(TAG, "Pipeline stopped. ${inferenceStats.summary()}")
    }

    /**
     * Whether the pipeline is running.
     */
    fun isRunning(): Boolean = isRunning.get()

    /**
     * Push a captured chunk directly (alternative to using captureBuffer).
     */
    fun pushCapturedChunk(chunk: FloatArray): Boolean {
        return captureBuffer.push(chunk)
    }

    /**
     * Thread 2: Processing loop.
     * Reads PCM chunks from captureBuffer, runs mel+ONNX, pushes results to outputBuffer.
     */
    private fun processingLoop() {
        Log.i(TAG, "Processing thread started")

        while (isRunning.get()) {
            val chunk = captureBuffer.pull()
            if (chunk == null) {
                // No data available — short sleep to avoid busy-waiting
                Thread.sleep(5)
                continue
            }

            try {
                val startTotal = System.nanoTime()

                // Run through privacy pipeline (streaming mode)
                val startMel = System.nanoTime()
                val output = pipeline.processChunkStreaming(chunk)
                val melAndEncodeMs = (System.nanoTime() - startMel) / 1_000_000
                val totalMs = (System.nanoTime() - startTotal) / 1_000_000

                // Record stats — split mel vs encoder is approximate in streaming mode
                val encoderMs = output.processingTimeMs
                val melMs = melAndEncodeMs - encoderMs
                inferenceStats.recordChunk(
                    melMs = maxOf(0, melMs),
                    encoderMs = maxOf(0, encoderMs),
                    totalMs = totalMs
                )

                // Push audio to output buffer based on tier
                if (output.tier == PrivacyTier.LOW && output.rawAudio != null) {
                    outputBuffer.push(output.rawAudio)
                } else if (output.reconstructedAudio != null && output.reconstructedAudio.isNotEmpty()) {
                    // MODERATE/HIGH: push vocoder-reconstructed audio
                    outputBuffer.push(output.reconstructedAudio)
                }

                // Notify output thread
                onOutputReady?.invoke(output)

            } catch (e: Exception) {
                Log.e(TAG, "Processing error", e)
            }
        }

        Log.i(TAG, "Processing thread ended")
    }

    /**
     * Thread 3: Output loop.
     * Reads processed audio from outputBuffer and writes to file or AudioTrack.
     */
    private fun outputLoop() {
        Log.i(TAG, "Output thread started")

        while (isRunning.get()) {
            val chunk = outputBuffer.pull()
            if (chunk == null) {
                Thread.sleep(5)
                continue
            }

            try {
                // File output if configured
                val dir = outputDirectory
                val service = audioOutputService
                if (dir != null && service != null) {
                    val prefix = "chunk_${"%06d".format(outputChunkIndex++)}"
                    service.writeToWavFile(
                        "$dir/$prefix.wav",
                        chunk
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Output error", e)
            }
        }

        Log.i(TAG, "Output thread ended")
    }
}
