package com.edgy.privacy.ml

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicLong

/**
 * Tracks inference latency statistics: p50, p95, p99, and throughput.
 * Thread-safe for use from the processing pipeline thread.
 *
 * Maintains a sliding window of the most recent measurements.
 */
class InferenceStats(private val windowSize: Int = 1000) {

    private val melLatencies = ConcurrentLinkedQueue<Long>()
    private val encoderLatencies = ConcurrentLinkedQueue<Long>()
    private val totalLatencies = ConcurrentLinkedQueue<Long>()
    private val chunksProcessed = AtomicLong(0)
    private var startTimeNs = System.nanoTime()

    /**
     * Record latency for a single chunk processing.
     */
    fun recordChunk(melMs: Long, encoderMs: Long, totalMs: Long) {
        addToWindow(melLatencies, melMs)
        addToWindow(encoderLatencies, encoderMs)
        addToWindow(totalLatencies, totalMs)
        chunksProcessed.incrementAndGet()
    }

    /**
     * Get the total number of chunks processed.
     */
    fun getChunksProcessed(): Long = chunksProcessed.get()

    /**
     * Get percentile statistics for mel extraction.
     */
    fun getMelStats(): LatencyPercentiles = computePercentiles(melLatencies)

    /**
     * Get percentile statistics for ONNX encoder inference.
     */
    fun getEncoderStats(): LatencyPercentiles = computePercentiles(encoderLatencies)

    /**
     * Get percentile statistics for total pipeline processing.
     */
    fun getTotalStats(): LatencyPercentiles = computePercentiles(totalLatencies)

    /**
     * Get throughput in chunks per second.
     */
    fun getThroughput(): Double {
        val elapsedSec = (System.nanoTime() - startTimeNs) / 1_000_000_000.0
        return if (elapsedSec > 0) chunksProcessed.get() / elapsedSec else 0.0
    }

    /**
     * Reset all statistics.
     */
    fun reset() {
        melLatencies.clear()
        encoderLatencies.clear()
        totalLatencies.clear()
        chunksProcessed.set(0)
        startTimeNs = System.nanoTime()
    }

    /**
     * Generate a summary report.
     */
    fun summary(): String {
        val mel = getMelStats()
        val enc = getEncoderStats()
        val total = getTotalStats()
        return buildString {
            append("Chunks processed: ${chunksProcessed.get()}\n")
            append("Throughput: ${"%.1f".format(getThroughput())} chunks/s\n")
            append("\nMel extraction:\n")
            append("  p50=${mel.p50}ms  p95=${mel.p95}ms  p99=${mel.p99}ms\n")
            append("\nONNX encoder:\n")
            append("  p50=${enc.p50}ms  p95=${enc.p95}ms  p99=${enc.p99}ms\n")
            append("\nTotal pipeline:\n")
            append("  p50=${total.p50}ms  p95=${total.p95}ms  p99=${total.p99}ms\n")
        }
    }

    /**
     * Compact one-line stats for live display.
     */
    fun compactSummary(): String {
        val total = getTotalStats()
        return "p50=${total.p50}ms p95=${total.p95}ms | ${chunksProcessed.get()} chunks"
    }

    private fun addToWindow(queue: ConcurrentLinkedQueue<Long>, value: Long) {
        queue.add(value)
        while (queue.size > windowSize) {
            queue.poll()
        }
    }

    private fun computePercentiles(queue: ConcurrentLinkedQueue<Long>): LatencyPercentiles {
        val sorted = queue.toList().sorted()
        if (sorted.isEmpty()) return LatencyPercentiles(0, 0, 0, 0, 0)

        return LatencyPercentiles(
            min = sorted.first(),
            p50 = sorted[percentileIndex(sorted.size, 50)],
            p95 = sorted[percentileIndex(sorted.size, 95)],
            p99 = sorted[percentileIndex(sorted.size, 99)],
            max = sorted.last()
        )
    }

    private fun percentileIndex(size: Int, percentile: Int): Int {
        return ((size - 1) * percentile / 100).coerceIn(0, size - 1)
    }

    data class LatencyPercentiles(
        val min: Long,
        val p50: Long,
        val p95: Long,
        val p99: Long,
        val max: Long
    )
}