package com.edgy.privacy

import com.edgy.privacy.ml.InferenceStats
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for InferenceStats tracker.
 */
class InferenceStatsTest {

    private lateinit var stats: InferenceStats

    @Before
    fun setUp() {
        stats = InferenceStats(windowSize = 100)
    }

    @Test
    fun testEmptyStats() {
        assertEquals(0, stats.getChunksProcessed())
        val p = stats.getTotalStats()
        assertEquals(0, p.p50)
        assertEquals(0, p.p95)
    }

    @Test
    fun testRecordChunks() {
        stats.recordChunk(melMs = 5, encoderMs = 15, totalMs = 20)
        stats.recordChunk(melMs = 4, encoderMs = 14, totalMs = 18)
        stats.recordChunk(melMs = 6, encoderMs = 16, totalMs = 22)

        assertEquals(3, stats.getChunksProcessed())
    }

    @Test
    fun testPercentiles() {
        // Add 100 measurements with known values
        for (i in 1..100) {
            stats.recordChunk(melMs = i.toLong(), encoderMs = (i * 2).toLong(), totalMs = (i * 3).toLong())
        }

        val totalStats = stats.getTotalStats()
        // p50 of 3,6,9,...,300 should be around 150
        assertTrue("p50 should be ~150, got ${totalStats.p50}", totalStats.p50 in 140..160)
        // p95 should be around 285
        assertTrue("p95 should be ~285, got ${totalStats.p95}", totalStats.p95 in 275..295)

        val melStats = stats.getMelStats()
        assertTrue("mel p50 should be ~50, got ${melStats.p50}", melStats.p50 in 45..55)

        val encStats = stats.getEncoderStats()
        assertTrue("enc p50 should be ~100, got ${encStats.p50}", encStats.p50 in 95..105)
    }

    @Test
    fun testSlidingWindow() {
        val smallStats = InferenceStats(windowSize = 10)

        // Add 20 measurements — window should only keep last 10
        for (i in 1..20) {
            smallStats.recordChunk(melMs = i.toLong(), encoderMs = i.toLong(), totalMs = i.toLong())
        }

        assertEquals(20, smallStats.getChunksProcessed())

        // p50 of the window [11..20] should be ~15
        val total = smallStats.getTotalStats()
        assertTrue("p50 should be ~15, got ${total.p50}", total.p50 in 14..16)
    }

    @Test
    fun testReset() {
        stats.recordChunk(melMs = 10, encoderMs = 20, totalMs = 30)
        stats.reset()

        assertEquals(0, stats.getChunksProcessed())
        assertEquals(0, stats.getTotalStats().p50)
    }

    @Test
    fun testCompactSummary() {
        stats.recordChunk(melMs = 5, encoderMs = 10, totalMs = 15)
        val summary = stats.compactSummary()
        assertTrue(summary.contains("p50="))
        assertTrue(summary.contains("p95="))
        assertTrue(summary.contains("1 chunks"))
    }

    @Test
    fun testThroughput() {
        // Record chunks and verify throughput is reasonable
        for (i in 1..10) {
            stats.recordChunk(melMs = 1, encoderMs = 1, totalMs = 2)
        }
        val throughput = stats.getThroughput()
        // Should be positive since we processed 10 chunks
        assertTrue("Throughput should be > 0, got $throughput", throughput > 0)
    }
}