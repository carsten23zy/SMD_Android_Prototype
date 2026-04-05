package com.edgy.privacy

import com.edgy.privacy.privacy.PrivacyTier
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for PrivacyTier enum.
 */
class PrivacyTierTest {

    @Test
    fun testTierLevels() {
        assertEquals(0, PrivacyTier.LOW.level)
        assertEquals(1, PrivacyTier.MODERATE.level)
        assertEquals(2, PrivacyTier.HIGH.level)
    }

    @Test
    fun testFromLevel() {
        assertEquals(PrivacyTier.LOW, PrivacyTier.fromLevel(0))
        assertEquals(PrivacyTier.MODERATE, PrivacyTier.fromLevel(1))
        assertEquals(PrivacyTier.HIGH, PrivacyTier.fromLevel(2))
    }

    @Test(expected = NoSuchElementException::class)
    fun testFromLevelInvalid() {
        PrivacyTier.fromLevel(99)
    }

    @Test
    fun testAllTiersPresent() {
        assertEquals(3, PrivacyTier.entries.size)
    }
}
