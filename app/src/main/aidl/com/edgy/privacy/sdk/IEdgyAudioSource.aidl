package com.edgy.privacy.sdk;

import android.os.ParcelFileDescriptor;

/**
 * AIDL interface for EDGY Audio Provider SDK.
 *
 * Client apps bind to EdgyAudioProvider and call openStream() to receive
 * a ParcelFileDescriptor that delivers privacy-filtered PCM audio.
 *
 * Audio format: 16kHz, mono, 16-bit PCM (little-endian).
 */
interface IEdgyAudioSource {

    /**
     * Open a streaming audio pipe.
     *
     * @param privacyTier 0=LOW (passthrough), 1=MODERATE (VQ+vocoder), 2=HIGH (VQ-only+vocoder)
     * @return ParcelFileDescriptor read-end of the audio pipe (null if capture not active)
     */
    ParcelFileDescriptor openStream(int privacyTier);

    /**
     * Close an active audio stream.
     */
    void closeStream();

    /**
     * Check if the audio capture service is currently active.
     */
    boolean isCapturing();

    /**
     * Get the current privacy tier (0=LOW, 1=MODERATE, 2=HIGH).
     */
    int getCurrentTier();

    /**
     * Get the audio sample rate in Hz.
     */
    int getSampleRate();
}
