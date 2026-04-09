# EDGY Android — Implementation

## 1. Project Context

### 1.1 EDGY

EDGY is a privacy-preserving voice analysis framework.
It strips sensitive paralinguistic information (gender, emotion, speaker identity,
accent, health indicators) from speech using a VQ-VAE encoder whose 512-code
discrete codebook acts as an information bottleneck. Linguistic content passes
through; speaker-specific attributes are destroyed by quantization.

Source: https://github.com/RanyaJumah/EDGY

Papers:
- "Privacy-preserving Voice Analysis via Disentangled Representations" (CCS Workshop 2020)
- "Paralinguistic Privacy Protection at the Edge" (ACM TOPS 2023)

### 1.2 What This App Does

Sits between the device microphone and consuming applications, filtering
paralinguistic information from speech in real-time. Three user-selectable
privacy tiers control what leaves the device.

### 1.3 Privacy Tiers (DDF — Dual-phase Disentangled Filter)

| Tier | What leaves the device | What is stripped |
|------|----------------------|-----------------|
| LOW | Raw audio (passthrough) | Nothing |
| MODERATE | VQ codes + speaker embedding | Emotion, accent, health |
| HIGH | VQ codes only | Everything including speaker identity |

---

## 2. Model Artifacts

Produced by the training Jupyter notebook and placed in `exported_models/`:

```
exported_models/
├── edgy_encoder_fp32.onnx        # ~32 MB — full precision encoder
├── edgy_encoder_int8.onnx        # ~8 MB — INT8 quantized (4× smaller)
├── vq_codebook.npy               # float32 [512, 64] — VQ codebook
├── speaker_embeddings.npy        # float32 [N_speakers, 64] — for MODERATE tier
├── model_config.json             # Preprocessing params + deployment config
└── test_fixtures/
    ├── sample_00_mel.npy          # Reference mel spectrogram [80, T]
    ├── sample_00_vq.npy           # Reference encoder output [1, T', 64]
    ├── sample_00_indices.npy      # Reference codebook indices [T']
    └── ... (10 samples)
```

### ONNX Encoder Interface

```
Input:   "mel_spectrogram"    float32 [batch, 80, T]       (T frames ≈ T/100 seconds)
Output:  "vq_embedding"       float32 [batch, T/2, 64]     (downsampled 2×)
         "codebook_indices"   int64   [batch * T/2]         (which of 512 codes)
```

### model_config.json

All preprocessing parameters live here. The app reads this file — never hardcode
these values. When a retrained model is pushed to the device, the config updates
and the app adapts without recompilation.

```json
{
  "preprocessing": {
    "sample_rate": 16000, "n_fft": 2048, "n_mels": 80,
    "hop_length": 160, "win_length": 400, "fmin": 50,
    "preemph": 0.97, "top_db": 80
  },
  "encoder": {
    "n_embeddings": 512, "embedding_dim": 64, "downsampling_factor": 2
  },
  "files": {
    "encoder_fp32": "edgy_encoder_fp32.onnx",
    "encoder_int8": "edgy_encoder_int8.onnx",
    "codebook": "vq_codebook.npy",
    "speaker_embeddings": "speaker_embeddings.npy"
  },
  "android_onnx_settings": {
    "intra_op_num_threads": 4, "inter_op_num_threads": 2,
    "execution_mode": "ORT_PARALLEL", "optimization_level": "ALL_OPT"
  }
}
```

---

## 3. Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | Kotlin | Coroutines for threading |
| ML Runtime | ONNX Runtime Mobile 1.17+ | CPU + NNAPI execution providers |
| Audio Capture | AudioRecord | 16kHz, mono, PCM 16-bit |
| Audio Output | AudioTrack | PCM output to speaker / REMOTE_SUBMIX |
| FFT | JTransforms 3.1 | Pure Java, no native deps |
| JSON | Gson 2.10+ | Parse model_config.json |
| Build | Gradle + Kotlin DSL | minSdk 26, targetSdk 36 |
| Testing | JUnit + Android Instrumented | Parity tests against Python fixtures |

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
    implementation("com.github.wendykierp:JTransforms:3.1")
    implementation("com.google.code.gson:gson:2.10.1")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}
```

---

## 4. Project Structure

```
edgy-android/
├── app/src/main/
│   ├── java/com/edgy/privacy/
│   │   ├── MainActivity.kt
│   │   ├── audio/
│   │   │   ├── AudioCaptureService.kt
│   │   │   ├── AudioOutputService.kt
│   │   │   ├── AudioChunkBuffer.kt
│   │   │   └── MelSpectrogramExtractor.kt
│   │   ├── ml/
│   │   │   ├── ModelManager.kt
│   │   │   └── EdgyEncoder.kt
│   │   ├── privacy/
│   │   │   ├── PrivacyTier.kt
│   │   │   └── PrivacyPipeline.kt
│   │   ├── vocoder/                          # Stage 5
│   │   │   └── GriffinLimVocoder.kt
│   │   ├── sdk/                              # Stage 6
│   │   │   ├── EdgyAudioProvider.kt
│   │   │   └── IEdgyAudioSource.aidl
│   │   └── util/
│   │       ├── DSP.kt
│   │       ├── NpyReader.kt
│   │       └── WavWriter.kt
│   ├── assets/models/
│   │   ├── edgy_encoder_fp32.onnx
│   │   ├── edgy_encoder_int8.onnx
│   │   ├── vq_codebook.npy
│   │   ├── speaker_embeddings.npy
│   │   └── model_config.json
│   └── res/layout/
│       └── activity_main.xml
├── app/src/test/                             # Unit tests
├── app/src/androidTest/                      # Instrumented tests
├── test_fixtures/                            # From notebook export
└── build.gradle.kts
```

---

## 5. Component Specifications

### 5.1 MelSpectrogramExtractor.kt

Converts raw 16kHz PCM into log-mel spectrograms matching EDGY's Python
preprocessing. This component must produce numerically identical output to
Python's librosa. If it doesn't, the ONNX encoder receives out-of-distribution
input and all downstream output is invalid.

```
Constructor(config: ModelConfig)
  — Precompute mel filterbank matrix [80, 1025] from config params
  — Initialise overlap buffer for streaming mode

extract(pcm: FloatArray): Array<FloatArray>
  — Input: PCM float [-1, 1]
  — Apply preemphasis: y[n] = x[n] - preemph * x[n-1]
  — Frame with hop_length=160, win_length=400
  — Apply Hann window per frame
  — FFT (size 2048) → magnitude spectrum (1025 bins), NOT power spectrum
  — Apply mel filterbank → 80 bins
  — Convert to dB: 10 * log10(mel + 1e-10)
  — Clip: max(mel_db, max(mel_db) - top_db)
  — Normalise: mel_db / top_db + 1
  — Output: [80, T]

extractStreaming(pcmChunk: FloatArray): Array<FloatArray>
  — Same computation but maintains overlap buffer (240 samples) across calls
  — Saves preemphasis state (last sample) for next chunk

Implementation:
  — FFT via JTransforms DoubleFFT_1D.realForward()
  — Mel filterbank: triangular filters via Hz↔Mel conversion, computed once
  — power=1 in the original librosa call means AMPLITUDE (magnitude), not squared
```

### 5.2 ModelManager.kt

Loads ONNX models, config, and auxiliary data. Supports hot-swap from external storage.

```
Constructor(context: Context)
  — Scans assets/models/ then /sdcard/edgy_models/ (external takes priority)

loadConfig(): ModelConfig
  — Parse model_config.json via Gson

loadEncoder(preferInt8: Boolean = false): EdgyEncoder
  — Copy ONNX from assets to cache dir (ORT needs file path)
  — Create OrtSession with config-specified thread settings
  — Return EdgyEncoder wrapper

loadCodebook(): Array<FloatArray>        — Read vq_codebook.npy [512, 64]
loadSpeakerEmbeddings(): Array<FloatArray> — Read speaker_embeddings.npy [N, 64]

reloadFromPath(path: String)
  — Close existing session, load new model + config from path
  — Set intra_op threads to min(config value, available processors)
```

### 5.3 EdgyEncoder.kt

Runs ONNX inference on mel input.

```
Constructor(session: OrtSession, config: ModelConfig)

encode(mel: Array<FloatArray>): EncoderOutput
  — Reshape [80, T] → OnnxTensor [1, 80, T]
  — Run session: input "mel_spectrogram" → outputs "vq_embedding", "codebook_indices"
  — Return EncoderOutput(vqEmbedding, codebookIndices, inferenceTimeMs)

Data class EncoderOutput:
  vqEmbedding: Array<FloatArray>   // [T', 64]
  codebookIndices: IntArray         // [T']
  inferenceTimeMs: Long
```

### 5.4 PrivacyPipeline.kt

Orchestrates processing based on selected tier.

```
Constructor(melExtractor, encoder, speakerEmbeddings, speakerIndex = 0)

setTier(tier: PrivacyTier)
processChunk(pcm: FloatArray): PrivacyOutput
  — LOW:      return PrivacyOutput(rawAudio = pcm)
  — MODERATE: mel → encode → PrivacyOutput(vqEmbedding, speakerEmbedding[index])
  — HIGH:     mel → encode → PrivacyOutput(vqEmbedding)

Data class PrivacyOutput:
  rawAudio: FloatArray?
  vqEmbedding: Array<FloatArray>?
  speakerEmbedding: FloatArray?
  codebookIndices: IntArray?
  tier: PrivacyTier
  processingTimeMs: Long
```

### 5.5 AudioCaptureService.kt

Foreground service capturing microphone via AudioRecord.

```
startCapture(sampleRate: Int = 16000, chunkSizeMs: Int = 100)
  — ENCODING_PCM_16BIT, CHANNEL_IN_MONO
  — Capture thread: read Int16, convert to Float32, push to AudioChunkBuffer
  — chunkSizeMs=100 → 1600 samples per chunk

stopCapture()
setOnChunkReadyListener(callback: (FloatArray) -> Unit)

Requires: RECORD_AUDIO permission, foreground notification
```

### 5.6 AudioOutputService.kt

Routes output to file, speaker, or internal audio bus.

```
writeToWavFile(path: String, pcm: FloatArray, sampleRate: Int)
writeEmbeddingFile(path: String, output: PrivacyOutput)

// Stage 7 (optional):
startSubmixOutput(sampleRate: Int)
  — AudioTrack targeting REMOTE_SUBMIX
writeAudioChunk(pcm: FloatArray)
stopOutput()
```

### 5.7 AudioChunkBuffer.kt

Lock-free ring buffer connecting threads.

```
push(chunk: FloatArray): Boolean    — returns false if full (drop chunk)
pull(): FloatArray?                  — returns null if empty
capacity: Int                        — number of chunks (default: 16)
```

### 5.8 NpyReader.kt

Minimal .npy file parser.

```
readFloatMatrix(inputStream: InputStream): Pair<IntArray, FloatArray>
  — Parse numpy .npy header (magic bytes, shape, dtype)
  — Read raw float32 data
  — Return (shape, flatData)
```

### 5.9 DSP.kt

Signal processing utilities.

```
applyPreemphasis(pcm: FloatArray, coeff: Float, lastSample: Float): Pair<FloatArray, Float>
hannWindow(size: Int): FloatArray
frameSignal(pcm: FloatArray, frameLen: Int, hopLen: Int): Array<FloatArray>
```

---

## 6. Threading Architecture

```
Thread 1 (Capture)  →  RingBuffer A  →  Thread 2 (Processing)  →  RingBuffer B  →  Thread 3 (Output)

Thread 1: AudioRecord read loop. Little core. Fires every 100ms.
Thread 2: Mel extraction + ONNX inference. Big cores. ONNX uses 4 intra-op threads internally.
Thread 3: File write / AudioTrack write. Little core. I/O bound.

Latency budget per 100ms chunk:
  Capture:     ~1ms
  Mel + ONNX:  ~15-20ms (FP32 on phone CPU)
  Output:      ~1ms
  Headroom:    ~78ms
```

---

## 7. Implementation Stages

### Stage 1: Offline Processing Pipeline

**Goal:** Load a .wav file on-device, compute mel spectrogram, run ONNX encoder,
produce VQ output that matches the Python notebook's output.

**Components to build:**
- NpyReader.kt
- DSP.kt
- MelSpectrogramExtractor.kt
- ModelManager.kt
- EdgyEncoder.kt
- Basic MainActivity.kt: button to process a bundled test .wav

**Verification:**
1. Load `test_fixtures/sample_00_mel.npy` (Python reference)
2. Load the corresponding .wav, compute mel in Kotlin
3. Max absolute difference between Kotlin mel and Python mel must be < 1e-3
4. Load computed mel into ONNX encoder
5. Compare output against `sample_00_vq.npy`: cosine similarity > 0.99
6. Compare indices against `sample_00_indices.npy`: match rate > 95%
7. Repeat for all 10 test fixtures — all must pass

**Exit criteria:** All 10 parity tests pass. If mel differs by > 1e-3, fix
MelSpectrogramExtractor before proceeding. Everything downstream depends on this.

---

### Stage 2: Three-Tier Privacy Pipeline

**Goal:** Implement DDF tier logic, process .wav files at each tier, save outputs.

**Components to build:**
- PrivacyTier.kt (enum)
- PrivacyPipeline.kt
- WavWriter.kt (for LOW tier output)
- Embedding serialisation (for MODERATE/HIGH output)
- UI: radio buttons for tier selection, file picker for input .wav

**Verification:**
1. LOW tier: output .wav is bit-identical to input .wav
2. MODERATE tier: output contains vqEmbedding (non-null, shape [T', 64]) AND
   speakerEmbedding (non-null, shape [64])
3. HIGH tier: output contains vqEmbedding (non-null) and speakerEmbedding is null
4. MODERATE and HIGH: vqEmbedding values are identical (same encoder path)
5. Privacy validation: pull MODERATE/HIGH .npz files to PC, run the notebook's
   gender classifier → accuracy must be ~50% (within 5pp of random)
6. Process 50+ utterances at each tier, compare gender attack accuracy:
   LOW raw mel > 70%, MODERATE/HIGH VQ embeddings 45-55%

**Exit criteria:** Three tiers produce correct output types. Gender classifier
on VQ embeddings scores within chance range.

---

### Stage 3: Real-Time Mic Capture

**Goal:** Capture live microphone audio, process through the pipeline in real-time,
save filtered output to files.

**Components to build:**
- AudioCaptureService.kt (foreground service)
- AudioChunkBuffer.kt (ring buffer)
- Streaming mel extraction (overlap buffer between chunks)
- Three-thread pipeline wiring
- UI: start/stop toggle, tier selector, live latency display

**Verification:**
1. Record 10 seconds of speech → output .wav (LOW tier) plays back correctly
2. Record same speech at HIGH tier → .npz file has valid VQ embeddings
3. Streaming parity: process a .wav file in 100ms chunks via extractStreaming(),
   concatenate output, compare against single-pass extract() → max diff < 1e-3
4. No audio glitches: record 60 seconds continuously, verify no dropped chunks
   (log chunk count: expected = duration_seconds * 10)
5. Latency: p50 < 30ms, p95 < 50ms per chunk (measured from chunk-ready callback
   to output-buffer push)
6. Battery: 30 minutes continuous capture at HIGH tier, verify < 10% battery drain

**Exit criteria:** Continuous real-time capture with no dropped chunks and
sub-50ms processing latency.

---

### Stage 4: Model Hot-Swap

**Goal:** Load models from external storage, swap without restart, support
iterative experimentation.

**Components to build:**
- External storage model scanning in ModelManager.kt
- "Reload Model" UI action
- Model info display (name, size, training steps, codebook utilisation)
- Inference stats display (p50/p95 latency, chunks processed)

**Verification:**
1. Push a different model to /sdcard/edgy_models/ via adb
2. Tap Reload → model loads without crash
3. model_config.json is re-read → preprocessing adapts to new params
4. Run privacy validation on output → classifier accuracy still ~50%
5. Push a corrupted .onnx file → app shows error message, continues with
   previous model (graceful fallback)
6. Push model with different n_mels or sample_rate → app adapts preprocessing

**Exit criteria:** Two different model variants loaded and validated without
app recompilation.

---

### Stage 5: Audio Reconstruction (Vocoder)

**Goal:** Reconstruct audible speech from VQ embeddings so MODERATE/HIGH tiers
can output audio, not just embeddings. This is a prerequisite for Stages 6 and 7
— without a vocoder, MODERATE/HIGH tiers produce only embeddings, making audio
routing and SDK integration meaningless for those tiers.

**Approach: Griffin-Lim (classical, no ML)**
- Train a small linear layer (64→80) to project VQ embeddings back to mel space
- Use Griffin-Lim algorithm to reconstruct waveform from estimated mel
- Quality: robotic but intelligible. Latency: ~10ms per chunk. No ONNX model needed.
- Export the 64→80 projection matrix from the notebook as `vq_to_mel_projection.npy`

**Components to build:**
- vocoder/GriffinLimVocoder.kt:
  ```
  Constructor(projectionMatrix: Array<FloatArray>, config: ModelConfig)
    — projectionMatrix: [64, 80] loaded from vq_to_mel_projection.npy
    — Precompute inverse mel filterbank for Griffin-Lim

  reconstruct(vqEmbedding: Array<FloatArray>): FloatArray
    — Project VQ embeddings [T', 64] → estimated mel [80, T']
    — Apply Griffin-Lim iterative phase reconstruction (30 iterations)
    — Return PCM float [-1, 1]

  reconstructStreaming(vqEmbedding: Array<FloatArray>): FloatArray
    — Same as above with overlap-add for streaming chunks
  ```
- Integration with PrivacyPipeline: MODERATE/HIGH tiers reconstruct audio
  after encoding, so PrivacyOutput includes `reconstructedAudio: FloatArray?`
- Output reconstructed audio to file and (later) AudioTrack

**New model artifact:**
```
exported_models/
└── vq_to_mel_projection.npy    # float32 [64, 80] — trained projection matrix
```

**Verification:**
1. Reconstructed speech is intelligible (human listening test)
2. WER (word error rate) of reconstructed vs original < 15% using device ASR
3. Gender classifier on reconstructed audio mel features → ~50% (privacy preserved)
4. Reconstruction latency < 100ms per 100ms chunk (fits within pipeline budget)

**Exit criteria:** MODERATE/HIGH tiers produce audible speech output. Privacy
filtering verified on the reconstructed waveform.

> **Note on alternatives considered:** The EDGY paper's WaveRNN decoder
> (single GRU, hidden=896, autoregressive mu-law at 24kHz) generates one sample
> at a time — expect ~200ms+ per 100ms of audio. This is unsuitable for real-time
> use but could be offered as an offline "high quality export" option in Stage 8.
> HiFi-GAN v3 (~1MB, faster than real-time on mobile) would be the best quality
> option but requires a separate training pipeline conditioned on EDGY's VQ codes.
> This is a future upgrade path — the GriffinLimVocoder interface is designed so
> a NeuralVocoder can be swapped in later without changing the pipeline.

---

### Stage 6: Audio Provider SDK

**Goal:** Package EDGY as a service that other apps on the device can use as
an audio source, similar to how Krisp provides a virtual microphone. This is the
primary integration path — it works on all API levels (minSdk 26+) and gives any
cooperating app direct access to privacy-filtered audio.

**Architecture:**
```
┌──────────────────────────────────────┐
│  EDGY Service (runs in background)   │
│                                       │
│  Mic → Processing → Filtered Audio   │
│              ↓                        │
│  Exposes: IEdgyAudioSource (AIDL)    │
└──────────┬───────────────────────────┘
           │  Binder IPC
           ▼
┌──────────────────────────────────────┐
│  Any App (VoIP client, assistant)    │
│                                       │
│  Binds to EDGY service               │
│  Calls: getAudioStream(tier)         │
│  Receives: filtered PCM stream       │
│  Uses as microphone input            │
└──────────────────────────────────────┘
```

**Components to build:**
- IEdgyAudioSource.aidl — AIDL interface definition:
  ```
  interface IEdgyAudioSource {
      ParcelFileDescriptor getAudioStream(int tier);  // 0=LOW, 1=MOD, 2=HIGH
      int getSampleRate();
      int getChannelCount();
      int getEncoding();
      void setPrivacyTier(int tier);
  }
  ```
- sdk/EdgyAudioProvider.kt — bound service implementing the AIDL interface
- Streams filtered PCM via ParcelFileDescriptor pipe
- Test client activity (within the same app or a minimal test app) for verification

**Verification:**
1. Test client binds to EDGY service
2. Test client calls getAudioStream(HIGH) → receives filtered PCM
3. Feed received PCM into a WebRTC peer connection → remote side hears speech
4. Gender classifier on the remote recording → ~50% accuracy
5. Switch tier mid-stream → audio content changes smoothly
6. EDGY service killed → client detects disconnect, falls back to raw mic
7. Multiple clients bind simultaneously → each receives independent stream

**Exit criteria:** Test client successfully uses EDGY as its audio source
via the AIDL interface. Privacy filtering verified end-to-end.

> **Note:** A packaged AAR client SDK library for easy third-party integration
> is deferred to Stage 8 (Production Hardening). For this stage, the AIDL
> interface and a test client are sufficient to validate the architecture.

---

### Stage 7: REMOTE_SUBMIX Audio Routing (Optional)

**Goal:** Output processed audio to Android's internal audio bus so cooperating
apps can receive it without direct AIDL integration.

> **This stage is optional.** Stage 6 (AIDL SDK) provides a superior integration
> path that works on all API levels. REMOTE_SUBMIX is limited to API 29+ (Android 10),
> requires the receiving app to explicitly use AudioPlaybackCapture, and will not
> work with unmodified Zoom/WhatsApp/Teams. It is useful primarily for demos,
> testing, and scenarios where modifying the receiving app is not possible.

**Components to build:**
- AudioTrack output targeting REMOTE_SUBMIX device type
- MediaProjection permission flow (required for AudioPlaybackCapture)
- Simple test receiver app (minimal app that reads from AudioPlaybackCapture
  and plays to speaker or saves to file)

**Verification:**
1. EDGY app outputs LOW tier audio to REMOTE_SUBMIX
2. Test receiver app captures it via AudioPlaybackCapture → plays audible speech
3. End-to-end latency: measure time from mic input to receiver output < 200ms
4. Test receiver app records 30 seconds → output .wav is intelligible
5. Switch EDGY to HIGH tier → receiver gets reconstructed audio (via Stage 5 vocoder)
6. Test on available Android 10+ devices

**Exit criteria:** Audio successfully routes from EDGY to a separate receiving
app via REMOTE_SUBMIX without root.

---

### Alternatives Considered (Stages 5-7)

The original handover ordered these stages as REMOTE_SUBMIX (5) → Vocoder (6) →
AIDL SDK (7). This was revised based on the following analysis:

**Why Vocoder was moved to Stage 5 (from Stage 6):**
Without a vocoder, MODERATE/HIGH tiers output only VQ embeddings — not audio.
This made the original Stage 5 (REMOTE_SUBMIX audio routing) functionally useless
at those tiers. Moving the vocoder first fixes the dependency: all three tiers
produce audio before any routing or SDK integration is attempted.

**Why WaveRNN was rejected for real-time:**
The EDGY paper's WaveRNN decoder (models.py Decoder class) is autoregressive —
it generates one sample at a time via a single GRU (hidden=896) with mu-law
10-bit output at 24kHz. On mobile CPU, this translates to ~200ms+ per 100ms of
audio, making it fundamentally incompatible with real-time processing. Griffin-Lim
is 20x faster (~10ms) with acceptable quality for a prototype.

**Why AIDL SDK was promoted over REMOTE_SUBMIX:**
- AIDL works on API 26+ (our minSdk); REMOTE_SUBMIX requires API 29+
- AIDL provides explicit app binding — any cooperating app can use it
- REMOTE_SUBMIX requires the receiving app to use AudioPlaybackCapture,
  which rules out unmodified Zoom, WhatsApp, Teams, and Signal
- AIDL supports multiple concurrent clients with independent tier selection
- REMOTE_SUBMIX is retained as Stage 7 (optional) for demos and testing

**Why HiFi-GAN/LPCNet are deferred:**
These lightweight neural vocoders offer the best quality (~1MB model, real-time
on mobile) but require a separate training pipeline conditioned on EDGY's VQ
codes. This is additional ML infrastructure beyond the scope of this prototype.
The GriffinLimVocoder interface is designed so a NeuralVocoder can be swapped in
later without pipeline changes.

---

### Stage 8: Production Hardening

**Goal:** Stability, performance, and UX for real-world daily use.

**Components to build:**
- Persistent settings (selected tier, model path, speaker index)
- Battery optimisation (adaptive chunk size, sleep when no audio detected)
- Notification controls (quick-toggle tier from notification shade)
- Crash reporting and telemetry (opt-in, no audio data collected)
- Model auto-update (check a URL for newer model versions)
- Speaker enrollment flow (record 10 seconds → compute speaker embedding →
  save for MODERATE tier)
- Client SDK library (AAR) wrapping the AIDL binding from Stage 6 for
  easy third-party integration
- WaveRNN offline export option (high-quality batch reconstruction)
- Accessibility service integration (for system-wide audio intercept where
  supported by OEM)

**Verification:**
1. 8-hour soak test: continuous capture, no crashes, no memory leaks
2. Battery: < 3% per hour at HIGH tier
3. Cold start to first processed chunk: < 3 seconds
4. Model update: push new model to server → app downloads and switches
5. Speaker enrollment: enroll, switch to MODERATE, verify speaker embedding
   matches enrolled identity

---

## 8. Validation Approach

### Parity Tests (Stage 1 — run on every build)

```
For each of the 10 test fixtures:
  1. Kotlin mel vs Python mel:        max absolute diff < 1e-3
  2. Kotlin ONNX vq vs Python vq:     cosine similarity > 0.99
  3. Kotlin indices vs Python indices: match rate > 95%

If any parity test fails, the build is broken. Fix MelSpectrogramExtractor.
```

### Privacy Tests (Stage 2+ — run after model changes)

```
1. Process 200+ utterances at each tier
2. Pull VQ embeddings to PC
3. Run notebook's gender classifier (LR, RF, SVM, MLP)
4. Expected results:
   — LOW (raw mel features):    > 70% gender accuracy
   — MODERATE (VQ embeddings):  45-55% (random chance)
   — HIGH (VQ embeddings):      45-55% (random chance)
5. If MODERATE/HIGH scores > 60%: model or preprocessing is broken
```

### Latency Tests (Stage 3+ — run on target devices)

```
1. Process 100 real-time chunks, log timestamps at each stage
2. Report p50, p95, p99 for:
   — Mel extraction
   — ONNX encoder inference
   — Total pipeline
3. Targets:
   — Mel: p95 < 10ms
   — ONNX: p95 < 30ms
   — Total: p95 < 50ms
4. Test on flagship (Snapdragon 8 Gen 3) and mid-range (Snapdragon 6 Gen 1)
```

### Integration Tests (Stage 6+)

```
1. AIDL service: test client binds → receives stream → plays audio
2. Concurrent clients: 2 apps bound simultaneously → both receive audio
3. End-to-end latency through full routing chain: < 200ms
4. REMOTE_SUBMIX (Stage 7, optional): EDGY → test receiver app → verify audio received
```

---

## 9. Model Swapping Workflow

```
1. Modify training params in notebook §0
2. Re-run training (§4) — resumes from checkpoint
3. Re-run edge optimization (§8) — produces INT8 ONNX
4. Re-run export (§11) — produces new exported_models/
5. adb push exported_models/ /sdcard/edgy_models/
6. App: tap Reload or restart
7. ModelManager reads new model_config.json, loads new ONNX session
8. Run privacy validation → confirm filtering still works

No Android recompilation needed for model changes.
```

---

## 10. Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| No Android virtual mic API | Cannot intercept arbitrary apps | AIDL service SDK (Stage 6, primary); REMOTE_SUBMIX for cooperating apps (Stage 7, optional) |
| Mel must match Python exactly | Wrong mel = garbage encoder output | Parity tests against Python fixtures; gate all builds on these tests |
| ONNX model tied to preprocessing | Different training params = different config | model_config.json keeps app in sync; app never hardcodes params |
| Speaker index for MODERATE | App needs to know which embedding to use | Default to index 0; speaker enrollment flow in Stage 8 |
| WaveRNN decoder is slow | Cannot reconstruct audio in real-time | Encoder-only through Stage 4; Griffin-Lim vocoder in Stage 5; neural vocoder (HiFi-GAN) as future upgrade |
| REMOTE_SUBMIX requires receiver opt-in | Won't work with unmodified Zoom/WhatsApp | AIDL service SDK (Stage 6) lets any app integrate; file output always works |
