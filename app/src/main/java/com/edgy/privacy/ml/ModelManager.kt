package com.edgy.privacy.ml

import android.content.Context
import android.os.Environment
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.edgy.privacy.audio.MelSpectrogramExtractor
import com.edgy.privacy.util.NpyReader
import com.google.gson.Gson
import java.io.File
import java.io.FileInputStream

/**
 * Loads ONNX models, config, and auxiliary data.
 * Supports hot-swap from external storage (/sdcard/edgy_models/).
 *
 * Stage 4 additions:
 * - scanExternalModels(): discovers model directories on external storage
 * - reloadModel(): hot-swap model without app restart
 * - getDetailedModelInfo(): model name, size, codebook utilisation
 * - Graceful fallback: if a corrupted model is loaded, keeps previous model
 *
 * Priority: external storage > bundled assets.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val ASSETS_MODEL_DIR = "models"
        private const val EXTERNAL_MODEL_DIR = "edgy_models"
        private const val CONFIG_FILENAME = "model_config.json"
    }

    private val gson = Gson()
    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var currentConfig: ModelConfig? = null
    private var currentEncoder: EdgyEncoder? = null
    private var currentModelPath: String? = null
    private var currentModelSource: ModelSource = ModelSource.BUNDLED
    private var currentCodebook: Array<FloatArray>? = null
    private var currentSpeakerEmbeddings: Array<FloatArray>? = null

    // For hot-swap fallback
    private var previousEncoder: EdgyEncoder? = null
    private var previousConfig: ModelConfig? = null

    enum class ModelSource { BUNDLED, EXTERNAL }

    /**
     * Load model_config.json from the active model directory.
     * Checks external storage first, falls back to assets.
     */
    fun loadConfig(): ModelConfig {
        currentConfig?.let { return it }

        val externalConfig = getExternalConfigFile()
        val configJson = if (externalConfig != null && externalConfig.exists()) {
            currentModelSource = ModelSource.EXTERNAL
            externalConfig.readText()
        } else {
            currentModelSource = ModelSource.BUNDLED
            context.assets.open("$ASSETS_MODEL_DIR/$CONFIG_FILENAME")
                .bufferedReader().use { it.readText() }
        }

        val config = gson.fromJson(configJson, ModelConfig::class.java)
        currentConfig = config
        return config
    }

    /**
     * Load the ONNX encoder model.
     * Copies model from assets/external to cache dir (ONNX Runtime needs a file path).
     *
     * @param preferInt8 Use INT8 quantized model if available (4x smaller, slightly less accurate)
     * @return EdgyEncoder wrapping the ONNX session
     */
    fun loadEncoder(preferInt8: Boolean = false): EdgyEncoder {
        currentEncoder?.let { return it }

        val config = loadConfig()
        val modelFilename = if (preferInt8) config.files.encoderInt8 else config.files.encoderFp32

        val modelFile = resolveModelFile(modelFilename)
            ?: throw IllegalStateException("ONNX model not found: $modelFilename")

        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(
                minOf(config.onnxSettings.intraOpNumThreads, Runtime.getRuntime().availableProcessors())
            )
            setInterOpNumThreads(config.onnxSettings.interOpNumThreads)

            when (config.onnxSettings.optimizationLevel) {
                "ALL_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                "BASIC_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                "EXTENDED_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
                else -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
        }

        val session = ortEnvironment.createSession(modelFile.absolutePath, sessionOptions)
        val encoder = EdgyEncoder(ortEnvironment, session, config)
        currentEncoder = encoder
        currentModelPath = modelFile.absolutePath
        return encoder
    }

    /**
     * Load VQ codebook from vq_codebook.npy [512, 64].
     */
    fun loadCodebook(): Array<FloatArray> {
        currentCodebook?.let { return it }
        val config = loadConfig()
        val stream = resolveModelStream(config.files.codebook)
            ?: throw IllegalStateException("Codebook not found: ${config.files.codebook}")
        val codebook = stream.use { NpyReader.read(it).toFloatMatrix() }
        currentCodebook = codebook
        return codebook
    }

    /**
     * Load speaker embeddings from speaker_embeddings.npy [N, 64].
     */
    fun loadSpeakerEmbeddings(): Array<FloatArray> {
        currentSpeakerEmbeddings?.let { return it }
        val config = loadConfig()
        val stream = resolveModelStream(config.files.speakerEmbeddings)
            ?: throw IllegalStateException("Speaker embeddings not found: ${config.files.speakerEmbeddings}")
        val embeddings = stream.use { NpyReader.read(it).toFloatMatrix() }
        currentSpeakerEmbeddings = embeddings
        return embeddings
    }

    // ──────────────────────────────────────────────
    // Stage 4: Hot-swap and external model scanning
    // ──────────────────────────────────────────────

    /**
     * Scan external storage for available model directories.
     * Each directory should contain model_config.json + ONNX model files.
     *
     * Checks both /sdcard/edgy_models/ and app-specific external storage.
     *
     * @return List of ExternalModelInfo for discovered models
     */
    fun scanExternalModels(): List<ExternalModelInfo> {
        val results = mutableListOf<ExternalModelInfo>()

        // Scan /sdcard/edgy_models/ (primary location for adb push)
        val sdcardDir = File(Environment.getExternalStorageDirectory(), EXTERNAL_MODEL_DIR)
        scanDirectory(sdcardDir, results)

        // Scan app-specific external storage
        val appExternalDir = context.getExternalFilesDir(EXTERNAL_MODEL_DIR)
        if (appExternalDir != null) {
            scanDirectory(appExternalDir, results)
        }

        Log.i(TAG, "Found ${results.size} external model(s)")
        return results
    }

    private fun scanDirectory(dir: File, results: MutableList<ExternalModelInfo>) {
        if (!dir.exists() || !dir.isDirectory) return

        // Check if the directory itself contains models
        val configFile = File(dir, CONFIG_FILENAME)
        if (configFile.exists()) {
            tryAddModelInfo(dir, configFile, results)
        }

        // Check subdirectories
        dir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val subConfig = File(subDir, CONFIG_FILENAME)
            if (subConfig.exists()) {
                tryAddModelInfo(subDir, subConfig, results)
            }
        }
    }

    private fun tryAddModelInfo(dir: File, configFile: File, results: MutableList<ExternalModelInfo>) {
        try {
            val config = gson.fromJson(configFile.readText(), ModelConfig::class.java)
            val onnxFp32 = File(dir, config.files.encoderFp32)
            val onnxInt8 = File(dir, config.files.encoderInt8)
            val hasOnnx = onnxFp32.exists() || onnxInt8.exists()

            results.add(
                ExternalModelInfo(
                    path = dir.absolutePath,
                    name = dir.name,
                    config = config,
                    hasFp32 = onnxFp32.exists(),
                    hasInt8 = onnxInt8.exists(),
                    fp32SizeBytes = if (onnxFp32.exists()) onnxFp32.length() else 0,
                    int8SizeBytes = if (onnxInt8.exists()) onnxInt8.length() else 0,
                    isValid = hasOnnx
                )
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse model at ${dir.absolutePath}: ${e.message}")
            results.add(
                ExternalModelInfo(
                    path = dir.absolutePath,
                    name = dir.name,
                    config = null,
                    isValid = false,
                    error = e.message
                )
            )
        }
    }

    /**
     * Reload model from a specified directory path.
     *
     * Graceful fallback: if loading fails, the previous model is kept.
     * The caller can continue using the previous encoder if this throws.
     *
     * @param path Directory containing model_config.json and ONNX files
     * @param preferInt8 Whether to prefer the INT8 quantized model
     * @return ReloadResult with success status and new config/encoder
     */
    fun reloadModel(path: String, preferInt8: Boolean = false): ReloadResult {
        Log.i(TAG, "Reloading model from: $path")

        // Save previous state for fallback
        previousEncoder = currentEncoder
        previousConfig = currentConfig

        try {
            // Parse new config
            val configFile = File(path, CONFIG_FILENAME)
            if (!configFile.exists()) {
                return ReloadResult(
                    success = false,
                    error = "model_config.json not found at $path"
                )
            }

            val newConfig = gson.fromJson(configFile.readText(), ModelConfig::class.java)

            // Find ONNX model file
            val modelFilename = if (preferInt8) newConfig.files.encoderInt8 else newConfig.files.encoderFp32
            val modelFile = File(path, modelFilename)
            if (!modelFile.exists()) {
                // Try the other variant
                val altFilename = if (preferInt8) newConfig.files.encoderFp32 else newConfig.files.encoderInt8
                val altFile = File(path, altFilename)
                if (!altFile.exists()) {
                    return ReloadResult(
                        success = false,
                        error = "ONNX model not found: $modelFilename (or $altFilename)"
                    )
                }
                return loadNewSession(altFile, newConfig, path)
            }

            return loadNewSession(modelFile, newConfig, path)

        } catch (e: Exception) {
            Log.e(TAG, "Model reload failed, keeping previous model", e)
            // Rollback
            currentEncoder = previousEncoder
            currentConfig = previousConfig
            previousEncoder = null
            previousConfig = null
            return ReloadResult(success = false, error = e.message ?: "Unknown error")
        }
    }

    private fun loadNewSession(modelFile: File, newConfig: ModelConfig, path: String): ReloadResult {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(
                minOf(newConfig.onnxSettings.intraOpNumThreads, Runtime.getRuntime().availableProcessors())
            )
            setInterOpNumThreads(newConfig.onnxSettings.interOpNumThreads)
            when (newConfig.onnxSettings.optimizationLevel) {
                "ALL_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                "BASIC_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                "EXTENDED_OPT" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
                else -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
        }

        val newSession = ortEnvironment.createSession(modelFile.absolutePath, sessionOptions)
        val newEncoder = EdgyEncoder(ortEnvironment, newSession, newConfig)

        // Close previous encoder
        previousEncoder?.close()
        previousEncoder = null
        previousConfig = null

        // Set new state
        currentEncoder = newEncoder
        currentConfig = newConfig
        currentModelPath = modelFile.absolutePath
        currentModelSource = ModelSource.EXTERNAL
        currentCodebook = null
        currentSpeakerEmbeddings = null

        // Load new codebook and speaker embeddings
        try {
            val codebookFile = File(path, newConfig.files.codebook)
            if (codebookFile.exists()) {
                currentCodebook = FileInputStream(codebookFile).use {
                    NpyReader.read(it).toFloatMatrix()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load codebook from new model: ${e.message}")
        }

        try {
            val spkFile = File(path, newConfig.files.speakerEmbeddings)
            if (spkFile.exists()) {
                currentSpeakerEmbeddings = FileInputStream(spkFile).use {
                    NpyReader.read(it).toFloatMatrix()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load speaker embeddings from new model: ${e.message}")
        }

        // Create new MelSpectrogramExtractor with updated config
        val newMelExtractor = MelSpectrogramExtractor(newConfig)

        Log.i(TAG, "Model reloaded successfully from $path")
        return ReloadResult(
            success = true,
            config = newConfig,
            encoder = newEncoder,
            melExtractor = newMelExtractor,
            codebook = currentCodebook,
            speakerEmbeddings = currentSpeakerEmbeddings,
            modelSizeBytes = modelFile.length()
        )
    }

    /**
     * Reload from bundled assets (reset to default).
     */
    fun reloadBundledModel(preferInt8: Boolean = false): ReloadResult {
        close()
        currentConfig = null
        currentCodebook = null
        currentSpeakerEmbeddings = null
        currentModelSource = ModelSource.BUNDLED

        return try {
            val config = loadConfig()
            val encoder = loadEncoder(preferInt8)
            val codebook = try { loadCodebook() } catch (e: Exception) { null }
            val spkEmb = try { loadSpeakerEmbeddings() } catch (e: Exception) { null }
            val melExtractor = MelSpectrogramExtractor(config)

            ReloadResult(
                success = true,
                config = config,
                encoder = encoder,
                melExtractor = melExtractor,
                codebook = codebook,
                speakerEmbeddings = spkEmb
            )
        } catch (e: Exception) {
            ReloadResult(success = false, error = e.message ?: "Unknown error")
        }
    }

    /**
     * Get information about the currently loaded model.
     */
    fun getModelInfo(): String {
        val config = currentConfig ?: return "No model loaded"
        return buildString {
            append("Source: ${currentModelSource.name.lowercase()}\n")
            append("Codebook: ${config.encoder.nEmbeddings} codes x ${config.encoder.embeddingDim}d\n")
            append("Sample rate: ${config.preprocessing.sampleRate} Hz\n")
            append("Mel bins: ${config.preprocessing.nMels}\n")
            append("FFT size: ${config.preprocessing.nFft}\n")
            append("ONNX threads: ${config.onnxSettings.intraOpNumThreads} intra, ${config.onnxSettings.interOpNumThreads} inter")
        }
    }

    /**
     * Stage 4: Get detailed model info including file sizes and codebook utilisation.
     */
    fun getDetailedModelInfo(): DetailedModelInfo {
        val config = currentConfig
        val modelPath = currentModelPath

        val modelSizeBytes = if (modelPath != null) {
            File(modelPath).let { if (it.exists()) it.length() else 0L }
        } else 0L

        val codebookUtil = currentCodebook?.let { cb ->
            // Measure how many codebook vectors are non-zero / unique
            val nonZero = cb.count { row -> row.any { it != 0f } }
            nonZero.toFloat() / cb.size
        }

        return DetailedModelInfo(
            source = currentModelSource,
            path = modelPath,
            config = config,
            modelSizeBytes = modelSizeBytes,
            codebookUtilisation = codebookUtil,
            numSpeakers = currentSpeakerEmbeddings?.size ?: 0
        )
    }

    /**
     * Get the currently loaded encoder (or null).
     */
    fun getCurrentEncoder(): EdgyEncoder? = currentEncoder

    /**
     * Get the current config (or null).
     */
    fun getCurrentConfig(): ModelConfig? = currentConfig

    /**
     * Get speaker embeddings (cached).
     */
    fun getCachedSpeakerEmbeddings(): Array<FloatArray>? = currentSpeakerEmbeddings

    /**
     * Close all open sessions and release resources.
     */
    fun close() {
        currentEncoder?.close()
        currentEncoder = null
        previousEncoder?.close()
        previousEncoder = null
    }

    /**
     * Resolve a model file by name, checking external storage first, then assets.
     * For assets, copies to cache directory since ONNX Runtime requires file paths.
     */
    private fun resolveModelFile(filename: String): File? {
        // Check external storage first
        val externalFile = getExternalModelFile(filename)
        if (externalFile != null && externalFile.exists()) {
            return externalFile
        }

        // Copy from assets to cache
        return try {
            val cacheFile = File(context.cacheDir, filename)
            if (!cacheFile.exists()) {
                context.assets.open("$ASSETS_MODEL_DIR/$filename").use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            cacheFile
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Resolve a model stream by name, checking external storage first, then assets.
     */
    private fun resolveModelStream(filename: String): java.io.InputStream? {
        // Check external storage first
        val externalFile = getExternalModelFile(filename)
        if (externalFile != null && externalFile.exists()) {
            return FileInputStream(externalFile)
        }

        // Fall back to assets
        return try {
            context.assets.open("$ASSETS_MODEL_DIR/$filename")
        } catch (e: Exception) {
            null
        }
    }

    private fun getExternalModelFile(filename: String): File? {
        // Check /sdcard/edgy_models/
        val sdcardFile = File(
            Environment.getExternalStorageDirectory(),
            "$EXTERNAL_MODEL_DIR/$filename"
        )
        if (sdcardFile.exists()) return sdcardFile

        // Check app-specific external storage
        val appExternalDir = context.getExternalFilesDir(EXTERNAL_MODEL_DIR)
        if (appExternalDir != null) {
            val appFile = File(appExternalDir, filename)
            if (appFile.exists()) return appFile
        }

        return null
    }

    private fun getExternalConfigFile(): File? {
        return getExternalModelFile(CONFIG_FILENAME)
    }

    // ─── Data classes ───

    data class ExternalModelInfo(
        val path: String,
        val name: String,
        val config: ModelConfig? = null,
        val hasFp32: Boolean = false,
        val hasInt8: Boolean = false,
        val fp32SizeBytes: Long = 0,
        val int8SizeBytes: Long = 0,
        val isValid: Boolean = false,
        val error: String? = null
    ) {
        fun summary(): String = buildString {
            append("$name (${if (isValid) "valid" else "INVALID"})\n")
            append("  Path: $path\n")
            if (config != null) {
                append("  Codebook: ${config.encoder.nEmbeddings} x ${config.encoder.embeddingDim}\n")
                append("  Sample rate: ${config.preprocessing.sampleRate}Hz\n")
                if (hasFp32) append("  FP32: ${"%.1f".format(fp32SizeBytes / 1024.0 / 1024.0)} MB\n")
                if (hasInt8) append("  INT8: ${"%.1f".format(int8SizeBytes / 1024.0 / 1024.0)} MB\n")
            }
            error?.let { append("  Error: $it\n") }
        }
    }

    data class DetailedModelInfo(
        val source: ModelSource,
        val path: String?,
        val config: ModelConfig?,
        val modelSizeBytes: Long = 0,
        val codebookUtilisation: Float? = null,
        val numSpeakers: Int = 0
    ) {
        fun summary(): String {
            val config = this.config ?: return "No model loaded"
            return buildString {
                append("Source: ${source.name.lowercase()}\n")
                path?.let { append("Path: $it\n") }
                append("Size: ${"%.1f".format(modelSizeBytes / 1024.0 / 1024.0)} MB\n")
                append("Codebook: ${config.encoder.nEmbeddings} codes x ${config.encoder.embeddingDim}d\n")
                codebookUtilisation?.let {
                    append("Codebook utilisation: ${"%.1f".format(it * 100)}%\n")
                }
                append("Speakers: $numSpeakers\n")
                append("Sample rate: ${config.preprocessing.sampleRate} Hz\n")
                append("Mel bins: ${config.preprocessing.nMels}, FFT: ${config.preprocessing.nFft}\n")
                append("ONNX threads: ${config.onnxSettings.intraOpNumThreads} intra, ${config.onnxSettings.interOpNumThreads} inter")
            }
        }
    }

    data class ReloadResult(
        val success: Boolean,
        val error: String? = null,
        val config: ModelConfig? = null,
        val encoder: EdgyEncoder? = null,
        val melExtractor: MelSpectrogramExtractor? = null,
        val codebook: Array<FloatArray>? = null,
        val speakerEmbeddings: Array<FloatArray>? = null,
        val modelSizeBytes: Long = 0
    )
}