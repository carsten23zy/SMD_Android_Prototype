package com.edgy.privacy.ml

import android.content.Context
import ai.onnxruntime.OnnxRuntime
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.edgy.privacy.util.NpyReader
import com.google.gson.Gson
import java.io.File
import java.io.FileInputStream

/**
 * Loads ONNX models, config, and auxiliary data.
 * Supports hot-swap from external storage (/sdcard/edgy_models/).
 *
 * Priority: external storage > bundled assets.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val ASSETS_MODEL_DIR = "models"
        private const val EXTERNAL_MODEL_DIR = "edgy_models"
        private const val CONFIG_FILENAME = "model_config.json"
    }

    private val gson = Gson()
    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var currentConfig: ModelConfig? = null
    private var currentEncoder: EdgyEncoder? = null
    private var currentModelPath: String? = null

    /**
     * Load model_config.json from the active model directory.
     * Checks external storage first, falls back to assets.
     */
    fun loadConfig(): ModelConfig {
        currentConfig?.let { return it }

        val externalConfig = getExternalConfigFile()
        val configJson = if (externalConfig != null && externalConfig.exists()) {
            externalConfig.readText()
        } else {
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
        val encoder = EdgyEncoder(session, config)
        currentEncoder = encoder
        currentModelPath = modelFile.absolutePath
        return encoder
    }

    /**
     * Load VQ codebook from vq_codebook.npy [512, 64].
     */
    fun loadCodebook(): Array<FloatArray> {
        val config = loadConfig()
        val stream = resolveModelStream(config.files.codebook)
            ?: throw IllegalStateException("Codebook not found: ${config.files.codebook}")
        return stream.use { NpyReader.read(it).toFloatMatrix() }
    }

    /**
     * Load speaker embeddings from speaker_embeddings.npy [N, 64].
     */
    fun loadSpeakerEmbeddings(): Array<FloatArray> {
        val config = loadConfig()
        val stream = resolveModelStream(config.files.speakerEmbeddings)
            ?: throw IllegalStateException("Speaker embeddings not found: ${config.files.speakerEmbeddings}")
        return stream.use { NpyReader.read(it).toFloatMatrix() }
    }

    /**
     * Reload all models from a specified path.
     * Closes existing sessions and reloads from the new path.
     */
    fun reloadFromPath(path: String) {
        close()
        currentConfig = null

        val configFile = File(path, CONFIG_FILENAME)
        require(configFile.exists()) { "model_config.json not found at $path" }

        val configJson = configFile.readText()
        currentConfig = gson.fromJson(configJson, ModelConfig::class.java)
    }

    /**
     * Get information about the currently loaded model.
     */
    fun getModelInfo(): String {
        val config = currentConfig ?: return "No model loaded"
        val source = if (currentModelPath?.contains(EXTERNAL_MODEL_DIR) == true) {
            "external"
        } else {
            "bundled"
        }
        return buildString {
            append("Source: $source\n")
            append("Codebook: ${config.encoder.nEmbeddings} codes × ${config.encoder.embeddingDim}d\n")
            append("Sample rate: ${config.preprocessing.sampleRate} Hz\n")
            append("Mel bins: ${config.preprocessing.nMels}\n")
            append("FFT size: ${config.preprocessing.nFft}\n")
            append("ONNX threads: ${config.onnxSettings.intraOpNumThreads} intra, ${config.onnxSettings.interOpNumThreads} inter")
        }
    }

    /**
     * Close all open sessions and release resources.
     */
    fun close() {
        currentEncoder?.close()
        currentEncoder = null
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
        val externalDir = context.getExternalFilesDir(null)?.parentFile
            ?: return null
        return File(File(externalDir, EXTERNAL_MODEL_DIR), filename)
    }

    private fun getExternalConfigFile(): File? {
        return getExternalModelFile(CONFIG_FILENAME)
    }
}
