package ai.vosr.vosr_restore

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import java.io.InputStream
import android.provider.MediaStore
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : FlutterActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val processing = AtomicBoolean(false)
    private var events: EventChannel.EventSink? = null
    private var pickerResult: MethodChannel.Result? = null
    private var pickerKind = 0
    private var pickerBackend = "npu"
    @Volatile private var activeRunner: QnnRunner? = null
    private fun modelDirectory(backend: String) = File(getExternalFilesDir(null), if (backend=="npu") "models" else "models-gpu").apply {
        check(isDirectory || mkdirs()) { "无法创建应用模型目录" }
    }
    private fun selectedBackend(value: String?): String {
        val backend=value ?: "npu"
        require(backend=="npu" || backend=="gpu") { "不支持的后端" }
        return backend
    }
    private fun npuSupported() = Build.VERSION.SDK_INT>=31 && Build.SOC_MODEL.uppercase().contains("SM8550")

    override fun configureFlutterEngine(engine: FlutterEngine) {
        super.configureFlutterEngine(engine)
        EventChannel(engine.dartExecutor.binaryMessenger, "ai.vosr/progress")
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, sink: EventChannel.EventSink) { events = sink }
                override fun onCancel(arguments: Any?) { events = null }
            })
        MethodChannel(engine.dartExecutor.binaryMessenger, "ai.vosr/restore")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "diagnostics" -> background(result) {
                        val backend=selectedBackend(call.argument<String>("backend"))
                        val gpu=try { JSONObject(NativeGpu.probe()) }
                            catch (error: LinkageError) { android.util.Log.w("VOSR","GPU library unavailable",error); JSONObject().put("gpuCount",0) }
                            catch (error: RuntimeException) { android.util.Log.w("VOSR","Vulkan probe failed",error); JSONObject().put("gpuCount",0) }
                        val supported=if (backend=="npu") npuSupported() else gpu.optInt("gpuCount")>0
                        val ready = supported && QnnRunner.verifyBundle(modelDirectory(backend),backend)
                        mapOf("device" to "${Build.MANUFACTURER} ${Build.MODEL}",
                            "ready" to ready,"supported" to supported,"gpu" to gpu.optString("device"),
                            "message" to if (!supported) "此设备不支持当前 ${backend.uppercase()} 后端" else if (ready) "${backend.uppercase()} 模型已就绪" else "请导入 ${backend.uppercase()} 专用模型包")
                    }
                    "pickImage", "importModels" -> {
                        if (pickerResult != null || processing.get()) result.error("BUSY", "正在处理其他任务", null)
                        else {
                            pickerResult = result
                            pickerBackend=selectedBackend(call.argument<String>("backend"))
                            pickerKind = if (call.method == "pickImage") 100 else 101
                            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                                type = if (pickerKind == 100) "image/*" else "application/zip"
                                addCategory(Intent.CATEGORY_OPENABLE)
                            }
                            startActivityForResult(intent, pickerKind)
                        }
                    }
                    "cancel" -> { activeRunner?.cancel(); result.success(null) }
                    "restore" -> background(result, true) {
                        val input = call.argument<String>("path") ?: error("没有选择照片")
                        val backend=selectedBackend(call.argument<String>("backend"))
                        if (backend=="npu") require(npuSupported()) { "当前 NPU 模型仅支持 SM8550 / HTP v73" }
                        val scale = call.argument<Int>("scale") ?: 4
                        require(scale == 2 || scale == 4) { "不支持的放大倍数" }
                        require(File(input).canonicalPath.startsWith(cacheDir.canonicalPath + File.separator)) { "图像路径无效" }
                        val runner = QnnRunner(this, modelDirectory(backend),backend) { value, label -> progress(value, label) }
                        activeRunner = runner
                        try { runner.restore(File(input), scale, (call.argument<Int>("seed") ?: 42).toLong()) }
                        finally { activeRunner = null }
                    }
                    "saveImage" -> background(result) {
                        val file = File(call.argument<String>("path") ?: error("没有可保存的结果"))
                        require(file.canonicalPath.startsWith(cacheDir.canonicalPath + File.separator)) { "结果路径无效" }
                        saveImage(file)
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun saveImage(file: File): String {
        require(file.canonicalPath.startsWith(cacheDir.canonicalPath + File.separator)) { "结果路径无效" }
                        val values = ContentValues().apply {
                            put(MediaStore.Images.Media.DISPLAY_NAME, "VOSR_${System.currentTimeMillis()}.png")
                            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/VOSR")
                            put(MediaStore.Images.Media.IS_PENDING, 1)
                        }
                        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                            ?: error("无法创建相册文件")
                        try {
                            contentResolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
                            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                            contentResolver.update(uri, values, null, null)
                        } catch (error: Exception) {
                            contentResolver.delete(uri, null, null)
                            throw error
                        }
        return uri.toString()
    }

    private fun importImage(source: InputStream): Map<String,Any> {
        val file=File(cacheDir,"input_${System.currentTimeMillis()}.img")
        try {
            file.outputStream().use { source.copyTo(it) }
            return NativeImage(file).use { image ->
                mapOf("path" to file.absolutePath,"width" to image.width,"height" to image.height,
                    "exifOrientation" to image.orientation,"inputDownscaled" to false)
            }
        } catch (error: Throwable) { file.delete(); throw error }
    }

    private fun progress(value: Double, label: String) {
        runOnUiThread { events?.success(mapOf("progress" to value, "message" to label)) }
    }

    private fun background(result: MethodChannel.Result, keepScreenOn: Boolean = false, operation: () -> Any?) {
        if (!processing.compareAndSet(false, true)) {
            result.error("BUSY", "正在处理其他任务", null); return
        }
        if (keepScreenOn) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        worker.execute {
            try {
                val value = operation()
                runOnUiThread { result.success(value) }
            } catch (error: Throwable) {
                android.util.Log.e("VOSR", "Processing failed", error)
                runOnUiThread { result.error("RESTORE_ERROR", error.message ?: error.javaClass.simpleName, null) }
            } finally {
                processing.set(false)
                if (keepScreenOn) runOnUiThread { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
        }
    }

    @Deprecated("Android result API is used by FlutterActivity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickerKind) return
        val result = pickerResult ?: return
        pickerResult = null
        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) { result.success(null); return }
        background(result) {
            if (requestCode == 101) {
                contentResolver.openInputStream(uri)!!.use { stream ->
                    QnnRunner.importBundle(stream, modelDirectory(pickerBackend),pickerBackend) { progress(it, "正在导入 ${pickerBackend.uppercase()} 模型") }
                }
                true
            } else {
                contentResolver.openInputStream(uri)!!.use { importImage(it) }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelDirectory("npu"); modelDirectory("gpu")
        runDebugTest(intent)
    }

    override fun onNewIntent(next: Intent) {
        super.onNewIntent(next)
        setIntent(next)
        runDebugTest(next)
    }

    private fun runDebugTest(intent: Intent) {
        // Explicit debug harness, restricted to this debuggable development APK.
        if ((applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
            val testInput = intent.getStringExtra("vosr_test_input")
            if (testInput != null) {
              window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
              worker.execute {
                val destination = File(getExternalFilesDir(null), "app-test-result.json")
                try {
                    val source = File(testInput)
                    require(source.canonicalPath.startsWith(getExternalFilesDir(null)!!.canonicalPath + File.separator))
                    val imported=source.inputStream().use { importImage(it) }
                    val local=File(imported["path"] as String)
                    val scale=intent.getIntExtra("vosr_test_scale",4)
                    val backend=selectedBackend(intent.getStringExtra("vosr_test_backend"))
                    val detailed=intent.getBooleanExtra("vosr_test_profile",false)
                    val firstProfileStage=intent.getIntExtra("vosr_test_profile_start_stage",0)
                    val result = QnnRunner(this, modelDirectory(backend),backend,detailed,firstProfileStage) { value, label -> progress(value,label) }.restore(local,scale,42).toMutableMap()
                    result["testRunId"]=intent.getStringExtra("vosr_test_run_id") ?: ""
                    if (intent.getBooleanExtra("vosr_test_save",false)) result["galleryUri"]=saveImage(File(result["path"] as String))
                    val shared = File(getExternalFilesDir(null), "app-test-output.png")
                    File(result["path"] as String).copyTo(shared,overwrite=true)
                    destination.writeText(JSONObject(result + ("sharedOutput" to shared.absolutePath)).toString(2))
                    runOnUiThread {
                        events?.success(mapOf("progress" to 1.0,"message" to "真机复原验证完成",
                            "input" to local.absolutePath,"output" to result["path"],"backend" to result["backend"],
                            "inputWidth" to result["inputWidth"],"inputHeight" to result["inputHeight"]))
                    }
                } catch (error: Throwable) {
                    destination.writeText(JSONObject(mapOf("error" to (error.message ?: error.toString()))).toString(2))
                } finally {
                    runOnUiThread { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
                }
              }
            }
        }
    }
}
