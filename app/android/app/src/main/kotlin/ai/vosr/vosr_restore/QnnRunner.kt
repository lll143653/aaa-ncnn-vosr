package ai.vosr.vosr_restore

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.util.UUID
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Persistent in-process HTP contexts; no subprocess or intermediate tensor files. */
class QnnRunner(private val context: Context, private val models: File, private val backend: String = "npu",
                private val nativeProfile: Boolean = false,
                private val firstProfileStage: Int = 0,
                private val progress: (Double, String) -> Unit) {
    companion object {
        private var verifiedPath: String? = null
        private var verifiedStamp: Long = 0

        private fun sha(file: File): String {
            val hash = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { source ->
                val buffer = ByteArray(1024 * 1024)
                while (true) { val n = source.read(buffer); if (n < 0) break; hash.update(buffer, 0, n) }
            }
            return hash.digest().joinToString("") { "%02x".format(it) }
        }

        @Synchronized fun verifyBundle(directory: File, backend: String = "npu",expectedTile: Int?=null): Boolean {
            val manifest = File(directory, "manifest.json")
            if (!manifest.isFile) {
                android.util.Log.e("VOSR", "Model manifest unavailable: ${manifest.absolutePath}; dir=${directory.exists()}; readable=${directory.canRead()}")
                return false
            }
            val tile=try { JSONObject(manifest.readText()).getInt("tileSize") } catch(_: Exception) { return false }
            if(tile !in listOf(512,1024) || (expectedTile!=null && tile!=expectedTile)) return false
            if (verifiedPath == directory.absolutePath && verifiedStamp == manifest.lastModified()) return true
            try {
                val data = JSONObject(manifest.readText())
                if(backend=="npu") require(NpuTarget.current()?.accepts(data)==true) { "模型芯片与当前设备不匹配" }
                else require(data.getString("backend")=="NCNN_VULKAN_FP16_WEIGHTS_FP32")
                if (backend=="gpu") require(data.getString("model")=="VOSR_0.5B_os")
                require(data.getInt("tileSize") in listOf(512,1024))
                require(data.getBoolean(if (backend=="npu") "deviceValidated" else "conversionValidated"))
                val stages = data.getJSONArray("stages")
                require(stages.length() == 12)
                for (i in 0 until stages.length()) {
                    val stage = stages.getJSONObject(i)
                    val name = stage.getString("file")
                    require(name.matches(Regex("[a-z0-9_]+(\\.ncnn)?\\.bin"))) { "Unexpected model filename: $name" }
                    val file = File(directory, name)
                    require(file.isFile && file.canRead()) { "Model file unavailable: ${file.absolutePath}" }
                    val actual = sha(file)
                    require(actual == stage.getString("sha256")) { "SHA mismatch $name: $actual" }
                    if (backend=="gpu") {
                        val param=stage.getString("param")
                        require(param.matches(Regex("[a-z0-9_]+\\.ncnn\\.param")))
                        require(sha(File(directory,param))==stage.getString("paramSha256"))
                    }
                }
                verifiedPath = directory.absolutePath; verifiedStamp = manifest.lastModified()
                return true
            } catch (error: Exception) {
                android.util.Log.e("VOSR", "Model verification failed in ${directory.absolutePath}",error)
                return false
            }
        }

        fun importBundle(source: InputStream, target: File, backend: String, expectedTile: Int=512, update: (Double) -> Unit) {
            val temporary = File(target.parentFile, "models-import-${UUID.randomUUID()}")
            temporary.mkdirs()
            try {
                var total = 0L
                ZipInputStream(source).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        require(entry.name.matches(Regex("(manifest\\.json|[a-z0-9_]+(\\.ncnn)?\\.bin|[a-z0-9_]+\\.ncnn\\.param)"))) { "模型包包含不支持的文件" }
                        val file = File(temporary,entry.name)
                        require(!file.exists()) { "模型包包含重复文件" }
                        file.outputStream().use { out ->
                            val buffer = ByteArray(1024*1024)
                            while (true) {
                                val n = zip.read(buffer); if (n < 0) break
                                total += n
                                require(total <= 3L*1024*1024*1024) { "模型包过大" }
                                out.write(buffer,0,n)
                                update(minOf(.9,total/(1.5*1024*1024*1024)))
                            }
                        }
                        zip.closeEntry()
                    }
                }
                require(verifyBundle(temporary,backend,expectedTile)) { "模型文件校验失败，或芯片、后端、输出尺寸不匹配" }
                ResidentEngine.close()
                val backup = File(target.parentFile,"models-previous-${UUID.randomUUID()}")
                if (target.exists()) require(target.renameTo(backup)) { "无法替换旧模型" }
                if (!temporary.renameTo(target)) { backup.renameTo(target); error("无法保存模型") }
                backup.deleteRecursively()
                verifiedPath = null; verifiedStamp = 0
                update(1.0)
            } finally { temporary.deleteRecursively() }
        }
    }

    private val timing = mutableListOf<Map<String, Any>>()
    private val nativePatchTimes = mutableListOf<Double>()
    private val nativePatchProfiles = mutableListOf<String>()
    private val cancelled = java.util.concurrent.atomic.AtomicBoolean(false)
    @Volatile private var engineHandle=0L
    fun isCancelled()=cancelled.get()
    fun cancel() {
        cancelled.set(true); val handle=engineHandle
        if (handle!=0L) { if (backend=="npu") NativeQnn.cancel(handle) else NativeGpu.cancel(handle) }
    }

    private fun stats() = if (backend=="npu") NativeQnn.stats(engineHandle) else NativeGpu.stats(engineHandle)

    private fun infer(rgb: FloatArray,dino: FloatArray,noise: FloatArray): FloatArray {
        if (cancelled.get()) throw java.util.concurrent.CancellationException("已取消处理")
        val output=if (backend=="npu") NativeQnn.run(engineHandle,rgb,dino,noise) else NativeGpu.run(engineHandle,rgb,dino,noise)
        if (cancelled.get()) throw java.util.concurrent.CancellationException("已取消处理")
        val stats=JSONObject(stats())
        val times=stats.getJSONArray("stageMs")
        val names=listOf("encoder","dino","dit_prepare","dit_blocks_00","dit_blocks_04","dit_blocks_08","dit_blocks_12","dit_blocks_16","dit_blocks_20","dit_blocks_24","dit_finish","decoder")
        for (i in names.indices) timing.add(mapOf("stage" to names[i],"elapsedMs" to times.getDouble(i),"patch" to nativePatchTimes.size+1))
        nativePatchTimes.add(stats.getDouble("lastPatchMs"))
        if (nativeProfile) nativePatchProfiles.add(stats.toString())
        return output
    }

    fun restore(input: File, scale: Int, seed: Long, overlap: Int=64): Map<String,Any> {
        val started=System.nanoTime()
        require(backend=="npu" || backend=="gpu") { "不支持的后端" }
        require(verifyBundle(models,backend)) { "模型缺失或校验失败，请重新导入所选后端的模型包" }
        val verifiedMs=(System.nanoTime()-started)/1_000_000
        progress(0.0,"正在准备 ${backend.uppercase()} 常驻模型，首次使用需要加载")
        val lease=ResidentEngine.acquire(context,models,backend,nativeProfile,firstProfileStage)
        engineHandle=lease.handle
        if (backend=="npu") NativeQnn.profile(lease.handle,nativeProfile) else NativeGpu.profile(lease.handle,nativeProfile)
        if (backend=="npu") NativeQnn.resetCancel(lease.handle) else NativeGpu.resetCancel(lease.handle)
        if (cancelled.get()) throw java.util.concurrent.CancellationException("已取消处理")
        val tile=JSONObject(File(models,"manifest.json").readText()).getInt("tileSize")
        val result=PatchPipeline(context,models,progress,::infer,timing,{ cancelled.get() },tile).restore(input,scale,seed,overlap).toMutableMap()
        result["pipelineMs"]=result["elapsedMs"]!!
        result["elapsedMs"]=(System.nanoTime()-started)/1_000_000
        result["modelVerificationMs"]=verifiedMs
        result["engineInitializationMs"]=lease.initializationMs
        result["residentEngineReused"]=lease.reused
        result["engineStats"]=stats()
        result["nativePatchMs"]=nativePatchTimes
        result["nativePatchProfiles"]=nativePatchProfiles
        result["backend"]=if (backend=="npu") "Qualcomm HTP / NPU · FP16" else "ncnn / Vulkan + CPU · FP32 计算"
        result["backendId"]=backend
        return result
    }
}
