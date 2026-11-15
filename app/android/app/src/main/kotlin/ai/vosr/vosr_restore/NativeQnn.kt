package ai.vosr.vosr_restore

import android.content.Context
import java.io.File

internal object NativeQnn {
    init { System.loadLibrary("vosr_jni") }
    @JvmStatic external fun create(models: String,nativeDirectory: String,detailed: Boolean,firstProfileStage: Int): Long
    @JvmStatic external fun run(handle: Long,rgb: FloatArray,dino: FloatArray,noise: FloatArray): FloatArray
    @JvmStatic external fun stats(handle: Long): String
    @JvmStatic external fun resetCancel(handle: Long)
    @JvmStatic external fun profile(handle: Long,enabled: Boolean)
    @JvmStatic external fun cancel(handle: Long)
    @JvmStatic external fun release(handle: Long)
}

internal object ResidentEngine {
    data class Lease(val handle: Long,val reused: Boolean,val initializationMs: Long)
    private var handle=0L
    private var key=""
    private var backend="npu"
    @Synchronized fun acquire(context: Context,models: File,selected: String,detailed: Boolean=false,firstProfileStage: Int=0): Lease {
        val requested=selected+":"+models.canonicalPath+":"+File(models,"manifest.json").lastModified()+":"+(selected=="npu" && detailed)+":"+firstProfileStage
        if (handle!=0L && requested==key) return Lease(handle,true,0)
        close()
        val start=System.nanoTime()
        val created=if (selected=="npu") NativeQnn.create(models.absolutePath,context.applicationInfo.nativeLibraryDir,detailed,firstProfileStage)
                    else NativeGpu.create(models.absolutePath,org.json.JSONObject(File(models,"manifest.json").readText()).getInt("tileSize"))
        check(created!=0L) { "NPU 常驻引擎初始化失败" }
        handle=created; key=requested; backend=selected
        return Lease(created,false,(System.nanoTime()-start)/1_000_000)
    }
    @Synchronized fun close() {
        val old=handle; handle=0; key=""
        if (old!=0L) {
            if (backend=="npu") NativeQnn.release(old) else NativeGpu.release(old)
        }
    }
}
