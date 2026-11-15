package ai.vosr.vosr_restore

import android.os.Build
import org.json.JSONObject

internal data class NpuTarget(val soc: String,val arch: String,val extendedUdma: Boolean,val streamDit: Boolean) {
    val backend get()="QNN_HTP_${arch.uppercase()}_FP16"
    fun folder(large: Boolean)=(if(soc=="SM8550")"models" else "models-${soc.lowercase()}")+(if(large)"-1024" else "")
    fun accepts(data: JSONObject)=data.optString("backend")==backend && data.optString("targetSoc",if(arch=="v73")"SM8550" else "")==soc
    companion object {
        fun forSoc(soc: String): NpuTarget? = when {
            soc.uppercase().contains("SM8550") -> NpuTarget("SM8550","v73",false,true)
            soc.uppercase().contains("SM8850") -> NpuTarget("SM8850","v81",true,true)
            else -> null
        }
        fun current()=if(Build.VERSION.SDK_INT>=31)forSoc(Build.SOC_MODEL) else null
    }
}
