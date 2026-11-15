package ai.vosr.vosr_restore

import android.content.Context
import android.graphics.Bitmap
import android.os.StatFs
import java.io.File
import java.util.UUID
import java.util.concurrent.CancellationException

/** Bounded working size, fixed 2x, then original-size pyramid fusion. Camera inputs also require verified zoom. */
internal class CameraPipeline(private val context: Context,
    private val progress: (Double,String)->Unit,
    private val cancelled: ()->Boolean,
    private val restore: (File)->Map<String,Any>,
    private val requireCameraZoom: Boolean=true) {
    private fun checkCancelled() { if(cancelled()) throw CancellationException("已取消处理") }
    fun run(input: File): Map<String,Any> {
        val started=System.nanoTime()
        val zoom=if(requireCameraZoom) CameraPhoto.zoom(input) else null
        val working=File(context.cacheDir,"camera_work_${UUID.randomUUID()}.png")
        val output=File(context.cacheDir,"vosr_camera_${UUID.randomUUID()}.png")
        val partial=File(output.absolutePath+".partial")
        var intermediate: File?=null
        try {
            NativeImage(input).use { lr ->
                val required=Math.addExact(Math.multiplyExact(Math.multiplyExact(lr.width.toLong(),lr.height.toLong()),3L),64L*1024*1024)
                require(StatFs(context.cacheDir.absolutePath).availableBytes>required) { "缓存空间不足，无法保存原尺寸融合图像" }
                val (w,h)=CameraPhoto.workingSize(lr.width,lr.height)
                progress(.01,"准备图像 · ${lr.width}×${lr.height} → ${w}×${h}")
                val bitmap=lr.render(0,0,w,h,w,h,true)
                try { working.outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.PNG,100,it)) } }
                finally { bitmap.recycle() }
                checkCancelled()
                val preparedMs=(System.nanoTime()-started)/1_000_000
                val result=restore(working).toMutableMap()
                check(result["patchSize"]==PhotoPlan.OUTPUT_TILE && result["overlap"]==PhotoPlan.OUTPUT_OVERLAP) {
                    "当前模型不支持 512 输入块的真实 2× 输出，请导入 1024 输出模型"
                }
                check((result["tiles"] as Number).toInt()==PhotoPlan.count(w)*PhotoPlan.count(h))
                intermediate=File(result["path"] as String)
                val fusionStart=System.nanoTime()
                NativeImage(intermediate!!).use { sr ->
                    check(sr.width==w*2 && sr.height==h*2)
                    StreamingPng(partial,lr.width,lr.height).use { png ->
                        // 128px halo covers five pyramid reductions and expansions.
                        // All origins align to 32px so adjacent tiles share the same pyramid grid.
                        val bandHeight=minOf(256,maxOf(32,(8L*1024*1024/(lr.width.toLong()*3)/32*32).toInt()))
                        val tileWidth=512; val halo=128
                        var top=0
                        while(top<lr.height) {
                            checkCancelled()
                            val rows=minOf(bandHeight,lr.height-top)
                            val band=ByteArray(Math.multiplyExact(Math.multiplyExact(lr.width,rows),3))
                            var left=0
                            while(left<lr.width) {
                                checkCancelled()
                                val cols=minOf(tileWidth,lr.width-left)
                                val a=lr.render(left-halo,top-halo,tileWidth+2*halo,bandHeight+2*halo,lr.width,lr.height)
                                try {
                                    val b=sr.render(left-halo,top-halo,tileWidth+2*halo,bandHeight+2*halo,lr.width,lr.height)
                                    try {
                                        val fused=PhotoNative.fuse(a,b,halo,cols,rows)
                                        for(y in 0 until rows) System.arraycopy(fused,y*cols*3,band,(y*lr.width+left)*3,cols*3)
                                    } finally { b.recycle() }
                                } finally { a.recycle() }
                                left+=tileWidth
                            }
                            val row=ByteArray(lr.width*3)
                            for(y in 0 until rows) { checkCancelled(); System.arraycopy(band,y*row.size,row,0,row.size); png.writeRow(row) }
                            top+=rows
                            progress(.85+.14*top/lr.height,"高斯金字塔融合 · $top/${lr.height} 行")
                        }
                    }
                }
                checkCancelled()
                check(partial.renameTo(output)) { "无法保存拍照复原结果" }
                result["superResolutionWidth"]=w*2; result["superResolutionHeight"]=h*2
                result["workingWidth"]=w; result["workingHeight"]=h
                result["inputWidth"]=lr.width; result["inputHeight"]=lr.height
                result["width"]=lr.width; result["height"]=lr.height
                result["inputDownscaled"]=maxOf(lr.width,lr.height)>832
                result["exifOrientation"]=lr.orientation
                result["mode"]=if(requireCameraZoom) "camera" else "photo"
                result["scale"]=PhotoPlan.SCALE; result["overlap"]=PhotoPlan.INPUT_OVERLAP
                result["inputPatchSize"]=PhotoPlan.INPUT_TILE; result["outputPatchSize"]=PhotoPlan.OUTPUT_TILE
                result["inputStride"]=PhotoPlan.INPUT_STRIDE; result["outputOverlap"]=PhotoPlan.OUTPUT_OVERLAP
                result["overlapCoordinateSpace"]="resized-input"
                if(zoom!=null) { result["zoomRatio"]=zoom; result["zoomVerification"]="EXIF DigitalZoomRatio >= 10" }
                result["fusion"]="gaussian-pyramid-sr-detail-lr-base"; result["pyramidLevels"]=5
                result["cameraPreparationMs"]=preparedMs
                val fusionMs=(System.nanoTime()-fusionStart)/1_000_000
                result["superResolutionElapsedMs"]=result["elapsedMs"]!!
                result["superResolutionPipelineMs"]=result["pipelineMs"]!!
                result["pipelineMs"]=(result["pipelineMs"] as Number).toLong()+preparedMs+fusionMs
                result["fusionMs"]=fusionMs
                result["elapsedMs"]=(System.nanoTime()-started)/1_000_000
                result["path"]=output.absolutePath
                progress(1.0,"复原完成")
                return result
            }
        } catch(error: Throwable) { output.delete(); throw error }
        finally { working.delete(); intermediate?.delete(); partial.delete() }
    }
}
