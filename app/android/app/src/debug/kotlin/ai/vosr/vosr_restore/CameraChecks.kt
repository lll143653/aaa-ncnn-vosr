package ai.vosr.vosr_restore

import android.app.Activity
import android.app.Instrumentation
import android.graphics.Bitmap
import android.graphics.Color
import androidx.exifinterface.media.ExifInterface
import android.os.Bundle
import java.io.File
import kotlin.math.abs

/** Runs on the device without screen access or model inference. */
class CameraChecks : Instrumentation() {
    override fun onCreate(arguments: Bundle?) { super.onCreate(arguments); start() }
    override fun onStart() {
        val checks=mutableListOf<String>()
        val folder=File(targetContext.cacheDir,"camera-checks").apply { mkdirs() }
        try {
            val v73=NpuTarget.forSoc("SM8550")!!; val v81=NpuTarget.forSoc("SM8850")!!
            val old=org.json.JSONObject().put("backend","QNN_HTP_V73_FP16")
            val current=org.json.JSONObject().put("backend","QNN_HTP_V81_FP16").put("targetSoc","SM8850")
            check(v73.accepts(old) && !v81.accepts(old) && v81.accepts(current) && !v73.accepts(current))
            check(v81.extendedUdma && v81.streamDit && v73.streamDit && !v73.extendedUdma)
            check(v81.folder(true)=="models-sm8850-1024" && NpuTarget.forSoc("unknown")==null)
            checks.add("v73/v81 target matching, separate model directories and far-mapping policy")
            check(CameraPhoto.workingSize(4000,3000)==Pair(832,624))
            check(CameraPhoto.workingSize(3000,4000)==Pair(624,832))
            check(CameraPhoto.workingSize(100,50)==Pair(832,416))
            checks.add("832 up/down scaling and aspect ratio")
            check(PhotoPlan.count(832)*PhotoPlan.count(624)==4)
            check(PhotoPlan.count(624)*PhotoPlan.count(832)==4)
            check(PhotoPlan.count(832)*PhotoPlan.count(416)==2)
            check(PhotoPlan.count(512)==1 && PhotoPlan.count(513)==2)
            check(PhotoPlan.INPUT_STRIDE==416 && PhotoPlan.OUTPUT_TILE==1024 && PhotoPlan.OUTPUT_OVERLAP==192)
            checks.add("512 LR tiles / 96 LR overlap / true 1024 output: 832x624 uses four passes")
            val file=File(folder,"exif.jpg")
            val base=Bitmap.createBitmap(19,13,Bitmap.Config.ARGB_8888)
            for(y in 0 until 13) for(x in 0 until 19) base.setPixel(x,y,Color.rgb(x*11,y*17,(x+y)*7))
            file.outputStream().use { base.compress(Bitmap.CompressFormat.JPEG,100,it) }; base.recycle()
            fun exif(zoom: String?) { ExifInterface(file.absolutePath).apply { setAttribute(ExifInterface.TAG_DIGITAL_ZOOM_RATIO,zoom); saveAttributes() } }
            for(value in listOf(null,"0/1","9/1","9999/1000")) {
                exif(value); check(runCatching { CameraPhoto.zoom(file) }.isFailure) { "Accepted EXIF $value" }
            }
            exif("10/1"); check(CameraPhoto.zoom(file)==10.0)
            exif("12/1"); check(CameraPhoto.zoom(file)==12.0)
            checks.add("strict EXIF: missing/zero/<10 rejected; 10 and 12 accepted")
            for(o in 1..8) {
                ExifInterface(file.absolutePath).apply { setAttribute(ExifInterface.TAG_ORIENTATION,o.toString()); saveAttributes() }
                NativeImage(file).use { image ->
                    val reference=image.patch(0,0,24,1)
                    val actual=image.render(0,0,24,24,image.width,image.height)
                    try {
                        for(y in 0 until 24) for(x in 0 until 24) for(c in 0..2) {
                            val value=(actual.getPixel(x,y) shr (16-c*8)) and 255
                            check(abs(value-reference[c*24*24+y*24+x]*255)<1.1) { "EXIF $o mismatch at $x,$y" }
                        }
                    } finally { actual.recycle() }
                }
            }
            checks.add("all 8 EXIF orientations match original region decoder")
            ExifInterface(file.absolutePath).apply { setAttribute(ExifInterface.TAG_ORIENTATION,"1"); saveAttributes() }
            NativeImage(file).use { image ->
                val whole=image.render(0,0,77,53,77,53)
                try {
                    for(top in 0 until 53 step 16) for(left in 0 until 77 step 16) {
                        val w=minOf(16,77-left); val h=minOf(16,53-top)
                        val part=image.render(left,top,w,h,77,53)
                        try { for(y in 0 until h) for(x in 0 until w) for(c in 0..2)
                            check(abs(((whole.getPixel(left+x,top+y) shr (c*8)) and 255)-((part.getPixel(x,y) shr (c*8)) and 255))<=1)
                        } finally { part.recycle() }
                    }
                } finally { whole.recycle() }
            }
            checks.add("separable cubic resampling has consistent tile coordinates")
            fun image(left: Int,top: Int,w: Int,h: Int,sr: Boolean): Bitmap {
                val b=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
                val values=IntArray(w*h)
                for(y in 0 until h) for(x in 0 until w) {
                    val xx=(left+x).coerceIn(0,1023); val yy=(top+y).coerceIn(0,767)
                    val v=(xx*3+yy*7+if(sr) ((xx/3+yy/5)%2)*40 else 0)%256
                    values[y*w+x]=Color.rgb(v,(xx/4+if(sr)20 else 0)%256,yy/4)
                }
                b.setPixels(values,0,w,0,0,w,h); return b
            }
            val a=image(-128,-128,1280,1024,false); val b=image(-128,-128,1280,1024,true)
            val full=PhotoNative.fuse(a,b,128,1024,768)
            a.recycle(); b.recycle()
            var maxDifference=0
            for(top in 0 until 768 step 256) for(left in 0 until 1024 step 512) {
                val lr=image(left-128,top-128,768,512,false); val sr=image(left-128,top-128,768,512,true)
                val tile=PhotoNative.fuse(lr,sr,128,512,256)
                lr.recycle(); sr.recycle()
                for(y in 0 until 256) for(x in 0 until 512) for(c in 0..2) {
                    val diff=abs((tile[(y*512+x)*3+c].toInt() and 255)-(full[((top+y)*1024+left+x)*3+c].toInt() and 255))
                    maxDifference=maxOf(maxDifference,diff)
                }
            }
            check(maxDifference<=1) { "Pyramid seams: max difference $maxDifference" }
            checks.add("tiled/full pyramid fusion agrees within $maxDifference LSB")
            val same=image(0,0,96,80,false)
            val identical=PhotoNative.fuse(same,same,0,96,80)
            for(y in 0 until 80) for(x in 0 until 96) for(c in 0..2)
                check((identical[(y*96+x)*3+c].toInt() and 255)==((same.getPixel(x,y) shr (16-c*8)) and 255))
            same.recycle(); checks.add("identical inputs preserve every pixel")
            val solidA=Bitmap.createBitmap(96,80,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.rgb(30,90,180)) }
            val solidB=Bitmap.createBitmap(96,80,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.rgb(100,50,120)) }
            val solid=PhotoNative.fuse(solidA,solidB,0,96,80)
            for(i in solid.indices) check((solid[i].toInt() and 255)==intArrayOf(30,90,180)[i%3])
            solidA.recycle(); solidB.recycle(); checks.add("LR low-frequency color preserved")
            val flat=Bitmap.createBitmap(96,80,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.rgb(100,100,100)) }
            val textured=Bitmap.createBitmap(96,80,Bitmap.Config.ARGB_8888)
            for(y in 0 until 80) for(x in 0 until 96) { val v=if((x+y)%2==0)80 else 120; textured.setPixel(x,y,Color.rgb(v,v,v)) }
            val enhanced=PhotoNative.fuse(flat,textured,0,96,80)
            check(abs((enhanced[(40*96+40)*3].toInt() and 255)-80)<=1)
            check(abs((enhanced[(40*96+41)*3].toInt() and 255)-120)<=1)
            flat.recycle(); textured.recycle(); checks.add("SR high-frequency detail retained")
            finish(Activity.RESULT_OK,Bundle().apply { putString("passed",checks.joinToString("; ")) })
        } catch(error: Throwable) {
            finish(Activity.RESULT_CANCELED,Bundle().apply {
                putString("error",error.stackTraceToString()); putString("completed",checks.joinToString("; "))
                putString("rawZoom",runCatching { ExifInterface(File(folder,"exif.jpg").absolutePath).getAttribute(ExifInterface.TAG_DIGITAL_ZOOM_RATIO) }.getOrNull())
            })
        } finally { folder.deleteRecursively() }
    }
}
