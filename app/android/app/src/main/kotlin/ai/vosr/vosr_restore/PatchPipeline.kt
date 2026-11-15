package ai.vosr.vosr_restore

import android.content.Context
import android.os.StatFs
import android.os.Trace
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.CancellationException
import kotlin.math.*

class PatchPipeline(
    private val context: Context,
    private val models: File,
    private val progress: (Double,String)->Unit,
    private val infer: (FloatArray,FloatArray,FloatArray)->FloatArray,
    private val timings: MutableList<Map<String,Any>>,
    private val cancelled: ()->Boolean,
    private val tile: Int=512
) {
    private val plane=tile*tile
    private val costs=linkedMapOf<String,Double>()
    private val patchCosts=mutableListOf<Map<String,Any>>()
    private inline fun <T> measured(name: String,operation: ()->T): T {
        Trace.beginSection("vosr.$name"); val start=System.nanoTime()
        try { return operation() }
        finally {
            costs[name]=(costs[name] ?: 0.0)+(System.nanoTime()-start)/1_000_000.0
            Trace.endSection()
        }
    }
    private fun checkCancelled() { if (cancelled()) throw CancellationException("已取消处理") }
    private data class TileFile(val file: File,val left: Int,val top: Int) {
        val reader=RandomAccessFile(file,"r")
        fun dispose() { reader.close(); file.delete() }
    }
    private fun write(file: File, values: FloatArray): File {
        val buffer=ByteBuffer.allocate(values.size*4).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(values); file.writeBytes(buffer.array()); return file
    }
    private fun dinoInput(unit: FloatArray): FloatArray {
        val size=448; val result=FloatArray(3*size*size)
        val mean=floatArrayOf(.485f,.456f,.406f); val std=floatArrayOf(.229f,.224f,.225f)
        for (y in 0 until size) for (x in 0 until size) {
            val sy=(y+.5)*tile/size-.5; val sx=(x+.5)*tile/size-.5
            val iy=floor(sy).toInt(); val ix=floor(sx).toInt()
            for (c in 0..2) {
                var sum=0.0
                for (dy in -1..2) for (dx in -1..2) {
                    val index=c*plane+(iy+dy).coerceIn(0,tile-1)*tile+(ix+dx).coerceIn(0,tile-1)
                    sum+=unit[index]*NativeImage.cubic(sy-iy-dy,-.75)*NativeImage.cubic(sx-ix-dx,-.75)
                }
                result[c*size*size+y*size+x]=(sum.toFloat().coerceIn(0f,1f)-mean[c])/std[c]
            }
        }
        return result
    }
    private fun mix(value: Long): Long {
        var z=value-7046029254386353131L
        z=(z xor (z ushr 30))*-4658895280553007687L
        z=(z xor (z ushr 27))*-7723592293110705685L
        return z xor (z ushr 31)
    }
    private fun noise(seed: Long,left: Int,top: Int): FloatArray {
        val side=tile/8
        val result=FloatArray(4*side*side)
        for (c in 0..3) for (y in 0 until side) for (x in 0 until side) {
            val key=seed xor (c.toLong()*-7046029254386353131L) xor ((left/8+x).toLong() shl 32) xor (top/8+y).toLong()
            val u1=((mix(key) ushr 11).toDouble()+.5)/9007199254740992.0
            val u2=((mix(key xor -4658895280553007687L) ushr 11).toDouble()+.5)/9007199254740992.0
            result[c*side*side+y*side+x]=(sqrt(-2*ln(u1))*cos(2*PI*u2)).toFloat()
        }
        return result
    }
    private val gaussian=FloatArray(tile) { x -> exp(-.5*(x-(tile-1)*.5).pow(2)/(tile*.25).pow(2)).toFloat() }

    /** Emits only final rows, after every patch contributing to them is available. */
    private fun emitRows(writer: StreamingPng, tiles: List<TileFile>, from: Int,to: Int,width: Int) {
        val sums=FloatArray(width*3); val weights=FloatArray(width)
        val rgb=ByteArray(width*3); val bytes=ByteArray(tile*4)
        val values=FloatArray(tile)
        for (y in from until to) {
            checkCancelled(); sums.fill(0f); weights.fill(0f)
            for (part in tiles) {
                val localY=y-part.top
                if (localY !in 0 until tile) continue
                val valid=minOf(tile,width-part.left)
                for (x in 0 until valid) weights[part.left+x]+=gaussian[x]*gaussian[localY]
                for (c in 0..2) {
                    part.reader.seek((c.toLong()*plane+localY*tile)*4)
                    part.reader.readFully(bytes)
                    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
                    for (x in 0 until valid) {
                        require(values[x].isFinite()) { "推理输出包含无效数值" }
                        sums[c*width+part.left+x]+=(values[x]*.5f+.5f)*gaussian[x]*gaussian[localY]
                    }
                }
            }
            for (x in 0 until width) {
                check(weights[x]>0) { "patch 拼接存在未覆盖像素" }
                for (c in 0..2) rgb[x*3+c]=(sums[c*width+x]/weights[x]*255).roundToInt().coerceIn(0,255).toByte()
            }
            measured("pngEncodeWrite") { writer.writeRow(rgb) }
        }
    }

    fun restore(input: File,scale: Int,seed: Long,overlap: Int=64): Map<String,Any> {
        require(scale==2 || scale==4)
        require(overlap in 1 until tile)
        val stride=tile-overlap
        val started=System.nanoTime()
        val root=File(context.cacheDir,"restore-${UUID.randomUUID()}"); root.mkdirs()
        val output=File(context.cacheDir,"vosr_${System.currentTimeMillis()}.png")
        val partial=File(output.absolutePath+".partial")
        val live=mutableListOf<TileFile>()
        try {
            measured("regionDecoderOpen") { NativeImage(input) }.use { source ->
                val widthLong=source.width.toLong()*scale; val heightLong=source.height.toLong()*scale
                require(widthLong<=Int.MAX_VALUE && heightLong<=Int.MAX_VALUE) { "输出尺寸超出 PNG 支持范围" }
                val width=widthLong.toInt(); val height=heightLong.toInt()
                val rows=if (height<=tile) 1 else ((heightLong-tile+stride-1)/stride+1).toInt()
                val cols=if (width<=tile) 1 else ((widthLong-tile+stride-1)/stride+1).toInt()
                val count=rows.toLong()*cols
                val diskNeeded=Math.multiplyExact(Math.multiplyExact(widthLong,heightLong),3L)+cols.toLong()*plane*24+64*1024*1024
                require(StatFs(context.cacheDir.absolutePath).availableBytes>diskNeeded) { "缓存空间不足，无法保存原始分辨率的输出" }
                progress(0.0,"原图 ${source.width}×${source.height} → ${width}×${height}，共 $count 个 patch")
                var emitted=0
                val writer=measured("pngOpen") { StreamingPng(partial,width,height) }
                try {
                    for (row in 0 until rows) {
                        val current=mutableListOf<TileFile>()
                        for (col in 0 until cols) {
                            checkCancelled()
                            val left=col*stride; val top=row*stride
                            val before=costs.toMap()
                            val unit=measured("regionDecodeAndBicubic") { source.patch(left,top,tile,scale) }
                            val directory=File(root,"tile_${row}_$col"); directory.mkdirs()
                            progress((row.toLong()*cols+col).toDouble()/count,
                                "patch ${row.toLong()*cols+col+1}/$count · 端侧复原")
                            val rgb=measured("rgbNormalize") { FloatArray(unit.size){unit[it]*2-1} }
                            val dino=measured("dinoResizeNormalize") { dinoInput(unit) }
                            val latentNoise=measured("noiseGenerate") { noise(seed,left,top) }
                            val pixels=measured("jniInference") { infer(rgb,dino,latentNoise) }
                            require(pixels.size==plane*3) { "patch 输出尺寸错误" }
                            val cached=File(root,"row_${row}_col_$col.f32")
                            measured("patchCacheWrite") { write(cached,pixels) }
                            val part=TileFile(cached,left,top); current.add(part); live.add(part)
                            directory.deleteRecursively()
                            patchCosts.add(mapOf("row" to row,"col" to col,"phasesMs" to costs.mapValues { (k,v)->v-(before[k] ?: 0.0) }))
                        }
                        val end=if (row==rows-1) height else (row+1)*stride
                        measured("stitchAndPng") { emitRows(writer,live,emitted,end,width) }; emitted=end
                        val old=live.filter { it !in current }
                        for (part in old) { part.dispose(); live.remove(part) }
                    }
                } finally { measured("pngFinalize") { writer.close() } }
                measured("outputFinalize") { check(partial.renameTo(output)) { "无法完成结果文件" } }
                val elapsed=(System.nanoTime()-started)/1_000_000
                File(root,"timing.json").writeText(JSONObject(mapOf("stages" to timings,"elapsedMs" to elapsed,"pipelineBreakdownMs" to costs,"patches" to patchCosts)).toString(2))
                progress(1.0,"复原完成")
                return mapOf("path" to output.absolutePath,"backend" to "端侧推理",
                    "elapsedMs" to elapsed,"inputWidth" to source.width,"inputHeight" to source.height,
                    "width" to width,"height" to height,"scale" to scale,"tiles" to count,
                    "patchSize" to tile,"overlap" to tile-stride,"inputDownscaled" to false,
                    "streamingOutput" to true,"exifOrientation" to source.orientation,
                    "pipelineBreakdownMs" to costs.toMap(),"patchTimings" to patchCosts,
                    "timingFile" to File(root,"timing.json").absolutePath)
            }
        } finally {
            for (part in live) part.dispose()
            partial.delete()
        }
    }
}
