package ai.vosr.vosr_restore

import java.io.*
import java.util.zip.CRC32
import java.util.zip.Deflater
import java.util.zip.DeflaterOutputStream

/** RGB PNG written scanline-by-scanline, without a full output Bitmap. */
class StreamingPng(file: File, private val width: Int, private val height: Int) : Closeable {
    private val output=DataOutputStream(BufferedOutputStream(file.outputStream(),65536))
    private var rows=0
    private var closed=false
    private val deflater=Deflater(3)
    private val chunks=object : OutputStream() {
        val buffer=ByteArray(65536)
        var count=0
        fun emit() { if (count>0) { chunk("IDAT",buffer,count); count=0 } }
        override fun write(value: Int) { buffer[count++]=value.toByte(); if (count==buffer.size) emit() }
        override fun write(bytes: ByteArray, offset: Int, length: Int) {
            var position=offset; var remaining=length
            while (remaining>0) {
                val n=minOf(remaining,buffer.size-count)
                bytes.copyInto(buffer,count,position,position+n)
                count+=n; position+=n; remaining-=n
                if (count==buffer.size) emit()
            }
        }
    }
    private val compressed=DeflaterOutputStream(chunks,deflater,65536)
    init {
        require(width>0 && height>0)
        output.write(byteArrayOf(0x89.toByte(),80,78,71,13,10,26,10))
        val header=ByteArrayOutputStream()
        DataOutputStream(header).use { it.writeInt(width); it.writeInt(height); it.write(byteArrayOf(8,2,0,0,0)) }
        chunk("IHDR",header.toByteArray())
    }
    private fun chunk(name: String, bytes: ByteArray, count: Int=bytes.size) {
        val type=name.toByteArray(Charsets.US_ASCII)
        val crc=CRC32(); crc.update(type); crc.update(bytes,0,count)
        output.writeInt(count); output.write(type); output.write(bytes,0,count); output.writeInt(crc.value.toInt())
    }
    fun writeRow(rgb: ByteArray) {
        check(!closed && rows<height && rgb.size==width*3)
        compressed.write(0) // PNG filter None
        compressed.write(rgb)
        rows++
    }
    override fun close() {
        if (closed) return
        closed=true
        try {
            if (rows==height) {
                compressed.finish(); chunks.emit(); chunk("IEND",ByteArray(0))
            }
        } finally { deflater.end(); output.close() }
    }
}
