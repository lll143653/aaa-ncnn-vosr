// File/tensor-only Android Vulkan diagnostics. No screen access.
#include "net.h"
#include "gpu.h"
#include "layer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <chrono>

int main(int argc,char** argv) {
    if(argc<4) { fprintf(stderr,"probe_gpu MODEL_STEM INPUT_RAW OUTPUT_RAW [flags] [blob]\n"); return 2; }
    int flags=argc>4?atoi(argv[4]):0;
    const char* blob=argc>5?argv[5]:"out0";
    ncnn::create_gpu_instance();
    if(ncnn::get_gpu_count()==0) return 3;
    ncnn::Net net;
    net.opt.num_threads=4;
    net.opt.use_vulkan_compute=(flags&128)==0;
    net.opt.use_fp16_packed=false; net.opt.use_fp16_storage=false; net.opt.use_fp16_arithmetic=false;
    net.opt.use_cooperative_matrix=false;
    if(flags&1) net.opt.use_packing_layout=false;
    if(flags&2) net.opt.use_winograd_convolution=false;
    if(flags&4) net.opt.use_sgemm_convolution=false;
    if(flags&8) net.opt.use_subgroup_ops=false;
    if(flags&32) net.opt.use_shader_local_memory=false;
    if(flags&64) net.opt.lightmode=false;
    if(net.load_param((std::string(argv[1])+".ncnn.param").c_str())!=0) return 4;
    if(flags&16) for (auto* layer:net.layers()) if(layer->type=="GroupNorm") {
        layer->support_vulkan=false; layer->support_packing=false;
        layer->support_fp16_storage=false; layer->support_bf16_storage=false;
    }
    if(net.load_model((std::string(argv[1])+".ncnn.bin").c_str())!=0) return 5;
    ncnn::Mat input(512,512,3);
    FILE* f=fopen(argv[2],"rb"); if(!f) return 6;
    for(int c=0;c<3;c++) if(fread(input.channel(c),4,512*512,f)!=512*512) return 7;
    fclose(f);
    auto start=std::chrono::steady_clock::now();
    auto ex=net.create_extractor(); ex.input("in0",input);
    ncnn::Mat out; if(ex.extract(blob,out)!=0) return 8;
    size_t bad=0,total=0; float low=INFINITY,high=-INFINITY;
    f=fopen(argv[3],"wb"); if(!f) return 9;
    for(int c=0;c<out.c;c++) {
        const float* v=out.channel(c);
        for(int i=0;i<out.w*out.h*out.d;i++) { total++; if(!std::isfinite(v[i])) bad++; else {low=std::min(low,v[i]);high=std::max(high,v[i]);} }
        fwrite(v,4,out.w*out.h*out.d,f);
    }
    fclose(f);
    auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    printf("{\"blob\":\"%s\",\"flags\":%d,\"w\":%d,\"h\":%d,\"d\":%d,\"c\":%d,\"total\":%zu,\"nonFinite\":%zu,\"min\":%g,\"max\":%g,\"ms\":%g}\n",blob,flags,out.w,out.h,out.d,out.c,total,bad,low,high,elapsed);
    fflush(stdout); fflush(stderr);
    // Standalone diagnostic only; app JNI does not hard-exit the process.
    std::_Exit(bad?10:0);
}
