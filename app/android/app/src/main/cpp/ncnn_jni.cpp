#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "net.h"
#include "gpu.h"
#include "layer.h"
#include "profile_trace.h"

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
void require(bool value,const std::string& message) { if (!value) throw std::runtime_error(message); }
std::once_flag gpuOnce;
int gpuCount=0;
void initializeGpu() {
    std::call_once(gpuOnce,[] { if (ncnn::create_gpu_instance()==0) gpuCount=ncnn::get_gpu_count(); });
}
std::string gpuName() { initializeGpu(); return gpuCount>0?ncnn::get_gpu_info(ncnn::get_default_gpu_index()).device_name():""; }
const std::vector<std::string> names={"encoder","dino","dit_prepare","dit_blocks_00","dit_blocks_04","dit_blocks_08","dit_blocks_12","dit_blocks_16","dit_blocks_20","dit_blocks_24","dit_finish","decoder"};
struct CpuMetric { std::string stage,layer,type; double elapsedMs=0; uint64_t calls=0; };
class TimedCpuLayer final:public ncnn::Layer {
    std::unique_ptr<ncnn::Layer> original;
    std::shared_ptr<CpuMetric> metric;
    const bool* enabled;
    template<class Operation> int invoke(Operation operation) const {
        if (!*enabled) return operation();
        TraceSpan trace("ncnn.cpu."+metric->type); auto start=Clock::now();
        int result=operation(); metric->elapsedMs+=ms(start); metric->calls++; return result;
    }
public:
    TimedCpuLayer(ncnn::Layer* layer,std::shared_ptr<CpuMetric> item,const bool* flag):original(layer),metric(std::move(item)),enabled(flag) {
        ncnn::Layer::operator=(*layer);
    }
    int destroy_pipeline(const ncnn::Option& opt) override { return original->destroy_pipeline(opt); }
    int forward(const ncnn::Mat& a,ncnn::Mat& b,const ncnn::Option& opt) const override { return invoke([&]{ return original->forward(a,b,opt); }); }
    int forward(const std::vector<ncnn::Mat>& a,std::vector<ncnn::Mat>& b,const ncnn::Option& opt) const override { return invoke([&]{ return original->forward(a,b,opt); }); }
    int forward_inplace(ncnn::Mat& a,const ncnn::Option& opt) const override { return invoke([&]{ return original->forward_inplace(a,opt); }); }
    int forward_inplace(std::vector<ncnn::Mat>& a,const ncnn::Option& opt) const override { return invoke([&]{ return original->forward_inplace(a,opt); }); }
};
class Engine {
    int tile=512;
    std::string directory;
    bool streamGraphs=false;
    bool hostVaeFallback=false;
    uint64_t storageBufferLimit=0;
    uint64_t netCreates=0;
    size_t maxResidentNets=0;
    double netLoadMs=0;
    std::vector<std::unique_ptr<ncnn::Net>> nets;
    std::mutex mutex;
    std::vector<double> times=std::vector<double>(12);
    std::vector<double> validationTimes=std::vector<double>(12);
    double loadMs=0,runMs=0;
    uint64_t count=0;
    int hostReductionLayers=0;
    bool profiling=false;
    std::vector<std::shared_ptr<CpuMetric>> cpuMetrics;
    std::vector<double> initializationTimes;
    double inputCopyMs=0,outputCopyMs=0;
    ncnn::Mat input(JNIEnv* env,jfloatArray array,int w,int h,int c) {
        require(array && env->GetArrayLength(array)==w*h*c,"Invalid GPU patch input");
        ncnn::Mat mat(w,h,c);
        require(!mat.empty(),"Cannot allocate aligned ncnn input");
        for (int ch=0;ch<c;ch++) env->GetFloatArrayRegion(array,ch*w*h,w*h,mat.channel(ch));
        require(!env->ExceptionCheck(),"Cannot copy GPU input");
        return mat;
    }
    std::vector<ncnn::Mat> execute(int stage,const std::vector<ncnn::Mat>& inputs,int outputs=1) {
        if (cancelled.load()) throw std::runtime_error("已取消处理");
        auto start=Clock::now();
        TraceSpan trace("vosr.gpu."+names[stage]);
        __android_log_print(ANDROID_LOG_INFO,"VOSR-GPU","Execute %s",names[stage].c_str());
        if(!nets[stage]) loadNet(stage);
        struct StageNet {
            std::unique_ptr<ncnn::Net>* net; bool release;
            ~StageNet() { if(release) net->reset(); }
        } stageNet{&nets[stage],streamGraphs};
        auto ex=nets[stage]->create_extractor();
        for (size_t i=0;i<inputs.size();i++) {
            require(ex.input(("in"+std::to_string(i)).c_str(),inputs[i])==0,"ncnn input failed: "+names[stage]);
        }
        std::vector<ncnn::Mat> values(outputs);
        for (int i=0;i<outputs;i++) {
            require(ex.extract(("out"+std::to_string(i)).c_str(),values[i])==0 && !values[i].empty(),"ncnn execute failed: "+names[stage]);
            const auto& value=values[i];
            require(value.elempack==1 && value.elemsize==4,"Unexpected GPU intermediate type: "+names[stage]);
            auto validationStart=Clock::now();
            for (int c=0;c<value.c;c++) {
                const float* channel=value.channel(c);
                for (int j=0;j<value.w*value.h*value.d;j++) {
                    if (!std::isfinite(channel[j])) throw std::runtime_error("Non-finite Vulkan output: "+names[stage]+"/out"+std::to_string(i));
                }
            }
            validationTimes[stage]+=ms(validationStart);
        }
        times[stage]=ms(start);
        __android_log_print(ANDROID_LOG_INFO,"VOSR-GPU","Completed %s: %.1f ms",names[stage].c_str(),times[stage]);
        return values;
    }
public:
    std::atomic<bool> cancelled{false};
    void loadNet(size_t index) {
        const auto& name=names[index];
        auto stageStart=Clock::now();
        TraceSpan trace("vosr.load.gpu."+name);
        auto net=std::make_unique<ncnn::Net>();
        net->opt.num_threads=4;
        // Adreno 740 exposes 128 MiB per storage-buffer binding. The
        // 1024 VAE has 512 MiB feature maps; keep its encoder/decoder on
        // CPU rather than submitting invalid Vulkan descriptor ranges.
        const bool largeVae=tile==1024 && (name=="encoder" || name=="decoder");
        const bool hostVae=hostVaeFallback && largeVae;
        net->opt.use_vulkan_compute=!hostVae;
        if(largeVae) net->opt.use_sgemm_convolution=false;
        // Match the original project's norm precision policy.
        net->opt.use_fp16_packed=false;
        net->opt.use_fp16_storage=false;
        net->opt.use_fp16_arithmetic=false;
        // Use portable FP32 shaders for this Android 13 Adreno profile.
        net->opt.use_subgroup_ops=false;
        net->opt.use_cooperative_matrix=false;
        // Adreno 740 / Android 13: the Winograd path failed numerical
        // parity. The direct/SGEMM convolution path passes the same test.
        net->opt.use_winograd_convolution=false;
        net->set_vulkan_device(ncnn::get_default_gpu_index());
        require(net->load_param((directory+"/"+name+".ncnn.param").c_str())==0,"Cannot load GPU graph "+name);
        for (auto* layer:net->layers()) if (layer->type=="Reduction") {
            // This runtime has no valid Vulkan pipeline for the 4D RMS
            // reductions. Its generic CPU implementation is validated.
            layer->support_vulkan=false;
            layer->support_packing=false;
            layer->support_fp16_storage=false;
            layer->support_bf16_storage=false;
            hostReductionLayers++;
        }
        require(net->load_model((directory+"/"+name+".ncnn.bin").c_str())==0,"Cannot load GPU weights "+name);
        for (auto*& layer:net->mutable_layers()) if (!layer->support_vulkan) {
            auto metric=std::make_shared<CpuMetric>(); metric->stage=name; metric->layer=layer->name; metric->type=layer->type;
            cpuMetrics.push_back(metric);
            layer=new TimedCpuLayer(layer,metric,&profiling);
        }
        nets[index]=std::move(net); netCreates++;
        maxResidentNets=std::max(maxResidentNets,static_cast<size_t>(std::count_if(nets.begin(),nets.end(),[](const auto& n){return n!=nullptr;})));
        initializationTimes[index]=ms(stageStart); netLoadMs+=initializationTimes[index];
    }
    void initialize(const std::string& modelDirectory,int outputTile) {
        require(outputTile==512 || outputTile==1024,"Unsupported GPU output tile"); tile=outputTile;
        directory=modelDirectory; streamGraphs=tile==1024;
        initializeGpu(); require(gpuCount>0,"No Vulkan GPU available; CPU fallback is disabled");
        storageBufferLimit=ncnn::get_gpu_info(ncnn::get_default_gpu_index()).physicalDeviceProperties().limits.maxStorageBufferRange;
        hostVaeFallback=tile==1024 && storageBufferLimit<512ULL*1024*1024;
        auto start=Clock::now();
        nets.resize(names.size()); initializationTimes.resize(names.size());
        if(!streamGraphs) for(size_t i=0;i<names.size();i++) loadNet(i);
        loadMs=ms(start);
        __android_log_print(ANDROID_LOG_INFO,"VOSR-GPU","Vulkan ready: %s, stream=%d, %.1f ms",gpuName().c_str(),streamGraphs,loadMs);
    }
    jfloatArray run(JNIEnv* env,jfloatArray rgb,jfloatArray dino,jfloatArray noise) {
        std::lock_guard<std::mutex> lock(mutex); auto start=Clock::now();
        TraceSpan trace("vosr.gpu.patch");
        if(streamGraphs) { cpuMetrics.clear(); hostReductionLayers=0; netLoadMs=0; std::fill(initializationTimes.begin(),initializationTimes.end(),0); }
        for (auto& metric:cpuMetrics) { metric->elapsedMs=0; metric->calls=0; }
        std::fill(validationTimes.begin(),validationTimes.end(),0);
        auto image=input(env,rgb,tile,tile,3);
        auto dinoImage=input(env,dino,448,448,3);
        auto n=input(env,noise,tile/8,tile/8,4);
        inputCopyMs=ms(start);
        auto condition=execute(0,{image})[0];
        auto features=execute(1,{dinoImage})[0];
        auto prepared=execute(2,{condition,n,features},2);
        auto tokens=prepared[0]; auto context=prepared[1];
        for (int stage=3;stage<=9;stage++) tokens=execute(stage,{tokens,context})[0];
        auto latent=execute(10,{tokens,n})[0];
        auto pixels=execute(11,{latent})[0];
        require(pixels.w==tile && pixels.h==tile && pixels.c==3 && pixels.elempack==1 && pixels.elemsize==4,"Invalid GPU output shape/type");
        if (cancelled.load()) throw std::runtime_error("已取消处理");
        auto copyStart=Clock::now();
        auto out=env->NewFloatArray(3*tile*tile); require(out,"Cannot allocate GPU output");
        for (int c=0;c<3;c++) env->SetFloatArrayRegion(out,c*tile*tile,tile*tile,pixels.channel(c));
        require(!env->ExceptionCheck(),"Cannot return GPU output");
        outputCopyMs=ms(copyStart); runMs=ms(start); count++; return out;
    }
    void setProfiling(bool enabled) { std::lock_guard<std::mutex> lock(mutex); profiling=enabled; }
    std::string stats() {
        std::lock_guard<std::mutex> lock(mutex); std::ostringstream out;
        out<<"{\"engine\":\"NCNN_VULKAN_RESIDENT\",\"device\":\""<<gpuName()<<"\",\"contextCreateCount\":"<<netCreates<<",\"graphCount\":"<<nets.size()
           <<",\"modelLoadMs\":"<<loadMs<<",\"patchRunCount\":"<<count<<",\"lastPatchMs\":"<<runMs
           <<",\"hostReductionLayers\":"<<hostReductionLayers
           <<",\"vaeBackend\":\""<<(hostVaeFallback?"cpu-buffer-limit":"vulkan")<<"\""<<",\"maxStorageBufferRange\":"<<storageBufferLimit
           <<",\"streamGraphs\":"<<(streamGraphs?"true":"false")<<",\"maxResidentNetCount\":"<<maxResidentNets<<",\"netLoadMs\":"<<netLoadMs
           <<",\"profilingEnabled\":"<<(profiling?"true":"false")<<",\"inputCopyMs\":"<<inputCopyMs<<",\"outputCopyMs\":"<<outputCopyMs
           <<",\"subprocessCount\":0,\"intermediateTensorFileBytes\":0,\"stageMs\":[";
        for (size_t i=0;i<times.size();i++) { if(i) out<<","; out<<times[i]; }
        out<<"],\"initializationStageMs\":[";
        for(size_t i=0;i<initializationTimes.size();i++) { if(i) out<<","; out<<initializationTimes[i]; }
        out<<"],\"finiteValidationStageMs\":[";
        for(size_t i=0;i<validationTimes.size();i++) { if(i) out<<","; out<<validationTimes[i]; }
        out<<"],\"cpuReductionTimings\":[";
        bool first=true;
        if (profiling) for(size_t i=0;i<cpuMetrics.size();i++) {
            const auto& item=*cpuMetrics[i]; if(item.type!="Reduction") continue;
            if(!first) out<<","; first=false;
            out<<"{\"stage\":"<<jsonString(item.stage)<<",\"layer\":"<<jsonString(item.layer)<<",\"calls\":"<<item.calls<<",\"elapsedMs\":"<<item.elapsedMs<<"}";
        }
        out<<"],\"cpuLayerTimings\":[";
        if(profiling) for(size_t i=0;i<cpuMetrics.size();i++) {
            if(i) out<<","; const auto& item=*cpuMetrics[i];
            out<<"{\"stage\":"<<jsonString(item.stage)<<",\"layer\":"<<jsonString(item.layer)<<",\"type\":"<<jsonString(item.type)<<",\"calls\":"<<item.calls<<",\"elapsedMs\":"<<item.elapsedMs<<"}";
        }
        out<<"]}"; return out.str();
    }
};
std::mutex registryMutex;
std::map<jlong,std::shared_ptr<Engine>> registry;
std::atomic<jlong> nextHandle{1};
std::shared_ptr<Engine> lookup(jlong handle) {
    std::lock_guard<std::mutex> lock(registryMutex); auto it=registry.find(handle);
    require(it!=registry.end(),"GPU engine is closed"); return it->second;
}
std::string string(JNIEnv* env,jstring value) {
    require(value,"Missing GPU model path"); const char* chars=env->GetStringUTFChars(value,nullptr);
    require(chars,"Cannot read model path"); std::string out(chars); env->ReleaseStringUTFChars(value,chars); return out;
}
void exception(JNIEnv* env,const std::exception& error) {
    __android_log_print(ANDROID_LOG_ERROR,"VOSR-GPU","%s",error.what());
    if(!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),error.what());
}
}
extern "C" JNIEXPORT jstring JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_probe(JNIEnv* env,jclass) {
    try { initializeGpu(); std::ostringstream out; out<<"{\"gpuCount\":"<<gpuCount<<",\"device\":\""<<gpuName()<<"\"}"; return env->NewStringUTF(out.str().c_str()); }
    catch(const std::exception& error) { exception(env,error); return nullptr; }
}
extern "C" JNIEXPORT jlong JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_create(JNIEnv* env,jclass,jstring path,jint tile) {
    try { auto engine=std::make_shared<Engine>(); engine->initialize(string(env,path),tile);
        auto handle=nextHandle.fetch_add(1); std::lock_guard<std::mutex> lock(registryMutex); registry.emplace(handle,std::move(engine)); return handle;
    } catch(const std::exception& error) { exception(env,error); return 0; }
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_run(JNIEnv* env,jclass,jlong handle,jfloatArray rgb,jfloatArray dino,jfloatArray noise) {
    try { return lookup(handle)->run(env,rgb,dino,noise); } catch(const std::exception& error) { exception(env,error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_stats(JNIEnv* env,jclass,jlong handle) {
    try { auto value=lookup(handle)->stats(); return env->NewStringUTF(value.c_str()); } catch(const std::exception& error) { exception(env,error); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_resetCancel(JNIEnv* env,jclass,jlong handle) {
    try { lookup(handle)->cancelled.store(false); } catch(const std::exception& error) { exception(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_profile(JNIEnv* env,jclass,jlong handle,jboolean enabled) {
    try { lookup(handle)->setProfiling(enabled); } catch(const std::exception& error) { exception(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_cancel(JNIEnv*,jclass,jlong handle) {
    std::lock_guard<std::mutex> lock(registryMutex); auto it=registry.find(handle); if(it!=registry.end()) it->second->cancelled.store(true);
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeGpu_release(JNIEnv*,jclass,jlong handle) {
    std::shared_ptr<Engine> engine;
    { std::lock_guard<std::mutex> lock(registryMutex); auto it=registry.find(handle); if(it!=registry.end()) { engine=std::move(it->second); registry.erase(it); } }
    if(engine) engine->cancelled.store(true);
}
