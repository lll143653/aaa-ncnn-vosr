#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <set>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpSystemContext.h"
#include "profile_trace.h"

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void require(bool value,const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(Qnn_ErrorHandle_t code,const std::string& operation) {
  if (code!=QNN_SUCCESS) throw std::runtime_error(operation+" failed: "+std::to_string(code));
}
void logCallback(const char* format,QnnLog_Level_t level,uint64_t,va_list args) {
  char text[2048]; vsnprintf(text,sizeof(text),format,args);
  __android_log_write(level==QNN_LOG_LEVEL_ERROR?ANDROID_LOG_ERROR:ANDROID_LOG_WARN,"VOSR-QNN",text);
}
struct Mapping {
  void* data=MAP_FAILED; size_t size=0;
  explicit Mapping(const std::string& path) {
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    require(fd>=0,"Cannot open model "+path);
    struct stat status{};
    if (fstat(fd,&status)!=0 || status.st_size<=0) { close(fd); throw std::runtime_error("Invalid model file "+path); }
    size=static_cast<size_t>(status.st_size);
    data=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0); close(fd);
    require(data!=MAP_FAILED,"Cannot map model "+path);
  }
  ~Mapping() { if (data!=MAP_FAILED) munmap(data,size); }
};
const char* tensorName(const Qnn_Tensor_t& tensor) {
  if (tensor.version==QNN_TENSOR_VERSION_1) return tensor.v1.name;
  require(tensor.version==QNN_TENSOR_VERSION_2,"Unsupported tensor metadata version"); return tensor.v2.name;
}
size_t elements(const Qnn_Tensor_t& tensor) {
  uint32_t rank; const uint32_t* dims; Qnn_DataType_t type;
  if (tensor.version==QNN_TENSOR_VERSION_1) { rank=tensor.v1.rank; dims=tensor.v1.dimensions; type=tensor.v1.dataType; }
  else { require(tensor.version==QNN_TENSOR_VERSION_2,"Unsupported tensor metadata version"); rank=tensor.v2.rank; dims=tensor.v2.dimensions; type=tensor.v2.dataType; }
  require(type==QNN_DATATYPE_FLOAT_32,"Model IO must be FLOAT32: "+std::string(tensorName(tensor)));
  size_t count=1;
  for (uint32_t i=0;i<rank;i++) { require(dims[i]>0 && count<=UINT32_MAX/sizeof(float)/dims[i],"Invalid tensor size"); count*=dims[i]; }
  return count;
}
void bind(Qnn_Tensor_t& tensor,std::vector<float>& buffer) {
  require(elements(tensor)==buffer.size(),"Tensor binding size mismatch");
  Qnn_ClientBuffer_t client{buffer.data(),static_cast<uint32_t>(buffer.size()*sizeof(float))};
  if (tensor.version==QNN_TENSOR_VERSION_1) { tensor.v1.memType=QNN_TENSORMEMTYPE_RAW; tensor.v1.clientBuf=client; }
  else { tensor.v2.memType=QNN_TENSORMEMTYPE_RAW; tensor.v2.clientBuf=client; }
}
struct Session {
  std::string name;
  std::unique_ptr<Mapping> binary;
  QnnSystemContext_Handle_t systemContext=nullptr;
  Qnn_ContextHandle_t context=nullptr;
  Qnn_ProfileHandle_t profile=nullptr;
  Qnn_GraphHandle_t graph=nullptr;
  std::vector<Qnn_Tensor_t> inputs,outputs;
  uint64_t scratchBytes=0;
};
class Engine {
  void* htpLibrary=nullptr; void* systemLibrary=nullptr;
  QNN_INTERFACE_VER_TYPE api{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_LogHandle_t logger=nullptr;
  Qnn_BackendHandle_t backend=nullptr;
  Qnn_DeviceHandle_t device=nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf{};
  uint32_t powerId=0;
  bool powerReady=false,lastBoostAccepted=false;
  std::vector<std::unique_ptr<Session>> sessions;
  std::map<std::string,std::vector<float>> buffers;
  std::mutex runMutex;
  std::vector<double> lastTimes;
  double loadMs=0,lastRunMs=0;
  bool sharedScratch=false;
  bool extendedUdma=false;
  bool streamDit=false;
  uint64_t contextCreates=0;
  size_t residentContexts=0,maxResidentContexts=0;
  double contextLoadMs=0,contextFreeMs=0;
  uint64_t scratchMax=0,scratchSum=0;
  uint64_t runCount=0;
  bool profiling=false;
  int profileStartStage=0;
  std::vector<double> initializationTimes;
  std::vector<std::string> hardwareProfiles;
  double inputCopyMs=0,outputCopyMs=0,profileCollectionMs=0,powerVoteMs=0;
  std::string collectProfile(const std::string& stage,Qnn_ProfileHandle_t profile) {
    const QnnProfile_EventId_t* roots=nullptr; uint32_t count=0;
    auto status=api.profileGetEvents(profile,&roots,&count);
    std::ostringstream out; out<<"{\"stage\":"<<jsonString(stage)<<",\"getEventsStatus\":"<<status<<",\"events\":[";
    bool first=true; size_t events=0; std::set<QnnProfile_EventId_t> seen;
    std::function<void(QnnProfile_EventId_t,int)> visit=[&](QnnProfile_EventId_t id,int depth) {
      if(depth>8 || events>=10000 || !seen.insert(id).second) return;
      QnnProfile_EventData_t data{}; auto result=api.profileGetEventData(id,&data);
      if(!first) out<<","; first=false; events++;
      out<<"{\"id\":"<<id<<",\"depth\":"<<depth<<",\"status\":"<<result;
      if(result==QNN_SUCCESS) out<<",\"type\":"<<data.type<<",\"unit\":"<<data.unit<<",\"value\":"<<data.value<<",\"identifier\":"<<jsonString(data.identifier?data.identifier:"");
      out<<"}";
      const QnnProfile_EventId_t* children=nullptr; uint32_t size=0;
      if(api.profileGetSubEvents(id,&children,&size)==QNN_SUCCESS) for(uint32_t i=0;i<size;i++) visit(children[i],depth+1);
    };
    if(status==QNN_SUCCESS) for(uint32_t i=0;i<count;i++) visit(roots[i],0);
    out<<"],\"eventCount\":"<<events<<",\"truncated\":"<<(events>=10000?"true":"false")<<"}"; return out.str();
  }
  bool vote(bool active) {
    if (!powerReady) return false;
    QnnHtpPerfInfrastructure_PowerConfig_t config=QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    config.option=QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& c=config.dcvsV3Config;
    c.contextId=powerId; c.setDcvsEnable=1; c.dcvsEnable=active?0:1;
    c.powerMode=active?QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE:QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
    c.setSleepLatency=1; c.sleepLatency=active?40:2000;
    c.setSleepDisable=1; c.sleepDisable=active?1:0;
    auto corner=active?DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER:DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    c.setBusParams=1; c.busVoltageCornerMin=corner; c.busVoltageCornerTarget=corner; c.busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    c.setCoreParams=1; c.coreVoltageCornerMin=corner; c.coreVoltageCornerTarget=corner; c.coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[]={&config,nullptr};
    const bool accepted=perf.setPowerConfig(powerId,configs)==QNN_SUCCESS;
    if(active) lastBoostAccepted=accepted;
    return accepted;
  }
  std::string binding(int stage,const std::string& tensor,bool output) {
    if (stage==1 && !output) { require(tensor=="image","Unexpected DINO input"); return "dino_input"; }
    if (stage>=3 && stage<=9) {
      if (output) { require(tensor=="next_tokens","Unexpected DiT output"); return stage%2?"tokens_b":"tokens_a"; }
      if (tensor=="tokens") return stage%2?"tokens_a":"tokens_b";
    }
    if (stage==2 && output && tensor=="tokens") return "tokens_a";
    if (stage==10 && !output && tensor=="tokens") return "tokens_b";
    if (stage==11 && output) { require(tensor=="rgb","Unexpected decoder output"); return "pixels"; }
    return tensor;
  }
  void bindTensor(Qnn_Tensor_t& tensor,int stage,bool output) {
    auto key=binding(stage,tensorName(tensor),output);
    auto count=elements(tensor);
    auto& buffer=buffers[key];
    if (buffer.empty()) buffer.resize(count);
    require(buffer.size()==count,"Inconsistent binding "+key);
    bind(tensor,buffer);
  }
  void load(const std::string& directory,const std::string& name,int index) {
    auto item=std::make_unique<Session>(); item->name=name;
    sessions.push_back(std::move(item)); auto& session=*sessions.back();
    session.binary=std::make_unique<Mapping>(directory+"/"+name+".bin");
    check(system.systemContextCreate(&session.systemContext),"systemContextCreate");
    const QnnSystemContext_BinaryInfo_t* info=nullptr; Qnn_ContextBinarySize_t infoSize=0;
    check(system.systemContextGetBinaryInfo(session.systemContext,session.binary->data,session.binary->size,&info,&infoSize),"systemContextGetBinaryInfo "+name);
    uint32_t graphCount=0; QnnSystemContext_GraphInfo_t* graphs=nullptr;
    switch (info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: graphCount=info->contextBinaryInfoV1.numGraphs; graphs=info->contextBinaryInfoV1.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: graphCount=info->contextBinaryInfoV2.numGraphs; graphs=info->contextBinaryInfoV2.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: graphCount=info->contextBinaryInfoV3.numGraphs; graphs=info->contextBinaryInfoV3.graphs; break;
      default: throw std::runtime_error("Unsupported context binary metadata");
    }
    require(graphCount==1 && graphs,"Expected one graph in "+name);
    if(graphs[0].version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      const auto& g=graphs[0].graphInfoV3;
      if(g.graphBlobInfo && g.graphBlobInfoSize>=sizeof(QnnHtpSystemContext_GraphBlobInfo_t)) {
        const auto* h=static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(g.graphBlobInfo);
        if(h->version==QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1)
          session.scratchBytes=h->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
        else if(h->version==QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V2 && g.graphBlobInfoSize>=sizeof(QnnHtpSystemContext_GraphBlobInfoV2_t))
          session.scratchBytes=static_cast<const QnnHtpSystemContext_GraphBlobInfoV2_t*>(g.graphBlobInfo)->ddrTensorSize;
      }
    }
    scratchMax=std::max(scratchMax,session.scratchBytes); scratchSum+=session.scratchBytes;
    const char* graphName=nullptr; Qnn_Tensor_t *inputs=nullptr,*outputs=nullptr; uint32_t ni=0,no=0;
    #define READ_GRAPH(versionName) { const auto& g=graphs[0].versionName; graphName=g.graphName; inputs=g.graphInputs; outputs=g.graphOutputs; ni=g.numGraphInputs; no=g.numGraphOutputs; }
    switch (graphs[0].version) {
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: READ_GRAPH(graphInfoV1); break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: READ_GRAPH(graphInfoV2); break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: READ_GRAPH(graphInfoV3); break;
      default: throw std::runtime_error("Unsupported graph metadata");
    }
    #undef READ_GRAPH
    require(graphName && name==graphName,"Context graph name mismatch: "+name);
    session.inputs.assign(inputs,inputs+ni); session.outputs.assign(outputs,outputs+no);
    for (auto& tensor:session.inputs) bindTensor(tensor,index,false);
    for (auto& tensor:session.outputs) bindTensor(tensor,index,true);
    if(profiling && index>=profileStartStage) check(api.profileCreate(backend,QNN_PROFILE_LEVEL_DETAILED,&session.profile),"profileCreate "+name);
  }
  void createContext(size_t index) {
    auto& session=*sessions[index];
    QnnHtpContext_CustomConfig_t group=QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    group.option=QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    group.groupRegistration.firstGroupHandle=index?sessions[0]->context:nullptr;
    group.groupRegistration.maxSpillFillBuffer=scratchMax;
    QnnContext_Config_t config=QNN_CONTEXT_CONFIG_INIT;
    config.option=QNN_CONTEXT_CONFIG_OPTION_CUSTOM; config.customConfig=&group;
    QnnHtpContext_CustomConfig_t far=QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    far.option=QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA; far.useExtendedUdma=true;
    QnnContext_Config_t farConfig=QNN_CONTEXT_CONFIG_INIT;
    farConfig.option=QNN_CONTEXT_CONFIG_OPTION_CUSTOM; farConfig.customConfig=&far;
    std::vector<const QnnContext_Config_t*> configs;
    if(sharedScratch) configs.push_back(&config);
    if(extendedUdma) configs.push_back(&farConfig);
    configs.push_back(nullptr);
    __android_log_print(ANDROID_LOG_INFO,"VOSR-QNN","Load %s: scratch %llu / shared %llu",session.name.c_str(),
        static_cast<unsigned long long>(session.scratchBytes),static_cast<unsigned long long>(scratchMax));
    check(api.contextCreateFromBinary(backend,device,configs.size()>1?configs.data():nullptr,session.binary->data,session.binary->size,&session.context,session.profile),"contextCreateFromBinary "+session.name);
    check(api.graphRetrieve(session.context,session.name.c_str(),&session.graph),"graphRetrieve "+session.name);
    contextCreates++; residentContexts++; maxResidentContexts=std::max(maxResidentContexts,residentContexts);
  }
public:
  std::atomic<bool> cancelled{false};
  ~Engine() {
    for (auto it=sessions.rbegin();it!=sessions.rend();++it) {
      if ((*it)->context && api.contextFree) api.contextFree((*it)->context,nullptr);
      if ((*it)->profile && api.profileFree) api.profileFree((*it)->profile);
      if ((*it)->systemContext && system.systemContextFree) system.systemContextFree((*it)->systemContext);
      (*it)->binary.reset();
    }
    if (powerReady) { vote(false); perf.destroyPowerConfigId(powerId); }
    if (device && api.deviceFree) api.deviceFree(device);
    if (backend && api.backendFree) api.backendFree(backend);
    if (logger && api.logFree) api.logFree(logger);
    if (htpLibrary) dlclose(htpLibrary);
    if (systemLibrary) dlclose(systemLibrary);
  }
  void initialize(const std::string& directory,const std::string& nativeDirectory,bool detailed,int firstProfileStage,bool useExtendedUdma,bool streamLargeDit) {
    auto start=Clock::now();
    profiling=detailed;
    extendedUdma=useExtendedUdma;
    require(firstProfileStage>=0 && firstProfileStage<12,"Invalid first profile stage");
    profileStartStage=firstProfileStage;
    setenv("ADSP_LIBRARY_PATH",(nativeDirectory+";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp").c_str(),1);
    htpLibrary=dlopen((nativeDirectory+"/libQnnHtp.so").c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!htpLibrary) throw std::runtime_error(std::string("Cannot load HTP: ")+dlerror());
    systemLibrary=dlopen((nativeDirectory+"/libQnnSystem.so").c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!systemLibrary) throw std::runtime_error(std::string("Cannot load QNN System: ")+dlerror());
    auto getProviders=reinterpret_cast<decltype(&QnnInterface_getProviders)>(dlsym(htpLibrary,"QnnInterface_getProviders"));
    require(getProviders,"Missing QNN interface");
    const QnnInterface_t** providers=nullptr; uint32_t count=0;
    check(getProviders(&providers,&count),"QnnInterface_getProviders"); bool found=false;
    for (uint32_t i=0;i<count;i++) if (providers[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR && providers[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR) { api=providers[i]->QNN_INTERFACE_VER_NAME; found=true; break; }
    require(found,"Incompatible QNN core API");
    auto getSystem=reinterpret_cast<decltype(&QnnSystemInterface_getProviders)>(dlsym(systemLibrary,"QnnSystemInterface_getProviders"));
    require(getSystem,"Missing QNN System interface");
    const QnnSystemInterface_t** systemProviders=nullptr;
    check(getSystem(&systemProviders,&count),"QnnSystemInterface_getProviders"); found=false;
    for (uint32_t i=0;i<count;i++) if (systemProviders[i]->systemApiVersion.major==QNN_SYSTEM_API_VERSION_MAJOR && systemProviders[i]->systemApiVersion.minor>=QNN_SYSTEM_API_VERSION_MINOR) { system=systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME; found=true; break; }
    require(found,"Incompatible QNN System API");
    check(api.logCreate(logCallback,QNN_LOG_LEVEL_WARN,&logger),"logCreate");
    check(api.backendCreate(logger,nullptr,&backend),"backendCreate HTP");
    check(api.deviceCreate(logger,nullptr,&device),"deviceCreate HTP");
    QnnDevice_Infrastructure_t infrastructure=nullptr;
    if (api.deviceGetInfrastructure && api.deviceGetInfrastructure(&infrastructure)==QNN_SUCCESS && infrastructure) {
      auto* htp=reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infrastructure);
      if (htp->infraType==QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        perf=htp->perfInfra;
        powerReady=perf.createPowerConfigId && perf.destroyPowerConfigId && perf.setPowerConfig && perf.createPowerConfigId(0,0,&powerId)==QNN_SUCCESS;
      }
    }
    const std::vector<std::string> names={"encoder","dino","dit_prepare","dit_blocks_00","dit_blocks_04","dit_blocks_08","dit_blocks_12","dit_blocks_16","dit_blocks_20","dit_blocks_24","dit_finish","decoder"};
    for (int i=0;i<static_cast<int>(names.size());i++) {
      TraceSpan trace("vosr.load.npu."+names[i]); auto stageStart=Clock::now();
      load(directory,names[i],i); initializationTimes.push_back(ms(stageStart));
    }
    sharedScratch=buffers.at("rgb").size()==3*1024*1024;
    streamDit=sharedScratch && streamLargeDit;
    if(sharedScratch) require(scratchMax>0,"Missing HTP scratch metadata for large tiles");
    for(size_t i=0;i<sessions.size();i++) {
      if(streamDit && i>=3 && i<=9) continue;
      auto stageStart=Clock::now(); createContext(i); initializationTimes[i]+=ms(stageStart);
    }
    lastTimes.resize(names.size()); loadMs=ms(start);
    __android_log_print(ANDROID_LOG_INFO,"VOSR-QNN","HTP ready: 12 graphs, %zu resident contexts, load %.1f ms",residentContexts,loadMs);
  }
  void copyInput(JNIEnv* env,jfloatArray array,const std::string& name) {
    auto& buffer=buffers.at(name);
    require(array && static_cast<size_t>(env->GetArrayLength(array))==buffer.size(),"Invalid patch input "+name);
    env->GetFloatArrayRegion(array,0,static_cast<jsize>(buffer.size()),buffer.data());
    require(!env->ExceptionCheck(),"Cannot copy patch input");
  }
  jfloatArray run(JNIEnv* env,jfloatArray rgb,jfloatArray dino,jfloatArray noise) {
    std::lock_guard<std::mutex> lock(runMutex); auto start=Clock::now();
    TraceSpan trace("vosr.npu.patch");
    vote(true); powerVoteMs=ms(start);
    struct IdleVote { Engine* engine; ~IdleVote() { engine->vote(false); } } idle{this};
    hardwareProfiles.clear(); profileCollectionMs=0; contextLoadMs=0; contextFreeMs=0;
    auto copyStart=Clock::now();
    copyInput(env,rgb,"rgb"); copyInput(env,dino,"dino_input"); copyInput(env,noise,"noise");
    inputCopyMs=ms(copyStart);
    for (size_t i=0;i<sessions.size();i++) {
      if (cancelled.load()) throw std::runtime_error("已取消处理");
      auto& session=*sessions[i];
      const bool transient=streamDit && i>=3 && i<=9;
      if(transient) { auto loadStart=Clock::now(); createContext(i); contextLoadMs+=ms(loadStart); }
      struct TransientContext {
        Engine* engine; Session* session; bool active;
        ~TransientContext() {
          if(active && session->context) {
            auto start=Clock::now();
            engine->api.contextFree(session->context,nullptr);
            session->context=nullptr; session->graph=nullptr;
            engine->residentContexts--; engine->contextFreeMs+=ms(start);
          }
        }
      } transientContext{this,&session,transient};
      Qnn_ProfileHandle_t profile=profiling?session.profile:nullptr;
      TraceSpan stageTrace("vosr.npu."+session.name); auto stageStart=Clock::now();
      auto executeStatus=api.graphExecute(session.graph,session.inputs.data(),session.inputs.size(),session.outputs.data(),session.outputs.size(),profile,nullptr);
      if(profile && executeStatus==QNN_GRAPH_ERROR_SET_PROFILE) {
        // A profiling failure must not discard earlier stage events or abort restoration.
        // Keep this sample distinct from latency benchmarks: this stage is retried.
        hardwareProfiles.push_back("{\"stage\":"+jsonString(session.name)+",\"executeStatus\":"+std::to_string(executeStatus)+",\"retriedWithoutProfile\":true,\"events\":[]}");
        profile=nullptr;
        executeStatus=api.graphExecute(session.graph,session.inputs.data(),session.inputs.size(),session.outputs.data(),session.outputs.size(),nullptr,nullptr);
      }
      check(executeStatus,"graphExecute "+session.name);
      lastTimes[i]=ms(stageStart);
      if(profile) { auto collectStart=Clock::now(); hardwareProfiles.push_back(collectProfile(session.name,profile)); profileCollectionMs+=ms(collectStart); }
    }
    if (cancelled.load()) throw std::runtime_error("已取消处理");
    auto& pixels=buffers.at("pixels");
    copyStart=Clock::now();
    auto result=env->NewFloatArray(static_cast<jsize>(pixels.size())); require(result,"Cannot allocate patch output");
    env->SetFloatArrayRegion(result,0,static_cast<jsize>(pixels.size()),pixels.data());
    outputCopyMs=ms(copyStart);
    lastRunMs=ms(start); runCount++; return result;
  }
  void setProfiling(bool enabled) { std::lock_guard<std::mutex> lock(runMutex); profiling=enabled; }
  std::string stats() {
    std::lock_guard<std::mutex> lock(runMutex); std::ostringstream out;
    out<<"{\"engine\":\"JNI_QNN_HTP_RESIDENT\",\"contextCreateCount\":"<<contextCreates<<",\"graphCount\":"<<sessions.size()
       <<",\"modelLoadMs\":"<<loadMs<<",\"patchRunCount\":"<<runCount<<",\"lastPatchMs\":"<<lastRunMs
       <<",\"powerVoteSupported\":"<<(powerReady?"true":"false")<<",\"boostVoteAccepted\":"<<(lastBoostAccepted?"true":"false")
       <<",\"sharedScratch\":"<<(sharedScratch?"true":"false")<<",\"scratchMaxBytes\":"<<scratchMax<<",\"scratchSumBytes\":"<<scratchSum
       <<",\"streamDit\":"<<(streamDit?"true":"false")<<",\"residentContextCount\":"<<residentContexts<<",\"maxResidentContextCount\":"<<maxResidentContexts
       <<",\"extendedUdma\":"<<(extendedUdma?"true":"false")
       <<",\"contextLoadMs\":"<<contextLoadMs<<",\"contextFreeMs\":"<<contextFreeMs
       <<",\"profilingEnabled\":"<<(profiling?"true":"false")<<",\"inputCopyMs\":"<<inputCopyMs<<",\"outputCopyMs\":"<<outputCopyMs
       <<",\"profileStartStage\":"<<profileStartStage
       <<",\"profileCollectionMs\":"<<profileCollectionMs<<",\"powerVoteMs\":"<<powerVoteMs
       <<",\"subprocessCount\":0,\"intermediateTensorFileBytes\":0,\"stageMs\":[";
    for(size_t i=0;i<lastTimes.size();i++) { if(i) out<<","; out<<lastTimes[i]; }
    out<<"],\"initializationStageMs\":[";
    for(size_t i=0;i<initializationTimes.size();i++) { if(i) out<<","; out<<initializationTimes[i]; }
    out<<"],\"qnnProfiles\":[";
    for(size_t i=0;i<hardwareProfiles.size();i++) { if(i) out<<","; out<<hardwareProfiles[i]; }
    out<<"]}"; return out.str();
  }
};
std::mutex registryMutex;
std::map<jlong,std::shared_ptr<Engine>> registry;
std::atomic<jlong> nextHandle{1};
std::shared_ptr<Engine> lookup(jlong handle) {
  std::lock_guard<std::mutex> lock(registryMutex); auto item=registry.find(handle);
  require(item!=registry.end(),"QNN engine is closed"); return item->second;
}
std::string string(JNIEnv* env,jstring value) {
  require(value,"Missing path"); const char* chars=env->GetStringUTFChars(value,nullptr);
  require(chars,"Cannot read path"); std::string result(chars); env->ReleaseStringUTFChars(value,chars); return result;
}
void exception(JNIEnv* env,const std::exception& error) {
  __android_log_print(ANDROID_LOG_ERROR,"VOSR-QNN","%s",error.what());
  if(!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),error.what());
}
}
extern "C" JNIEXPORT jlong JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_create(JNIEnv* env,jclass,jstring directory,jstring nativeDirectory,jboolean detailed,jint firstProfileStage,jboolean extendedUdma,jboolean streamDit) {
  try {
    auto engine=std::make_shared<Engine>(); engine->initialize(string(env,directory),string(env,nativeDirectory),detailed,firstProfileStage,extendedUdma,streamDit);
    auto handle=nextHandle.fetch_add(1); std::lock_guard<std::mutex> lock(registryMutex); registry.emplace(handle,std::move(engine)); return handle;
  } catch(const std::exception& error) { exception(env,error); return 0; }
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_run(JNIEnv* env,jclass,jlong handle,jfloatArray rgb,jfloatArray dino,jfloatArray noise) {
  try { return lookup(handle)->run(env,rgb,dino,noise); } catch(const std::exception& error) { exception(env,error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_stats(JNIEnv* env,jclass,jlong handle) {
  try { auto text=lookup(handle)->stats(); return env->NewStringUTF(text.c_str()); } catch(const std::exception& error) { exception(env,error); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_resetCancel(JNIEnv* env,jclass,jlong handle) {
  try { lookup(handle)->cancelled.store(false); } catch(const std::exception& error) { exception(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_profile(JNIEnv* env,jclass,jlong handle,jboolean enabled) {
  try { lookup(handle)->setProfiling(enabled); } catch(const std::exception& error) { exception(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_cancel(JNIEnv*,jclass,jlong handle) {
  std::lock_guard<std::mutex> lock(registryMutex); auto item=registry.find(handle); if(item!=registry.end()) item->second->cancelled.store(true);
}
extern "C" JNIEXPORT void JNICALL Java_ai_vosr_vosr_1restore_NativeQnn_release(JNIEnv*,jclass,jlong handle) {
  std::shared_ptr<Engine> engine;
  { std::lock_guard<std::mutex> lock(registryMutex); auto item=registry.find(handle); if(item!=registry.end()) {engine=std::move(item->second);registry.erase(item);} }
  if(engine) engine->cancelled.store(true);
}
