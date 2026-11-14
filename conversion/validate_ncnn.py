"""Check all ncnn 0.5B boundaries against FP32 ONNX and package the GPU bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import importlib.metadata

import ncnn
import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
NAMES=['encoder','dino','dit_prepare']+[f'dit_blocks_{i:02}' for i in range(0,28,4)]+['dit_finish','decoder']

def run_ncnn(directory,name,inputs):
    net=ncnn.Net(); net.opt.num_threads=8
    net.opt.use_fp16_arithmetic=False; net.opt.use_fp16_storage=False; net.opt.use_fp16_packed=False
    assert net.load_param(str(directory/f'{name}.ncnn.param'))==0
    assert net.load_model(str(directory/f'{name}.ncnn.bin'))==0
    ex=net.create_extractor(); keep=[]
    for i,array in enumerate(inputs):
        arr=np.asarray(array[0],dtype=np.float32)
        if arr.ndim==3: mat=ncnn.Mat(arr.shape[2],arr.shape[1],arr.shape[0])
        elif arr.ndim==2: mat=ncnn.Mat(arr.shape[1],arr.shape[0])
        else: raise ValueError(arr.shape)
        np.asarray(mat)[:]=arr; keep.append(mat)
        assert ex.input(f'in{i}',mat)==0
    values=[]
    for i in range(2 if name=='dit_prepare' else 1):
        code,value=ex.extract(f'out{i}'); assert code==0
        values.append(np.asarray(value,dtype=np.float32)[None].copy())
    return values

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--onnx',type=Path,required=True)
    parser.add_argument('--models',type=Path,default=PROJECT/'artifacts/ncnn-0.5b')
    args=parser.parse_args()
    values={n:np.load(args.reference/f'{n}.npy') for n in ('rgb','dino_input','noise')}
    reference=values.copy(); reports=[]; manifest=[]
    options=ort.SessionOptions(); options.intra_op_num_threads=8
    for name in NAMES:
        metadata=json.loads((args.onnx/f'{name}.validation.json').read_text())
        inputs=list(metadata['inputs'])
        keys=['dino_input' if name=='dino' else n for n in inputs]
        start=time.perf_counter()
        actual=run_ncnn(args.models,name,[values[k] for k in keys])
        sess=ort.InferenceSession(str(args.onnx/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        expected=sess.run(None,dict(zip(inputs,[values[k] for k in keys])))
        independent=sess.run(None,dict(zip(inputs,[reference[k] for k in keys])))
        metrics=[]
        for out,a,b,ref in zip(metadata['outputs'],actual,expected,independent):
            shape=tuple(out['shape']); a=a.reshape(shape)
            assert np.isfinite(a).all()
            nrmse=float(np.sqrt(np.mean((a-b)**2))/max(float(np.sqrt(np.mean(b**2))),1e-6))
            assert nrmse<.01,(name,nrmse)
            key=out['output']
            if name.startswith('dit_blocks'): key='tokens'
            if name=='decoder': key='pixels'
            values[key]=a; reference[key]=ref
            metrics.append({'output':key,'shape':list(shape),'nrmse':nrmse,'maxAbs':float(np.max(np.abs(a-b)))})
        report={'name':name,'seconds':time.perf_counter()-start,'metrics':metrics}
        reports.append(report); print(json.dumps(report),flush=True)
        binary=args.models/f'{name}.ncnn.bin'; param=args.models/f'{name}.ncnn.param'
        manifest.append({'name':name,'file':binary.name,'bytes':binary.stat().st_size,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                         'param':param.name,'paramSha256':hashlib.sha256(param.read_bytes()).hexdigest()})
        del sess
    a=values['pixels']; b=reference['pixels']
    rmse=float(np.sqrt(np.mean((a-b)**2))); psnr=float(20*np.log10(2/rmse))
    assert psnr>45,psnr
    Image.fromarray(((a[0].transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)).save(args.models/'validation-output.png')
    result={'passed':True,'precision':'FP16 stored weights / FP32 arithmetic','pipelinePsnrDb':psnr,'rmse':rmse,'stages':reports}
    (args.models/'validation.json').write_text(json.dumps(result,indent=2))
    bundle={'model':'VOSR_0.5B_os','backend':'NCNN_VULKAN_FP16_WEIGHTS_FP32','tileSize':512,
            'converterVersion':'pnnx 20261020','runtimeVersion':importlib.metadata.version('ncnn'),
            'gpuProfile':{'fp16Arithmetic':False,'subgroupOps':False,'winogradConvolution':False,'hostReductionFallback':True},
            'conversionValidated':True,'deviceValidated':False,'stages':manifest}
    (args.models/'manifest.json').write_text(json.dumps(bundle,indent=2))
    print(json.dumps({'passed':True,'pipelinePsnrDb':psnr}),flush=True)

if __name__=='__main__': main()
