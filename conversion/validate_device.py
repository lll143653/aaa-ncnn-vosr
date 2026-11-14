"""Execute all compiled graphs on the attached phone and compare with FP32 ORT."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import onnxruntime as ort

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
ADB=ROOT/'tools/android-sdk/platform-tools/adb.exe'
DEVICE=os.environ.get('ANDROID_SERIAL')
REMOTE='/data/local/tmp/vosr/qnn'
LOCAL=PROJECT/'artifacts/device512'
LOCAL.mkdir(parents=True,exist_ok=True)
REF=PROJECT/'artifacts/reference'
QNN=PROJECT/'artifacts/qnn'
NAMES=['encoder','dino','dit_prepare']+[f'dit_blocks_{i:02}' for i in range(0,28,4)]+['dit_finish','decoder']

def adb(*args):
    result=subprocess.run([str(ADB),*(['-s',DEVICE] if DEVICE else []),*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout+result.stderr

def binding(stage,name,output=False):
    if stage=='dino' and not output: return 'dino_input'
    if stage.startswith('dit_blocks') and output: return 'tokens'
    if stage=='decoder' and output: return 'pixels'
    return name

def main():
    run_prefix=f'validation512_{time.time_ns()}'
    for name in ('rgb','dino_input','noise'):
        adb('push',REF/f'{name}.raw',f'{REMOTE}/{name}.raw')
    values={name:np.load(REF/f'{name}.npy') for name in ('rgb','dino_input','noise')}
    paths={name:f'{REMOTE}/{name}.raw' for name in values}
    report=[]
    stages=[]
    start=time.perf_counter()
    for name in NAMES:
        validation=json.loads((PROJECT/f'artifacts/onnx/{name}.validation.json').read_text())
        binary=QNN/f'{name}.bin'
        print(f'PUSH {name}',flush=True)
        adb('push',binary,f'{REMOTE}/{name}.bin')
        inputs={key:binding(name,key) for key in validation['inputs']}
        listing=LOCAL/f'{name}-inputs.txt'
        listing.write_text(' '.join(f'{key}:={paths[alias]}' for key,alias in inputs.items())+'\n')
        adb('push',listing,f'{REMOTE}/{name}-inputs.txt')
        command=(f'cd {REMOTE}; LD_LIBRARY_PATH=. '
            'ADSP_LIBRARY_PATH=".;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp" '
            f'./qnn-net-run --backend ./libQnnHtp.so --retrieve_context ./{name}.bin '
            f'--input_list ./{name}-inputs.txt --output_dir ./{run_prefix}_{name}-output '
            '--profiling_level detailed --log_level info')
        before=time.perf_counter()
        log=adb('shell',command)
        elapsed=time.perf_counter()-before
        (LOCAL/f'{name}.log').write_text(log,encoding='utf-8')
        print(f'EXECUTED {name}: {elapsed:.3f}s',flush=True)
        options=ort.SessionOptions(); options.intra_op_num_threads=8
        session=ort.InferenceSession(str(PROJECT/f'artifacts/onnx/{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        expected=session.run(None,{key:values[alias] for key,alias in inputs.items()})
        metrics=[]; outputs=[]
        for metadata,reference in zip(validation['outputs'],expected):
            label=metadata['output']; alias=binding(name,label,True)
            remote=f'{REMOTE}/{run_prefix}_{name}-output/Result_0/{label}.raw'
            local=LOCAL/f'{name}-{label}.raw'
            adb('pull',remote,local)
            actual=np.fromfile(local,dtype=np.float32).reshape(reference.shape)
            rmse=float(np.sqrt(np.mean((actual-reference)**2)))
            nrmse=rmse/max(float(np.sqrt(np.mean(reference**2))),1e-6)
            max_abs=float(np.max(np.abs(actual-reference)))
            if not np.isfinite(actual).all() or nrmse>.05:
                raise AssertionError(f'NPU precision check failed {name}/{label}: NRMSE={nrmse}')
            metrics.append({'name':label,'shape':list(actual.shape),'rmse':rmse,'nrmse':nrmse,'max_abs':max_abs})
            values[alias]=reference
            paths[alias]=remote
            outputs.append({'name':label,'binding':alias,'bytes':actual.nbytes})
        del session
        profile=LOCAL/f'{name}-profiling.log'
        adb('pull',f'{REMOTE}/{run_prefix}_{name}-output/qnn-profiling-data_0.log',profile)
        viewer=ROOT/'tools/qairt/2.44.0.260225/bin/x86_64-windows-msvc/qnn-profile-viewer.exe'
        with (LOCAL/f'{name}-profile.txt').open('w') as f:
            subprocess.run([str(viewer),'--input_log',str(profile),'--output',str(LOCAL/f'{name}-profile.csv')],stdout=f,stderr=subprocess.STDOUT,check=True)
        report.append({'stage':name,'wall_s':elapsed,'outputs':metrics})
        (LOCAL/'validation-progress.json').write_text(json.dumps(report,indent=2))
        stages.append({'name':name,'label':name,'file':binary.name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                       'inputs':inputs,'outputs':outputs})
    actual=np.fromfile(LOCAL/'decoder-rgb.raw',dtype=np.float32).reshape(values['pixels'].shape)
    reference=values['pixels']
    rmse=float(np.sqrt(np.mean((actual-reference)**2)))
    psnr=float(20*np.log10(2/max(rmse,1e-12)))
    if psnr<30:
        raise AssertionError(f'End-to-end FP16 versus FP32 PSNR too low: {psnr}')
    Image.fromarray(((actual[0].transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)).save(LOCAL/'restored-npu.png')
    summary={'device':DEVICE,'backend':'QNN_HTP_V73_FP16','total_host_validation_s':time.perf_counter()-start,
             'pixel_psnr_vs_fp32_db':psnr,'pixel_rmse':rmse,'stages':report,'passed':True}
    (LOCAL/'validation.json').write_text(json.dumps(summary,indent=2))
    manifest={'formatVersion':1,'model':'VOSR_0.5B_os','backend':'QNN_HTP_V73_FP16','tileSize':512,
              'deviceValidated':True,'referencePsnrDb':psnr,'stages':stages}
    (QNN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'PASS complete NPU pipeline: PSNR={psnr:.3f} dB',flush=True)

if __name__=='__main__':
    main()
