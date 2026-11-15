"""Compile numerically checked components for a selected Qualcomm HTP target."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

PROJECT=Path(__file__).resolve().parents[1]
SDK=PROJECT.parent/'tools/qairt/2.44.0.260225'
OUT=PROJECT/'artifacts/qnn'
OUT.mkdir(parents=True,exist_ok=True)
ONNX=PROJECT/'artifacts/onnx'
NATIVE_VAE=None
SOC_MODEL=43
DSP_ARCH='v73'
VTCM_MB=8
EXTENDED_UDMA=False
DLC_SOURCE=None

def build(name):
    source=NATIVE_VAE if NATIVE_VAE is not None and name in ('encoder','decoder') else ONNX
    validation=json.loads((source/f'{name}.validation.json').read_text())
    converter=[sys.executable,str(SDK/'bin/x86_64-windows-msvc/qairt-converter'),
               '--input_network',str(source/f'{name}.onnx'),
               '--output_path',str(OUT/f'{name}.dlc'),'--float_bitwidth','16',
               '--preserve_io_datatype','--target_backend','HTP']
    for label,shape in validation['inputs'].items():
        layout='NCHW' if len(shape)==4 else 'NTF' if len(shape)==3 else None
        if layout:
            converter+=['--source_model_input_layout',label,layout,'--desired_input_layout',label,layout]
    for output in validation['outputs']:
        label,shape=output['output'],output['shape']
        layout='NCHW' if len(shape)==4 else 'NTF' if len(shape)==3 else None
        if layout:
            converter+=['--source_model_output_layout',label,layout,'--desired_output_layout',label,layout]
    logs=OUT/'logs'; logs.mkdir(exist_ok=True)
    if DLC_SOURCE is not None:
        shutil.copyfile(DLC_SOURCE/f'{name}.dlc',OUT/f'{name}.dlc')
    else:
        with (logs/f'convert-{name}.log').open('w') as log:
            subprocess.run(converter,stdout=log,stderr=subprocess.STDOUT,check=True)
    config=OUT/f'{name}.htp.json'
    settings={'graphs':[{'graph_names':[name],'vtcm_mb':VTCM_MB,'fp16_relaxed_precision':1}],
              'devices':[{'soc_model':SOC_MODEL,'dsp_arch':DSP_ARCH}]}
    if EXTENDED_UDMA: settings['context']={'extended_udma':True}
    config.write_text(json.dumps(settings,indent=2))
    extension=OUT/f'{name}.extension.json'
    extension.write_text(json.dumps({'backend_extensions':{
        'shared_library_path':str(SDK/'lib/x86_64-windows-msvc/QnnHtpNetRunExtensions.dll'),
        'config_file_path':str(config)}},indent=2))
    compiler=[str(SDK/'bin/x86_64-windows-msvc/qnn-context-binary-generator.exe'),
              '--model',str(SDK/'lib/x86_64-windows-msvc/QnnModelDlc.dll'),
              '--dlc_path',str(OUT/f'{name}.dlc'),
              '--backend',str(SDK/'lib/x86_64-windows-msvc/QnnHtp.dll'),
              '--binary_file',name,'--output_dir',str(OUT),
              '--config_file',str(extension),'--log_level','info']
    with (logs/f'compile-{name}.log').open('w') as log:
        subprocess.run(compiler,stdout=log,stderr=subprocess.STDOUT,check=True)
    (OUT/f'{name}.build.json').write_text(json.dumps({'socModelId':SOC_MODEL,'dspArch':DSP_ARCH,'vtcmMb':VTCM_MB,'extendedUdma':EXTENDED_UDMA,'sourceDlc':str(DLC_SOURCE) if DLC_SOURCE else None},indent=2))
    print(f'COMPILED {name}: {(OUT/f"{name}.bin").stat().st_size} bytes',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--onnx',type=Path,default=ONNX)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--native-vae-onnx',type=Path,help='Optional native-GroupNorm VAE graphs for HTP; validate device output against stable FP32 ONNX')
    parser.add_argument('--soc-model',type=int,default=43)
    parser.add_argument('--dsp-arch',choices=['v73','v81'],default='v73')
    parser.add_argument('--vtcm-mb',type=int,default=8,help='0 selects the target SoC maximum')
    parser.add_argument('--extended-udma',action='store_true',help='Enable far mapping on HTP v81+')
    parser.add_argument('--dlc-source',type=Path,help='Reuse already converted, target-independent FP16 DLCs')
    args=parser.parse_args()
    ONNX=args.onnx.resolve(); OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
    NATIVE_VAE=args.native_vae_onnx.resolve() if args.native_vae_onnx else None
    SOC_MODEL=args.soc_model; DSP_ARCH=args.dsp_arch; VTCM_MB=args.vtcm_mb; EXTENDED_UDMA=args.extended_udma
    DLC_SOURCE=args.dlc_source.resolve() if args.dlc_source else None
    if EXTENDED_UDMA and DSP_ARCH!='v81': parser.error('Extended uDMA requires v81+')
    stages=args.stages or ['encoder','dino','dit_prepare']+[f'dit_blocks_{i:02}' for i in range(0,28,4)]+['dit_finish','decoder']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(build,name) for name in stages]
        for future in concurrent.futures.as_completed(futures):
            future.result()
