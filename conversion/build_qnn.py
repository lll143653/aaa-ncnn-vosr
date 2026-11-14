"""Compile numerically checked ONNX components to SM8550 HTP v73 contexts."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
SDK=PROJECT.parent/'tools/qairt/2.44.0.260225'
OUT=PROJECT/'artifacts/qnn'
OUT.mkdir(parents=True,exist_ok=True)

def build(name):
    validation=json.loads((PROJECT/f'artifacts/onnx/{name}.validation.json').read_text())
    converter=[sys.executable,str(SDK/'bin/x86_64-windows-msvc/qairt-converter'),
               '--input_network',str(PROJECT/f'artifacts/onnx/{name}.onnx'),
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
    with (PROJECT/f'evidence/qnn-convert-{name}.log').open('w') as log:
        subprocess.run(converter,stdout=log,stderr=subprocess.STDOUT,check=True)
    config=OUT/f'{name}.htp.json'
    config.write_text(json.dumps({'graphs':[{'graph_names':[name],'vtcm_mb':8,'fp16_relaxed_precision':1}],
                                  'devices':[{'soc_model':43,'dsp_arch':'v73'}]},indent=2))
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
    with (PROJECT/f'evidence/qnn-compile-{name}.log').open('w') as log:
        subprocess.run(compiler,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'COMPILED {name}: {(OUT/f"{name}.bin").stat().st_size} bytes',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    stages=args.stages or ['encoder','dino','dit_prepare']+[f'dit_blocks_{i:02}' for i in range(0,28,4)]+['dit_finish','decoder']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(build,name) for name in stages]
        for future in concurrent.futures.as_completed(futures):
            future.result()
