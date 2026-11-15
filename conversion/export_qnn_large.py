"""Preserve native GroupNorm for QNN; validate hardware against the stable FP32 reference.

The pinned ORT CPU reduction drifts on large native GroupNorm. The general ONNX
reference uses tree reductions, but those prevent HTP GroupNorm fusion and make
compilation allocate excessive memory. Keep the two exports separate and require
actual HTP validation against the original stable reference before packaging.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from pipeline import Encoder,Decoder,load_vae,load_decoder
from large_tile import patch_attention

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False); torch.set_num_threads(8)
    for name,model,src,dst,out_label in [
        ('encoder',Encoder(load_vae()),'rgb','condition','condition'),
        ('decoder',Decoder(load_decoder(),.18215),'restored_latent','pixels','rgb'),
    ]:
        model.eval()
        x=torch.from_numpy(np.load(args.reference/f'{src}.npy'))
        original=model(x)
        reference_path=args.reference/f'{dst}.npy'
        reference=torch.from_numpy(np.load(reference_path)) if reference_path.exists() else original
        if name=='decoder':
            np.save(reference_path,original.numpy().astype(np.float32))
            original.numpy().astype(np.float32).tofile(args.reference/f'{dst}.raw')
        patch_attention(model)
        actual=model(x)
        torch.testing.assert_close(actual,original,rtol=3e-5,atol=1e-4)
        torch.testing.assert_close(actual,reference,rtol=3e-5,atol=1e-4)
        label='rgb' if name=='encoder' else 'latent'
        path=args.output/f'{name}.onnx'
        torch.onnx.export(model,(x,),str(path),input_names=[label],output_names=[out_label],opset_version=17,do_constant_folding=True)
        onnx.checker.check_model(str(path))
        options=ort.SessionOptions(); options.intra_op_num_threads=8
        session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        diagnostic=session.run(None,{label:x.numpy()})[0]
        nrmse=float(np.sqrt(np.mean((diagnostic-reference.numpy())**2))/np.sqrt(np.mean(reference.numpy()**2)))
        record={'name':name,'inputs':{label:list(x.shape)},'outputs':[{'output':out_label,'shape':list(actual.shape)}],
                'torchRewriteValidated':True,'torchReferenceMaxAbs':float((actual-reference).abs().max()),
                'ortCpuDiagnosticNrmse':nrmse,'deviceValidationRequired':True,
                'note':'Native GroupNorm for HTP fusion. ORT CPU diagnostic is not the acceptance reference.',
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (args.output/f'{name}.validation.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)

if __name__=='__main__': main()
