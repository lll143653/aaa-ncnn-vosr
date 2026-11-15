"""Export the SAME official 0.5B weights as the mobile QNN backend to ncnn.

All rewrites are compared with the original PyTorch boundary before conversion.
Avoid five/six dimensional reshapes, which pnnx cannot lower to ncnn.
"""
import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pnnx

from pipeline import (PROJECT, Encoder, Decoder, Dino, DitPrepare, DitBlocks,
                      DitFinish, load_vae, load_decoder, load_dino, load_dit)


def attention_forward(self, x, rope=None):
    b, n, c = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    q, k = self.q_norm(q), self.k_norm(k)
    if rope is not None:
        q, k = rope(q), rope(k)
    y = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1) @ v
    return self.proj_drop(self.proj(y.transpose(1, 2).reshape(b, n, c)))


class Rope(nn.Module):
    def __init__(self, original):
        super().__init__()
        closure = dict(zip(original.__code__.co_freevars,
                           [x.cell_contents for x in original.__closure__]))
        self.register_buffer('cos', closure['freqs_cos'].clone())
        self.register_buffer('sin', closure['freqs_sin'].clone())
        dim = self.cos.shape[-1]
        rotate = torch.zeros(dim, dim)
        for i in range(0, dim, 2):
            rotate[i + 1, i] = -1
            rotate[i, i + 1] = 1
        self.register_buffer('rotate', rotate)

    def forward(self, x):
        return x * self.cos + (x @ self.rotate) * self.sin


class Finish(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.final = original.final
        self.grid = original.grid
        self.register_buffer('c', original.c.clone())
        order=[(py * 2 + px) * 4 + c for c in range(4) for py in range(2) for px in range(2)]
        self.register_buffer('reorder', torch.eye(16)[:,order])

    def forward(self, tokens, noise):
        x = self.final(tokens, self.c)
        x = (x @ self.reorder).transpose(1, 2).reshape(1, 16, self.grid, self.grid)
        return noise - F.pixel_shuffle(x, 2)


def patch_attention(module):
    for m in module.modules():
        if hasattr(m, 'qkv') and hasattr(m, 'num_heads'):
            if not hasattr(m, 'head_dim'):
                m.head_dim = m.qkv.in_features // m.num_heads
            if not hasattr(m, 'q_norm'):
                m.q_norm, m.k_norm = nn.Identity(), nn.Identity()
            m.forward = types.MethodType(attention_forward, m)


def fp32_rms_forward(self, x):
    return self._norm(x) * self.weight


def attention_scores(self, query, key, attention_mask=None):
    scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if attention_mask is not None:
        scores = scores + attention_mask
    return scores.softmax(-1)


def patch_fp32(module):
    # All graph IO and computation are deliberately FP32. Eliminate redundant
    # aten::to dtype conversions and baddbmm, unsupported in the ncnn serializer.
    for m in module.modules():
        if m.__class__.__name__=='RMSNorm' and hasattr(m,'_norm'):
            m.forward=types.MethodType(fp32_rms_forward,m)
        if hasattr(m,'get_attention_scores') and hasattr(m,'to_q'):
            m.get_attention_scores=types.MethodType(attention_scores,m)


def convert(name, module, inputs, output, rewrite=None):
    module.eval()
    with torch.no_grad():
        expected = module(*inputs)
        if rewrite:
            module = rewrite(module).eval()
        if (inputs[0].ndim==4 and inputs[0].shape[-1]>=1024) or (name.startswith('dit_blocks') and inputs[0].shape[1]>1024) or (name=='decoder' and inputs[0].shape[-1]>=128):
            from large_tile import patch_attention as patch_large_attention
            patch_large_attention(module)
        patch_fp32(module)
        actual = module(*inputs)
        exps = expected if isinstance(expected, tuple) else (expected,)
        acts = actual if isinstance(actual, tuple) else (actual,)
        diffs = []
        for a, b in zip(acts, exps):
            # DINO uses fused SDPA in the reference and explicit matmul here.
            torch.testing.assert_close(a, b, rtol=3e-5, atol=1e-4 if name=='dino' else 3e-5)
            diffs.append(float((a-b).abs().max()))
        traced = torch.jit.trace(module, inputs, check_trace=False)
        path = output / f'{name}.pt'
        traced.save(str(path))
    shapes = ','.join('['+','.join(map(str, x.shape))+']' for x in inputs)
    log = output / f'{name}.pnnx.log'
    with log.open('w') as stream:
        result = subprocess.run([pnnx.EXEC_PATH, str(path), f'inputshape={shapes}', 'fp16=1'],
                                cwd=output, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(errors='replace')
    param = output / f'{name}.ncnn.param'
    binary = output / f'{name}.ncnn.bin'
    if result.returncode or not param.is_file() or not binary.is_file():
        raise RuntimeError(f'pnnx failed for {name}: {text[-2000:]}')
    unsupported = [s for s in text.splitlines() if 'ignore Tensor.' in s or 'not supported' in s or 'physical rank' in s]
    if unsupported:
        raise RuntimeError(f'Unsupported graph {name}: {unsupported[:10]}')
    custom=[s.split()[0] for s in param.read_text().splitlines()[2:] if s.strip() and ('.' in s.split()[0] or '::' in s.split()[0])]
    if custom:
        raise RuntimeError(f'Unlowered ncnn layers in {name}: {custom}')
    record = {'name': name, 'pnnxVersion':pnnx.__version__, 'rewriteMaxAbs': diffs,
              'inputs': [list(x.shape) for x in inputs],
              'outputs': [list(x.shape) for x in acts],
              'files': [{'file': f.name, 'bytes': f.stat().st_size,
                         'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in (param, binary)]}
    (output / f'{name}.validation.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return acts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'artifacts/ncnn-0.5b')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--components', nargs='+', default=['encoder','dino','dit','decoder'])
    parser.add_argument('--from-traces', action='store_true', help='Re-lower existing validated FP32 traces with the pinned pnnx')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.from_traces:
        names=['encoder','dino','dit_prepare']+[f'dit_blocks_{i:02}' for i in range(0,28,4)]+['dit_finish','decoder']
        for name in names:
            metadata=out/f'{name}.validation.json'
            data=json.loads(metadata.read_text())
            shapes=','.join('['+','.join(map(str,s))+']' for s in data['inputs'])
            with (out/f'{name}.pnnx.log').open('w') as stream:
                subprocess.run([pnnx.EXEC_PATH,str(out/f'{name}.pt'),f'inputshape={shapes}','fp16=1'],cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)
            param=out/f'{name}.ncnn.param'
            bad=[s.split()[0] for s in param.read_text().splitlines()[2:] if s.strip() and ('.' in s.split()[0] or '::' in s.split()[0])]
            if bad: raise RuntimeError((name,bad))
            data['files']=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [param,out/f'{name}.ncnn.bin']]
            data['pnnxVersion']=pnnx.__version__
            metadata.write_text(json.dumps(data,indent=2))
            print('LOWERED '+name,flush=True)
        return
    torch.set_grad_enabled(False); torch.set_num_threads(8)
    get = lambda name: torch.from_numpy(np.load(args.reference/f'{name}.npy'))
    rgb, noise = get('rgb'), get('noise')
    if 'encoder' in args.components:
        convert('encoder', Encoder(load_vae()), (rgb,), out)
        gc.collect()
    if 'dino' in args.components:
        def rewrite_dino(m):
            patch_attention(m)
            return m
        convert('dino', Dino(load_dino()), (get('dino_input'),), out, rewrite_dino)
        gc.collect()
    if 'decoder' in args.components:
        convert('decoder', Decoder(load_decoder(), .18215), (get('restored_latent'),), out)
        gc.collect()
    if 'dit' in args.components:
        model, _ = load_dit()
        tokens, context = convert('dit_prepare', DitPrepare(model),
                                 (get('condition'), noise, get('features')), out)
        for first in range(0, 28, 4):
            def rewrite_blocks(m):
                m.rope = Rope(m.rope)
                patch_attention(m)
                return m
            (tokens,) = convert(f'dit_blocks_{first:02}', DitBlocks(model, first, 4, args.size),
                                (tokens, context), out, rewrite_blocks)
        convert('dit_finish', DitFinish(model, args.size), (tokens, noise), out, Finish)


if __name__ == '__main__':
    main()
