"""Export and numerically validate actual pretrained components, no dummy weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import onnx
import onnxruntime as ort
from pipeline import (PROJECT, SOURCE, WEIGHTS, Encoder, Decoder, Dino, DitPrepare,
                      DitBlocks, DitFinish, load_vae, load_decoder, load_dino, load_dit,
                      preprocess_dino)

OUT = PROJECT/'artifacts/onnx'
DATA = PROJECT/'artifacts/reference'
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
torch.set_grad_enabled(False)
INPUT_SCALE=4

def save_tensor(name, value):
    array = value.detach().cpu().numpy().astype(np.float32)
    np.save(DATA/f'{name}.npy', array)
    array.tofile(DATA/f'{name}.raw')

def validate_export(name, model, inputs, input_names, output_names, reference=None):
    model.eval()
    before = time.perf_counter()
    expected = model(*inputs) if reference is None else reference
    if name.startswith('dit_blocks') and inputs[0].shape[1]>1024:
        from large_tile import patch_attention
        patch_attention(model)
        torch.testing.assert_close(model(*inputs),expected,rtol=3e-5,atol=1e-4)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    path = OUT/f'{name}.onnx'
    torch.onnx.export(model, inputs, str(path), opset_version=17,
                      input_names=input_names, output_names=output_names,
                      do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    feed = {n:x.numpy() for n,x in zip(input_names,inputs)}
    actual = session.run(None, feed)
    metrics = []
    for label, a, b in zip(output_names, actual, expected):
        b = b.numpy()
        rmse = float(np.sqrt(np.mean((a-b)**2)))
        nrmse = rmse/max(float(np.sqrt(np.mean(b**2))),1e-6)
        max_abs = float(np.max(np.abs(a-b)))
        if not np.isfinite(a).all() or nrmse > 1e-3:
            raise AssertionError(f'{name}/{label}: NRMSE {nrmse}, max {max_abs}')
        metrics.append({'output':label,'shape':list(a.shape),'max_abs':max_abs,'nrmse':nrmse})
    result = {'name':name,'inputs':{n:list(x.shape) for n,x in zip(input_names,inputs)},
              'outputs':metrics,'elapsed_s':time.perf_counter()-before,
              'bytes':path.stat().st_size,
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/f'{name}.validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    del session
    gc.collect()
    return tuple(torch.from_numpy(a) for a in actual)

def input_rgb(size):
    # A reproducible crop of the image shipped by the official project.
    image = Image.open(SOURCE/'preset/datasets/inp_data/Nikon_024_LR4.png').convert('RGB')
    side = size//INPUT_SCALE
    left, top = max(0,(image.width-side)//2), max(0,(image.height-side)//2)
    image = image.crop((left,top,left+side,top+side))
    image.save(DATA/'input.png')
    up = image.resize((size,size),Image.Resampling.BICUBIC)
    up.save(DATA/'bicubic.png')
    value = torch.from_numpy(np.array(up).copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1
    save_tensor('rgb',value)
    return value

def main():
    global OUT,DATA,INPUT_SCALE
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--components',nargs='+',default=['encoder','dino','decoder','dit'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--reference',type=Path,default=DATA)
    parser.add_argument('--input-scale',type=int,choices=[2,4],default=4)
    args=parser.parse_args()
    OUT=args.output.resolve(); DATA=args.reference.resolve(); INPUT_SCALE=args.input_scale
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    if args.size%16:
        raise ValueError('Size must be a multiple of 16')
    rgb=input_rgb(args.size)
    latent_size=args.size//8
    generator=torch.Generator().manual_seed(42)
    noise=torch.randn((1,4,latent_size,latent_size),generator=generator)
    save_tensor('noise',noise)
    if 'encoder' in args.components:
        vae=load_vae()
        wrapper=Encoder(vae)
        direct=vae.encode(rgb).latent_dist.mode()*vae.config.scaling_factor
        torch.testing.assert_close(wrapper(rgb),direct)
        if args.size>=1024:
            from large_tile import patch_group_norm,patch_attention
            patch_attention(patch_group_norm(wrapper))
            torch.testing.assert_close(wrapper(rgb),direct,rtol=3e-5,atol=1e-4)
        (condition,)=validate_export('encoder',wrapper,(rgb,),['rgb'],['condition'],reference=direct)
        save_tensor('condition',condition)
        del vae,wrapper
        gc.collect()
    if 'dino' in args.components:
        dino=load_dino()
        value=preprocess_dino(rgb)
        save_tensor('dino_input',value)
        wrapper=Dino(dino)
        direct=dino.get_intermediate_layers(value,n=[8],norm=False)[0]
        torch.testing.assert_close(wrapper(value),direct)
        (features,)=validate_export('dino',wrapper,(value,),['image'],['features'])
        save_tensor('features',features)
        del dino,wrapper
        gc.collect()
    if 'decoder' in args.components:
        decoder=Decoder(load_decoder(),0.18215)
        reference=decoder(noise)
        if args.size>=1024:
            from large_tile import patch_group_norm,patch_attention
            patch_attention(patch_group_norm(decoder))
            torch.testing.assert_close(decoder(noise),reference,rtol=3e-5,atol=1e-4)
        validate_export('decoder',decoder,(noise,),['latent'],['rgb'],reference=reference)
        del decoder
        gc.collect()
    if 'dit' in args.components:
        condition=torch.from_numpy(np.load(DATA/'condition.npy'))
        features=torch.from_numpy(np.load(DATA/'features.npy'))
        model,config=load_dit()
        # Compute an independent, unsplit official forward for comparison.
        expected=noise-model.forward_flexible(torch.cat((condition,noise),1),
                    torch.ones(1),torch.zeros(1),[features])
        prepare=DitPrepare(model)
        tokens,z=validate_export('dit_prepare',prepare,(condition,noise,features),
                            ['condition','noise','features'],['tokens','context'])
        save_tensor('dit_context',z)
        for first in range(0,28,4):
            part=DitBlocks(model,first,4,args.size)
            (tokens,)=validate_export(f'dit_blocks_{first:02}',part,(tokens,z),
                                       ['tokens','context'],['next_tokens'])
        (latent,)=validate_export('dit_finish',DitFinish(model,args.size),(tokens,noise),
                                  ['tokens','noise'],['latent'])
        torch.testing.assert_close(latent,expected,atol=0.003,rtol=0.001)
        save_tensor('restored_latent',latent)
        del model,prepare,part
        gc.collect()
        decoder=Decoder(load_decoder(),0.18215)
        restored=decoder(latent)
        save_tensor('pixels',restored)
        pixels=(restored[0].clamp(-1,1)+1)*127.5
        Image.fromarray(pixels.permute(1,2,0).numpy().clip(0,255).astype('uint8')).save(DATA/'restored.png')
        (DATA/'pipeline.json').write_text(json.dumps({
            'model':'VOSR_0.5B_os','output_size':args.size,'upscale':INPUT_SCALE,'seed':42,
            'precision':'float32','sampling_steps':1,'posterior':'mode',
            'color_fix':'none','split_vs_official_max_abs':float((latent-expected).abs().max())},indent=2))

if __name__=='__main__':
    main()
