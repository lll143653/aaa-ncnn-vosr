"""Official VOSR 0.5B one-step pipeline and fixed-shape export boundaries."""
from pathlib import Path
import json
import os
import sys
import types

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['XFORMERS_DISABLED'] = '1'
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = Path(os.environ.get('VOSR_SOURCE', str(PROJECT / 'third_party/VOSR-main')))
WEIGHTS = Path(os.environ.get('VOSR_WEIGHTS', str(ROOT / 'models/vosr')))
sys.path.insert(0, str(SOURCE))

def load_dit():
    import models.lightningdit as source
    # torch.compile is an optional optimization, not part of the model math.
    # Unwrap it for the legacy ONNX tracer on Windows.
    for name, obj in list(vars(source).items()):
        if hasattr(obj, '_torchdynamo_orig_callable'):
            setattr(source, name, obj._torchdynamo_orig_callable)
        elif isinstance(obj, type) and obj.__module__ == source.__name__:
            for attr, method in list(vars(obj).items()):
                if hasattr(method, '_torchdynamo_orig_callable'):
                    setattr(obj, attr, method._torchdynamo_orig_callable)
    args = json.loads((WEIGHTS/'VOSR_0.5B_os/args.json').read_text())
    model = source.LightningDiT(
        input_size=args['resolution']//8, patch_size=args['patch_size'],
        in_channels=8, out_channels=4, hidden_size=args['dim'], depth=args['depth'],
        num_heads=args['num_heads'], mlp_ratio=args['mlp_ratio'], z_dims=args['enc_dim'],
        encdim_ratio=args['encdim_ratio'], auxiliary_time_cond=True,
        use_qknorm=args['use_qknorm'], use_swiglu=args['use_swiglu'],
        use_rope=args['use_rope'], use_rmsnorm=args['use_rmsnorm'], wo_shift=args['wo_shift'],
        num_fused_layers=len(args['layer_dinov2b_list']))
    state = load_file(str(WEIGHTS/'VOSR_0.5B_os/checkpoints/ema_model.safetensors'))
    model.load_state_dict(state, strict=True)
    model.disable_fused_attn()
    return model.eval(), args

def load_vae():
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS/'stable-diffusion-2-1-base'),
                                        subfolder='vae', local_files_only=True)
    vae.set_attn_processor(AttnProcessor())
    return vae.eval()

def load_decoder():
    from models.light_decoder import LightDecoder
    checkpoint = torch.load(WEIGHTS/'sd21_lwdecoder.pth', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = LightDecoder(in_channels=config['in_channels'], out_channels=config['out_channels'],
                         block_out_channels=tuple(config['block_out_channels']),
                         layers_per_block=config['layers_per_block'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    for module in model.modules():
        if hasattr(module, 'set_processor'):
            module.set_processor(AttnProcessor())
    return model.eval()

def load_dino():
    model = torch.hub.load(str(WEIGHTS/'torch_cache/facebookresearch_dinov2_main'),
                          'dinov2_vitb14', source='local', pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS/'torch_cache/checkpoints/dinov2_vitb14_pretrain.pth',
                                    map_location='cpu', weights_only=True), strict=True)
    return model.eval()

class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder, self.quant_conv = vae.encoder, vae.quant_conv
        self.scale = vae.config.scaling_factor
    def forward(self, rgb):
        return self.quant_conv(self.encoder(rgb))[:, :4] * self.scale

class Decoder(nn.Module):
    def __init__(self, decoder, scale):
        super().__init__()
        self.decoder, self.scale = decoder, scale
    def forward(self, latent):
        return self.decoder(latent / self.scale).clamp(-1, 1)

class Dino(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.blocks = nn.ModuleList(list(model.blocks[:9]))
        with torch.no_grad():
            position = model.interpolate_pos_encoding(torch.zeros(1, 1025, 768), 448, 448)
        self.register_buffer('position', position)
    def forward(self, normalized):
        x = self.patch_embed(normalized)
        x = torch.cat((self.cls_token.expand(1, -1, -1), x), dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return x[:, 1:]

class DitPrepare(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.x_embedder
        self.norm, self.mlp = model.layer_norm, model.mlp_ca
    def forward(self, condition, noise, features):
        return self.embed(torch.cat((condition, noise), 1)), self.mlp(self.norm(features))

class DitBlocks(nn.Module):
    def __init__(self, model, first, count, output_size):
        super().__init__()
        self.blocks = nn.ModuleList(list(model.blocks[first:first+count]))
        with torch.no_grad():
            c = model.t_embedder(torch.ones(1)) + model.r_embedder(torch.zeros(1))
            self.register_buffer('c0', model.t_block(c))
        # The official flexible path uses train-grid-scaled RoPE at smaller tiles.
        self.rope = model._get_dynamic_rope(output_size//16, 'cpu', torch.float32)
    def forward(self, tokens, features):
        for block in self.blocks:
            tokens = block(tokens, self.c0, features, self.rope)
        return tokens

class DitFinish(nn.Module):
    def __init__(self, model, output_size):
        super().__init__()
        self.final = model.final_layer
        self.grid = output_size // 16
        with torch.no_grad():
            c = model.t_embedder(torch.ones(1)) + model.r_embedder(torch.zeros(1))
            self.register_buffer('c', c)
    def forward(self, tokens, noise):
        x = self.final(tokens, self.c).reshape(1, self.grid, self.grid, 2, 2, 4)
        velocity = x.permute(0, 5, 1, 3, 2, 4).reshape(1, 4, self.grid*2, self.grid*2)
        return noise - velocity

def preprocess_dino(rgb):
    unit = F.interpolate(rgb*0.5+0.5, (448,448), mode='bicubic', align_corners=False).clamp(0,1)
    mean = torch.tensor([0.485,0.456,0.406]).reshape(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).reshape(1,3,1,1)
    return (unit-mean)/std
