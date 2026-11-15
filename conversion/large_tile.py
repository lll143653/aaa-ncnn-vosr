"""Numerically stable reductions and bounded attention for 1024 output tiles."""
import types
import torch
from torch import nn

def tree_mean(x):
    # Bound each FP32 accumulation to 256 elements. Flat reductions over
    # millions of pixels drift on the pinned ONNX Runtime CPU implementation.
    while x.shape[-1]>256:
        assert x.shape[-1]%256==0
        x=x.reshape(*x.shape[:-1],-1,256).mean(-1)
    return x.mean(-1,keepdim=True)

def stable_group_norm(self,x):
    grouped=x.reshape(x.shape[0],self.num_groups,-1)
    centered=grouped-tree_mean(grouped)
    normalized=centered*torch.rsqrt(tree_mean(centered*centered)+self.eps)
    affine=[1,-1]+[1]*(x.ndim-2)
    return normalized.reshape_as(x)*self.weight.reshape(affine)+self.bias.reshape(affine)

def patch_group_norm(module):
    for layer in module.modules():
        if isinstance(layer,nn.GroupNorm): layer.forward=types.MethodType(stable_group_norm,layer)
    return module

def attend(q,k,v,scale,chunk=256):
    key=k.transpose(-2,-1)
    return torch.cat([((q[...,i:i+chunk,:]*scale)@key).softmax(-1)@v
                      for i in range(0,q.shape[-2],chunk)],dim=-2)

class ChunkedVaeAttention:
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,*args,**kwargs):
        assert encoder_hidden_states is None and attention_mask is None and attn.spatial_norm is None
        residual=hidden_states
        b,c,h,w=hidden_states.shape
        x=hidden_states.reshape(b,c,h*w).transpose(1,2)
        if attn.group_norm is not None: x=attn.group_norm(x.transpose(1,2)).transpose(1,2)
        q=attn.head_to_batch_dim(attn.to_q(x)); k=attn.head_to_batch_dim(attn.to_k(x)); v=attn.head_to_batch_dim(attn.to_v(x))
        y=attn.batch_to_head_dim(attend(q,k,v,attn.scale))
        y=attn.to_out[1](attn.to_out[0](y)).transpose(1,2).reshape(b,c,h,w)
        if attn.residual_connection: y=y+residual
        return y/attn.rescale_output_factor

def self_attention(self,x,rope=None):
    b,n,c=x.shape
    q,k,v=self.qkv(x).chunk(3,dim=-1)
    q=q.reshape(b,n,self.num_heads,self.head_dim).transpose(1,2)
    k=k.reshape(b,n,self.num_heads,self.head_dim).transpose(1,2)
    v=v.reshape(b,n,self.num_heads,self.head_dim).transpose(1,2)
    q,k=self.q_norm(q),self.k_norm(k)
    if rope is not None: q,k=rope(q),rope(k)
    return self.proj_drop(self.proj(attend(q,k,v,self.scale).transpose(1,2).reshape(b,n,c)))

def cross_attention(self,x,cond,mask=None):
    assert mask is None
    b,n,c=x.shape; bn,nk,_=cond.shape
    q=self.q_linear(x).reshape(b,n,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_linear(cond).reshape(bn,nk,self.num_heads,self.head_dim).transpose(1,2)
    v=self.v_linear(cond).reshape(bn,nk,self.num_heads,self.head_dim).transpose(1,2)
    q,k=self.q_norm(q),self.k_norm(k)
    return self.proj_drop(self.proj(attend(q,k,v,self.scale).transpose(1,2).reshape(b,n,c)))

def patch_attention(module):
    for layer in module.modules():
        if hasattr(layer,'set_processor') and hasattr(layer,'to_q'):
            layer.set_processor(ChunkedVaeAttention())
        elif hasattr(layer,'qkv') and hasattr(layer,'num_heads'):
            layer.forward=types.MethodType(self_attention,layer)
        elif hasattr(layer,'q_linear') and hasattr(layer,'k_linear'):
            layer.forward=types.MethodType(cross_attention,layer)
    return module
