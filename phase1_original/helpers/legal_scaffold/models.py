"""Fresh causal decoder components written for this project; no pretrained code/weights."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    layers: int
    heads: int
    kv_heads: int
    ffn_hidden: int
    context_length: int
    attention: str
    norm_placement: str
    tie_embeddings: bool
    rope_base: float
    norm_epsilon: float
    init_std: float
    mla_latent: int = None
    mla_query_rank: int = None
    mla_rope_dim: int = None

    def validate(self):
        for key in ('vocab_size','d_model','layers','heads','kv_heads','ffn_hidden','context_length'):
            if type(getattr(self,key)) is not int or getattr(self,key)<=0:raise ValueError(key+' must be a positive integer')
        if self.d_model%self.heads or (self.d_model//self.heads)%2:
            raise ValueError('d_model must divide into heads with even RoPE head dimension')
        if self.heads%self.kv_heads or self.kv_heads>self.heads:raise ValueError('kv_heads must divide heads')
        if self.attention not in ('mha','gqa','mla'):raise ValueError('attention must be mha/gqa/mla')
        if self.attention=='mha' and self.kv_heads!=self.heads:raise ValueError('MHA requires one KV head per query head')
        if self.attention=='mla':
            for key in ('mla_latent','mla_query_rank','mla_rope_dim'):
                if type(getattr(self,key)) is not int or getattr(self,key)<=0:raise ValueError('MLA requires explicit positive '+key)
            if self.mla_rope_dim%2:raise ValueError('MLA RoPE dimension must be even')
        if self.norm_placement not in ('pre','post') or type(self.tie_embeddings) is not bool:raise ValueError('explicit pre/post normalization and embedding tying required')
        for key in ('rope_base','norm_epsilon','init_std'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(key+' must be finite positive')
        return self


class RMSNorm(nn.Module):
    def __init__(self,width,epsilon):
        super().__init__();self.weight=nn.Parameter(torch.ones(width));self.epsilon=epsilon

    def forward(self,x):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(dim=-1,keepdim=True)+self.epsilon)
        return normalized.to(x.dtype)*self.weight.to(x.dtype)


def rotary(x,*,start,base):
    """Interleaved even/odd RoPE with explicit absolute offset for cached decoding."""
    width=x.shape[-1]
    frequency=base**(-torch.arange(0,width,2,device=x.device,dtype=torch.float32)/width)
    position=torch.arange(start,start+x.shape[-2],device=x.device,dtype=torch.float32)
    angle=position[:,None]*frequency[None,:]
    cosine=angle.cos().to(x.dtype);sine=angle.sin().to(x.dtype)
    even=x[...,0::2];odd=x[...,1::2]
    return torch.stack((even*cosine-odd*sine,even*sine+odd*cosine),dim=-1).flatten(-2)


def causal_mask(query_length,key_length,start,device):
    q=torch.arange(start,start+query_length,device=device)[:,None]
    k=torch.arange(key_length,device=device)[None,:]
    return k<=q  # SDPA boolean True means allowed, not masked.


class CausalAttention(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;self.head_dim=config.d_model//config.heads
        self.query=nn.Linear(config.d_model,config.d_model,bias=False)
        self.key=nn.Linear(config.d_model,config.kv_heads*self.head_dim,bias=False)
        self.value=nn.Linear(config.d_model,config.kv_heads*self.head_dim,bias=False)
        self.output=nn.Linear(config.d_model,config.d_model,bias=False)

    def forward(self,x,*,past=None,use_cache=False):
        b,t,_=x.shape;c=self.config
        start=0 if past is None else past[0].shape[-2]
        q=self.query(x).view(b,t,c.heads,self.head_dim).transpose(1,2)
        k=self.key(x).view(b,t,c.kv_heads,self.head_dim).transpose(1,2)
        v=self.value(x).view(b,t,c.kv_heads,self.head_dim).transpose(1,2)
        q=rotary(q,start=start,base=c.rope_base);k=rotary(k,start=start,base=c.rope_base)
        if past is not None:
            if past[0].shape[:2]!=(b,c.kv_heads) or past[1].shape!=past[0].shape:
                raise ValueError('incompatible MHA/GQA cache shape')
            k=torch.cat((past[0],k),dim=-2);v=torch.cat((past[1],v),dim=-2)
        cache=(k.detach(),v.detach()) if use_cache else None
        # Portable explicit expansion: compact stored KV cache, but transient
        # training attention repeats heads. This does not claim fused GQA speed.
        groups=c.heads//c.kv_heads
        if groups!=1:k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
        mask=None if start==0 else causal_mask(t,k.shape[-2],start,x.device)
        attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=start==0)
        return self.output(attended.transpose(1,2).contiguous().view(b,t,c.d_model)),cache


class SwiGLU(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.gate=nn.Linear(config.d_model,config.ffn_hidden,bias=False)
        self.up=nn.Linear(config.d_model,config.ffn_hidden,bias=False)
        self.down=nn.Linear(config.ffn_hidden,config.d_model,bias=False)

    def forward(self,x):return self.down(F.silu(self.gate(x))*self.up(x))


class LatentAttention(nn.Module):
    """Dense MLA pilot: decoupled RoPE and shared compressed KV, without MoE.

    Materialized training expands K/V for the entire context. Absorbed cached
    inference stores only latent KV and shared rotated positional keys.
    """
    def __init__(self,config):
        super().__init__();self.config=config;self.content_dim=config.d_model//config.heads
        self.query_down=nn.Linear(config.d_model,config.mla_query_rank,bias=False)
        self.query_norm=RMSNorm(config.mla_query_rank,config.norm_epsilon)
        self.query_up=nn.Linear(config.mla_query_rank,config.heads*(self.content_dim+config.mla_rope_dim),bias=False)
        self.kv_down=nn.Linear(config.d_model,config.mla_latent,bias=False)
        self.kv_norm=RMSNorm(config.mla_latent,config.norm_epsilon)
        self.key_up=nn.Linear(config.mla_latent,config.heads*self.content_dim,bias=False)
        self.value_up=nn.Linear(config.mla_latent,config.heads*self.content_dim,bias=False)
        self.position_key=nn.Linear(config.d_model,config.mla_rope_dim,bias=False)
        self.output=nn.Linear(config.d_model,config.d_model,bias=False)

    def forward(self,x,*,past=None,use_cache=False,mode='materialized'):
        if mode not in ('materialized','absorbed'):raise ValueError('MLA mode must be materialized/absorbed')
        b,t,_=x.shape;c=self.config;d=self.content_dim;r=c.mla_rope_dim
        start=0 if past is None else past[0].shape[-2]
        query=self.query_up(self.query_norm(self.query_down(x))).view(b,t,c.heads,d+r).transpose(1,2)
        q_content=query[...,:d];q_position=rotary(query[...,d:],start=start,base=c.rope_base)
        latent=self.kv_norm(self.kv_down(x))
        k_position=rotary(self.position_key(x).unsqueeze(1),start=start,base=c.rope_base)
        if past is not None:
            if past[0].shape[0]!=b or past[0].shape[-1]!=c.mla_latent or past[1].shape!=(b,1,start,r):
                raise ValueError('incompatible MLA latent/position cache shape')
            latent=torch.cat((past[0],latent),dim=1);k_position=torch.cat((past[1],k_position),dim=-2)
        cache=(latent.detach(),k_position.detach()) if use_cache else None
        length=latent.shape[1];mask=causal_mask(t,length,start,x.device)
        if mode=='materialized':
            k_content=self.key_up(latent).view(b,length,c.heads,d).transpose(1,2)
            value=self.value_up(latent).view(b,length,c.heads,d).transpose(1,2)
            q=torch.cat((q_content,q_position),dim=-1)
            k=torch.cat((k_content,k_position.expand(b,c.heads,length,r)),dim=-1)
            attended=F.scaled_dot_product_attention(q,k,value,attn_mask=mask,dropout_p=0.0)
        else:
            # q^T W_K c = (W_K^T q)^T c: no full-context expanded K/V.
            key_weight=self.key_up.weight.view(c.heads,d,c.mla_latent)
            absorbed_query=torch.einsum('bhtd,hdr->bhtr',q_content,key_weight)
            scores=torch.einsum('bhtr,bsr->bhts',absorbed_query,latent)
            scores=scores+torch.matmul(q_position,k_position.transpose(-1,-2))
            scores=scores/math.sqrt(d+r)
            stable=scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores
            probabilities=stable.masked_fill(~mask,float('-inf')).softmax(dim=-1).to(x.dtype)
            weighted_latent=torch.einsum('bhts,bsr->bhtr',probabilities,latent)
            value_weight=self.value_up.weight.view(c.heads,d,c.mla_latent)
            attended=torch.einsum('bhtr,hdr->bhtd',weighted_latent,value_weight)
        return self.output(attended.transpose(1,2).contiguous().view(b,t,c.d_model)),cache


class DecoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__();self.placement=config.norm_placement
        self.attention_norm=RMSNorm(config.d_model,config.norm_epsilon)
        self.ffn_norm=RMSNorm(config.d_model,config.norm_epsilon)
        self.attention=LatentAttention(config) if config.attention=='mla' else CausalAttention(config)
        self.ffn=SwiGLU(config)

    def forward(self,x,*,past=None,use_cache=False,mla_mode='materialized'):
        kwargs={'past':past,'use_cache':use_cache}
        if isinstance(self.attention,LatentAttention):kwargs['mode']=mla_mode
        if self.placement=='pre':
            delta,cache=self.attention(self.attention_norm(x),**kwargs)
            x=x+delta;x=x+self.ffn(self.ffn_norm(x))
        else:
            delta,cache=self.attention(x,**kwargs)
            x=self.attention_norm(x+delta);x=self.ffn_norm(x+self.ffn(x))
        return x,cache


class CausalDecoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config.validate()
        self.embedding=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm=RMSNorm(config.d_model,config.norm_epsilon)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:self.lm_head.weight=self.embedding.weight

    def _initialize(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):nn.init.normal_(module.weight,mean=0.0,std=self.config.init_std)

    def forward(self,input_ids,*,targets=None,cache=None,use_cache=False,mla_mode=None):
        if input_ids.ndim!=2 or input_ids.dtype!=torch.long or input_ids.shape[1]==0:raise ValueError('input_ids must be nonempty [batch,time] torch.long')
        if cache is not None and not use_cache:raise ValueError('cache requires use_cache=True')
        if use_cache and self.training:raise ValueError('cache is evaluation-only; use model.eval()')
        if cache is not None and len(cache)!=self.config.layers:raise ValueError('one cache entry per layer required')
        starts=[item[0].shape[-2] for item in cache] if cache is not None else [0]*self.config.layers
        if len(set(starts))!=1 or starts[0]+input_ids.shape[1]>self.config.context_length:raise ValueError('cache lengths or context bound invalid')
        x=self.embedding(input_ids);new_cache=[]
        mode=mla_mode or ('absorbed' if use_cache else 'materialized')
        for i,block in enumerate(self.blocks):
            x,item=block(x,past=None if cache is None else cache[i],use_cache=use_cache,mla_mode=mode)
            if use_cache:new_cache.append(item)
        logits=self.lm_head(self.final_norm(x));loss=None
        if targets is not None:
            if targets.shape!=input_ids.shape or targets.dtype!=torch.long:raise ValueError('targets must match input shape/dtype; caller supplies shifted targets')
            if not (targets!=-100).any():raise ValueError('at least one supervised target required')
            loss=F.cross_entropy(logits.float().reshape(-1,self.config.vocab_size),targets.reshape(-1),ignore_index=-100)
        return {'logits':logits,'loss':loss,'cache':new_cache if use_cache else None}

    @torch.inference_mode()
    def greedy(self,prompt,*,new_tokens):
        if type(new_tokens) is not int or new_tokens<0:raise ValueError('nonnegative explicit new_tokens required')
        if prompt.shape[1]+new_tokens>self.config.context_length:raise ValueError('generation exceeds explicit context bound')
        was_training=self.training;self.eval()
        try:
            tokens=prompt;cache=None;current=prompt
            for _ in range(new_tokens):
                result=self(current,cache=cache,use_cache=True)
                current=result['logits'][:,-1].argmax(dim=-1,keepdim=True);cache=result['cache']
                tokens=torch.cat((tokens,current),dim=1)
            return tokens[:,prompt.shape[1]:]  # Continuation token IDs only.
        finally:self.train(was_training)


def parameter_count(model):return sum(parameter.numel() for parameter in model.parameters())


def describe(config):
    # Metadata device counts proposed models without allocating/training weights.
    with torch.device('meta'):model=CausalDecoder(config)
    count=parameter_count(model)
    return {'config':asdict(config),'parameters':count,'course_size_range':50_000_000<=count<=300_000_000}
