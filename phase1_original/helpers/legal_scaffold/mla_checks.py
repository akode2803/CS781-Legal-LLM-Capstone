"""Tiny synthetic MLA algebra/cache checks and actual GPU dispatch probe."""
from dataclasses import replace
import contextlib,json
import torch
from .models import ModelConfig,CausalDecoder,LatentAttention
from .long_training import DEFAULT_MODEL
from .kernel_dispatch import classify

MLA_CONFIG=dict(DEFAULT_MODEL,attention='mla',mla_latent=256,mla_query_rank=256,mla_rope_dim=32)

def tiny_checks():
    config=ModelConfig(**dict(MLA_CONFIG,vocab_size=31,d_model=32,layers=2,heads=4,kv_heads=2,ffn_hidden=48,context_length=32,mla_latent=12,mla_query_rank=16,mla_rope_dim=4))
    torch.manual_seed(12);model=CausalDecoder(config).double().eval();inputs=torch.tensor([[1,2,3,4,5,6]])
    materialized=model(inputs,mla_mode='materialized')['logits']
    absorbed=model(inputs,mla_mode='absorbed')['logits']
    torch.testing.assert_close(materialized,absorbed,atol=1e-9,rtol=1e-8)
    model(inputs[:,:-1],targets=inputs[:,1:],mla_mode='materialized')['loss'].backward()
    gradients={name:p.grad.clone() for name,p in model.named_parameters()}
    model.zero_grad();model(inputs[:,:-1],targets=inputs[:,1:],mla_mode='absorbed')['loss'].backward()
    for name,p in model.named_parameters():
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(gradients[name],p.grad,atol=2e-7,rtol=2e-6)
    changed=inputs.clone();changed[:,3:]=(changed[:,3:]+1)%31
    torch.testing.assert_close(model(changed)['logits'][:,:3],materialized[:,:3],atol=0,rtol=0)
    with torch.no_grad():
        for mode in ('materialized','absorbed'):
            cache=None;pieces=[]
            for start,stop in ((0,2),(2,3),(3,6)):
                result=model(inputs[:,start:stop],cache=cache,use_cache=True,mla_mode=mode)
                cache=result['cache'];pieces.append(result['logits'])
            torch.testing.assert_close(torch.cat(pieces,dim=1),materialized,atol=1e-8,rtol=1e-7)
            assert cache[0][0].shape==(1,6,12) and cache[0][1].shape==(1,1,6,4)
    return {'forward_backward':'passed','materialized_vs_absorbed_outputs_and_gradients':'passed',
            'causality':'passed','token_and_chunk_cache_both_modes':'passed','scope':'tiny synthetic CPU fixture; not model quality or GPU speed'}

def gpu_dispatch():
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback.'
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from torch.profiler import profile,ProfilerActivity
    native=torch.cuda.is_bf16_supported(including_emulation=False)
    dtype=torch.bfloat16 if native else torch.float16
    model=LatentAttention(ModelConfig(**MLA_CONFIG)).cuda()
    x=torch.randn(2,1024,512,device='cuda',requires_grad=True)
    def step(backend=None):
        model.zero_grad(set_to_none=True);x.grad=None
        with (sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()),torch.autocast('cuda',dtype=dtype):
            out,_=model(x,mode='materialized');loss=out.float().square().mean()
        loss.backward()
        return out.detach().float().clone(),[x.grad.detach().float().clone()]+[p.grad.detach().float().clone() for p in model.parameters()]
    reference=step(SDPBackend.MATH)
    for _ in range(2):step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        found=step();torch.cuda.synchronize()
    atol,rtol=(.02,.08) if native else (.003,.02)
    def compare(result):
        for a,b in zip([result[0]]+result[1],[reference[0]]+reference[1]):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    compare(found)
    forced=[]
    for name in ('MATH','FLASH_ATTENTION','EFFICIENT_ATTENTION','CUDNN_ATTENTION'):
        if not hasattr(SDPBackend,name):forced.append({'backend':name,'status':'enum_unavailable'});continue
        try:compare(step(getattr(SDPBackend,name)))
        except RuntimeError as error:forced.append({'backend':name,'status':'unsupported','reason':str(error)});continue
        forced.append({'backend':name,'status':'passed_output_and_gradients'})
    operators=[e.key for e in p.key_averages() if 'scaled_dot_product' in e.key]
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
        original,_=model(x);changed=x.detach().clone();changed[:,512:]+=3
        perturbed,_=model(changed)
        torch.testing.assert_close(original[:,:512],perturbed[:,:512],atol=atol,rtol=rtol)
        prefix,cache=model(x[:,:4],use_cache=True)
        materialized,_=model(x[:,4:5],past=cache,use_cache=True,mode='materialized')
        absorbed,_=model(x[:,4:5],past=cache,use_cache=True,mode='absorbed')
        torch.testing.assert_close(materialized,absorbed,atol=atol,rtol=rtol)
    return {'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
       'native_bf16':native,'selected_dtype':str(dtype),'trainer_fp16_grad_scaler_policy':not native,
       'attention':'dense MLA materialized training','batch':2,'sequence':1024,'heads':8,'qk_dimension':96,'value_dimension':64,
       'mask':'explicit causal boolean mask; no shape padding or model math changes',
       'actual_sdpa_operators':operators,'dispatch':classify(operators),'forced_kernels':forced,
       'default_vs_math_outputs_and_gradients':'passed','causality':'passed','one_token_cached_materialized_vs_absorbed':'passed',
       'inference_limitation':'absorbed decoding uses explicit score/softmax/einsum math, not fused SDPA; compact stored cache does not imply faster inference',
       'scope':'synthetic attention; no optimizer update or speedup claim'}
