"""Synthetic actual-model attention dispatch, causality and gradient checks."""
import json
import contextlib
import torch
from torch.profiler import profile,ProfilerActivity
from .attention_preflight import run_preflight
from .models import CausalAttention,ModelConfig
from .long_training import DEFAULT_MODEL

def classify(operators):
    concrete=[name for name in operators if any(term in name for term in ('_scaled_dot_product_flash_attention','_scaled_dot_product_efficient_attention','_scaled_dot_product_cudnn_attention','_scaled_dot_product_attention_math'))]
    return {'status':'observed' if concrete else 'inconclusive','concrete_operators':concrete}

def run_dispatch():
    assert torch.cuda.is_available(), 'A CUDA GPU is required; no CPU fallback.'
    from torch.nn.attention import sdpa_kernel,SDPBackend
    native=torch.cuda.is_bf16_supported(including_emulation=False)
    dtype=torch.bfloat16 if native else torch.float16
    report={'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
            'native_bf16':native,'selected_dtype':str(dtype),'trainer_fp16_grad_scaler_policy':dtype==torch.float16,
            'scope':'synthetic actual attention only; no optimizer update, no speed claim',
            'forced_generic_fixture_checks':run_preflight('cuda'),'actual_model_checks':[]}
    for kv_heads in (2,8):
        config=ModelConfig(**dict(DEFAULT_MODEL,kv_heads=kv_heads,attention='gqa' if kv_heads==2 else 'mha'))
        model=CausalAttention(config).cuda()
        x=torch.randn(2,1024,512,device='cuda',requires_grad=True)
        def step(backend=None):
            model.zero_grad(set_to_none=True);x.grad=None
            with (sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()),torch.autocast('cuda',dtype=dtype):
                out,_=model(x);loss=out.float().square().mean()
            loss.backward()
            return out.detach().float().clone(),[x.grad.detach().float().clone()]+[p.grad.detach().float().clone() for p in model.parameters()]
        reference=step(SDPBackend.MATH)
        for _ in range(2):step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
            found=step();torch.cuda.synchronize()
        atol,rtol=(.02,.08) if dtype==torch.bfloat16 else (.003,.02)
        def compare(result):
            for a,b in zip([result[0]]+result[1],[reference[0]]+reference[1]):
                assert torch.isfinite(a).all(), 'Nonfinite attention output/gradient.'
                torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        compare(found)
        with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
            original,_=model(x);changed=x.detach().clone();changed[:,512:]+=3
            perturbed,_=model(changed)
            torch.testing.assert_close(original[:,:512],perturbed[:,:512],atol=atol,rtol=rtol)
        forced=[]
        for name in ('MATH','FLASH_ATTENTION','EFFICIENT_ATTENTION','CUDNN_ATTENTION'):
            if not hasattr(SDPBackend,name):forced.append({'backend':name,'status':'enum_unavailable'});continue
            try:compare(step(getattr(SDPBackend,name)))
            except RuntimeError as error:forced.append({'backend':name,'status':'unsupported','reason':str(error)});continue
            forced.append({'backend':name,'status':'passed_output_and_gradients'})
        operators=[e.key for e in p.key_averages() if 'scaled_dot_product' in e.key]
        report['actual_model_checks'].append({'attention':config.attention,'batch':2,'sequence':1024,'width':512,
             'query_heads':8,'kv_heads':kv_heads,'head_dim':64,'explicit_kv_expansion':kv_heads!=8,
             'actual_sdpa_operators':operators,'dispatch':classify(operators),'causality':'passed',
             'default_vs_math_output_and_gradients':'passed','forced_kernels':forced})
    return report

if __name__=='__main__':print(json.dumps(run_dispatch(),indent=2))
