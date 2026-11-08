"""Optional synthetic SDPA hardware check. No model, optimizer, or installation.

Run in a separate process with an already installed PyTorch runtime.
"""
import argparse
import importlib
import json
import math
import platform
import time
import warnings

SHAPE = (1, 2, 64, 32)  # B, heads, sequence, head dimension; fixture, not model choices
WARMUP = 2
STEPS = 5
TOLERANCES = {'float32': {'atol': 1e-5, 'rtol': 1e-4},
              'float16': {'atol': 3e-3, 'rtol': 2e-2},
              'bfloat16': {'atol': 2e-2, 'rtol': 8e-2}}
BACKENDS = ('MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION')


def backend_plan(enum, cuda):
    """Pure capability inventory: do not infer runtime support from GPU names."""
    return [{'backend': name, 'eligible': hasattr(enum, name) and (cuda or name == 'MATH'),
             'reason': ('enum unavailable' if not hasattr(enum, name) else
                        'CUDA-only probe omitted on CPU' if not cuda and name != 'MATH' else None)}
            for name in BACKENDS]


def _reference(torch, inputs, upstream):
    # A tiny mathematical oracle, independent of SDPA dispatch, on CPU float64.
    q, k, v = [x.detach().to(device='cpu', dtype=torch.float64).requires_grad_(True) for x in inputs]
    scores = q @ k.transpose(-2, -1) / math.sqrt(SHAPE[-1])
    mask = torch.ones(SHAPE[-2], SHAPE[-2], dtype=torch.bool).triu(1)
    output = scores.masked_fill(mask, float('-inf')).softmax(-1) @ v
    output.backward(upstream.detach().to(device='cpu', dtype=torch.float64))
    return [output.detach()] + [x.grad.detach() for x in (q, k, v)]


def _probe(torch, functional, sdpa_kernel, backend, dtype_name, device):
    dtype = getattr(torch, dtype_name)
    # Local CPU generator avoids modifying caller/global RNG states.
    generator = torch.Generator(device='cpu').manual_seed(781)
    inputs = [torch.randn(SHAPE, generator=generator).to(device=device, dtype=dtype).requires_grad_(True)
              for _ in range(3)]
    upstream = (torch.randn(SHAPE, generator=generator) * .1).to(device=device, dtype=dtype)
    reference = _reference(torch, inputs, upstream)
    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()
    def step():
        for x in inputs:
            x.grad = None
        result = functional.scaled_dot_product_attention(*inputs, dropout_p=0.0, is_causal=True)
        result.backward(upstream)
        return result
    with sdpa_kernel([backend]):  # Exactly one backend; automatic fallback disabled.
        output = step()
        synchronize()
        actual = [output.detach().to(device='cpu', dtype=torch.float64)] + [
            x.grad.detach().to(device='cpu', dtype=torch.float64) for x in inputs]
        errors = []
        passed = True
        for name, found, expected in zip(('output', 'dq', 'dk', 'dv'), actual, reference):
            finite = bool(torch.isfinite(found).all()) and bool(torch.isfinite(expected).all())
            close = finite and bool(torch.allclose(found, expected, **TOLERANCES[dtype_name]))
            passed = passed and close
            errors.append({'tensor': name, 'finite': finite, 'within_tolerance': close,
                           'max_abs_error': float((found - expected).abs().max()) if finite else None})
        del output, actual, reference
        if not passed:
            return {'status': 'numerical_mismatch', 'comparison': errors, 'tolerance': TOLERANCES[dtype_name]}
        for _ in range(WARMUP):
            step()
        synchronize()
        baseline = None
        if device == 'cuda':
            for x in inputs:
                x.grad = None
            synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(STEPS):
            step()
        synchronize()
        duration = time.perf_counter() - start
        return {'status': 'passed', 'comparison': errors, 'tolerance': TOLERANCES[dtype_name],
                'warmed_steps': WARMUP, 'timed_steps': STEPS,
                'mean_attention_step_ms': duration * 1000 / STEPS,
                'cuda_baseline_allocated_bytes': baseline,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                'cuda_peak_increment_bytes': torch.cuda.max_memory_allocated() - baseline if device == 'cuda' else None}


def run_preflight(device='auto', *, importer=importlib.import_module):
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    report = {'schema_version': 1, 'python': platform.python_version(), 'platform': platform.platform(),
              'shape': list(SHAPE), 'causal': True, 'dropout_p': 0.0,
              'backend_policy': 'one explicitly forced backend per probe; no automatic fallback',
              'scope': 'synthetic attention forward/backward only; no model or optimizer step', 'probes': []}
    try:
        torch = importer('torch')
    except (ImportError, OSError) as error:
        return dict(report, status='dependency_unavailable', reason=str(error))
    report.update(pytorch=str(torch.__version__), cuda_build=getattr(torch.version, 'cuda', None))
    cuda = torch.cuda.is_available()
    selected = 'cuda' if device == 'auto' and cuda else 'cpu' if device == 'auto' else device
    report.update(cuda_available=cuda, device=selected)
    if selected == 'cuda' and not cuda:
        return dict(report, status='device_unavailable', reason='CUDA requested but unavailable; no fallback performed')
    if selected == 'cuda':
        report.update(device_name=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
                      total_device_memory_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    else:
        report.update(device_name=platform.processor() or 'CPU', fallback_path='CPU math probe; CUDA fused probes omitted')
    try:
        functional = importer('torch.nn.functional')
        attention = importer('torch.nn.attention')
        enum, kernel = attention.SDPBackend, attention.sdpa_kernel
        if not hasattr(functional, 'scaled_dot_product_attention'):
            raise AttributeError('scaled_dot_product_attention unavailable')
    except (ImportError, AttributeError) as error:
        return dict(report, status='api_unavailable', reason=str(error),
                    fallback_path='no unverified legacy or automatic dispatcher fallback attempted')
    dtype_names = ('float32', 'float16', 'bfloat16') if selected == 'cuda' else ('float32',)
    report['dtype_policy'] = 'empirical forward/backward checks; dtype existence alone is not support'
    fatal = False
    for dtype_name in dtype_names:
        for item in backend_plan(enum, selected == 'cuda'):
            probe = {'dtype': dtype_name, 'backend': item['backend']}
            if not item['eligible'] or not hasattr(torch, dtype_name):
                probe.update(status='not_probed', reason=item['reason'] or 'dtype unavailable')
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        probe.update(_probe(torch, functional, kernel, getattr(enum, item['backend']), dtype_name, selected))
                    except (RuntimeError, NotImplementedError, TypeError, AttributeError) as error:
                        probe.update(status='execution_failed', reason=str(error))
                        # A failed probe is not evidence that another backend ran.
                        if 'out of memory' in str(error).lower() or 'device-side assert' in str(error).lower():
                            fatal = True
                    probe['warnings'] = [str(w.message) for w in caught]
            report['probes'].append(probe)
            if fatal:
                return dict(report, status='aborted', reason='resource/device failure; no retries attempted')
    report['status'] = 'completed'  # Completion is not a claim all backends passed.
    report['supported_for_fixture'] = [{'dtype': p['dtype'], 'backend': p['backend']}
                                       for p in report['probes'] if p['status'] == 'passed']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    report = run_preflight(args.device)
    print(json.dumps(report, indent=2, allow_nan=False))
    # Unsupported optimized kernels are expected. No passing backend is a failure.
    return 0 if report.get('supported_for_fixture') else 2


if __name__ == '__main__':
    raise SystemExit(main())
