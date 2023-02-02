import torch
import numpy as np
import pytest
from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import CPUAdamBuilder

if pytest.use_hpu:
    import habana_frameworks.torch.core as htcore
    pytest.skip("DeepSpeed CPU adam is not Supported by HPU", allow_module_level=True)

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible")

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower()


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    print("ATOL", atol)
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


@pytest.mark.parametrize('dtype', [torch.half, torch.float], ids=["fp16", "fp32"])
@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             #(55),
                             (128),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
def test_cpu_adam_opt(dtype, model_size):
    if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
        pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

    from deepspeed.ops.adam import DeepSpeedCPUAdam
    device = 'cpu'
    rng_state = torch.get_rng_state()
    param = torch.nn.Parameter(torch.randn(model_size, device=device).to(dtype))
    torch.set_rng_state(rng_state)
    param1_data = torch.randn(model_size, device=device)
    param1 = torch.nn.Parameter(param1_data)
    torch.set_rng_state(rng_state)
    if pytest.use_hpu:
        param2_data = torch.randn(model_size, device=device).to('hpu')
    else:
        param2_data = torch.randn(model_size, device=device).cuda()
    param2 = torch.nn.Parameter(param2_data)

    optimizer1 = torch.optim.AdamW([param1])
    optimizer2 = FusedAdam([param2])
    optimizer = DeepSpeedCPUAdam([param])

    for i in range(10):
        rng_state = torch.get_rng_state()
        param.grad = torch.randn(model_size, device=device).to(dtype)
        torch.set_rng_state(rng_state)
        param1.grad = torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        if pytest.use_hpu:
            param2.grad = torch.randn(model_size, device=device).to('hpu')
        else:
            param2.grad = torch.randn(model_size, device=device).cuda()

        optimizer.step()
        optimizer2.step()
        optimizer1.step()
    tolerance = param1.float().norm().detach().numpy() * 1e-2
    check_equal(param.float().norm(),
                param1.float().norm(),
                atol=tolerance,
                verbose=True)
    check_equal(param.float().norm(),
                param2.float().cpu().norm(),
                atol=tolerance,
                verbose=True)


def test_cpu_adam_gpu_error():
    model_size = 64
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    if pytest.use_hpu:
        device = 'hpu:0'
    else:
        device = 'cuda:0'
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    optimizer = DeepSpeedCPUAdam([param])

    param.grad = torch.randn(model_size, device=device)
    with pytest.raises(AssertionError):
        optimizer.step()
