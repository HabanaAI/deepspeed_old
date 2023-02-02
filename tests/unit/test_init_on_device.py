import torch
import pytest
import os
from .simple_model import SimpleModel
from deepspeed import OnDevice
from packaging import version as pkg_version

if pytest.use_hpu:
    devices = ['meta', 'hpu:0']
else:
    devices = ['meta', 'cuda:0']


@pytest.mark.parametrize('device', [devices[0],
                                    pytest.param(devices[1], marks=pytest.mark.xfail(pytest.use_hpu == True, reason="xfail, due to SW-112908"))])
def test_on_device(device):
    if device == "meta" and pkg_version.parse(
            torch.__version__) < pkg_version.parse("1.10"):
        pytest.skip("meta tensors only became stable after torch 1.10")

    if pytest.use_hpu:
        if os.getenv("REPLACE_FP16", default = None):
            datatype=torch.bfloat16
        else:
            datatype=torch.half
        with OnDevice(dtype = datatype, device = device):
            model=SimpleModel(4)
    else:
        with OnDevice(dtype = torch.half, device = device):
            model=SimpleModel(4)

    for p in model.parameters():
        assert p.device == torch.device(device)
        if pytest.use_hpu:
            assert p.dtype == datatype
        else:
            assert p.dtype == torch.half
