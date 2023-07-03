import torch
import deepspeed
import pytest
from unit.common import DistributedTest
from unit.simple_model import SimplePRMoEModel, SimpleMoEModel, sequence_dataloader
from unit.util import required_torch_version
from unit.hpu import *


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("use_residual", [True, False])
class TestMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, use_residual):
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 8,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            }
        }
        hidden_dim = 16
        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled" : True}
                dtype = torch.float
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device,
                                          dtype=dtype)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("ep_size, use_residual", [(2, True), (2, False)])
class TestPRMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, use_residual):
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 8,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            }
        }
        hidden_dim = 16
        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled" : True}
                dtype = torch.float
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimplePRMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device,
                                          dtype=dtype)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
