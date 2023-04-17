import deepspeed
from deepspeed.ops.op_builder import CPUAdamBuilder

from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import checkpoint_correctness_verification
from unit.hpu import *
import pytest


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [(0,
                           False),
                          (1,
                           False),
                          (2,
                           False),
                          (2,
                           True),
                          (3,
                           False),
                          (3,
                           True)])
class TestLRSchedulerCheckpoint(DistributedTest):
    world_size = 2

    def test_checkpoint_lr_scheduler(self, tmpdir, zero_stage, use_cpu_offload):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8,
                              0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.001,
                    "warmup_num_steps": 1000
                }
            }
        }
        hidden_dim = 10
        fp16=True
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled" : True}
                if zero_stage == 1:
                    config_dict["communication_data_type"] = 'bfp16'
                fp16=False
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        if zero_stage == 3:
            global DeepSpeedZeroOptimizer_Stage3
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=False,
                                            load_lr_scheduler_states=True,
                                            fp16=fp16)

    def test_checkpoint_no_lr_scheduler(self, tmpdir, zero_stage, use_cpu_offload):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 1e-5
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.001,
                    "warmup_num_steps": 1000
                }
            },
        }
        hidden_dim = 10

        fp16=True
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled" : True}
                if zero_stage == 1:
                    config_dict["communication_data_type"] = 'bfp16'
                fp16=False
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)


        if zero_stage == 3:
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=False,
                                            load_lr_scheduler_states=False,
                                            fp16=fp16)
