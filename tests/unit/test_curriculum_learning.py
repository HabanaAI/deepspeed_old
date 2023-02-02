import deepspeed
import pytest
import os
import torch
from .common import distributed_test, is_hpu_supported
from .simple_model import Curriculum_SimpleModel, random_dataloader, args_from_dict


def test_curriculum_scheduler_fixed_discrete(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "curriculum_learning": {
            "enabled": True,
            "curriculum_type": "seqlen",
            "min_difficulty": 1,
            "max_difficulty": 5,
            "schedule_type": "fixed_discrete",
            "schedule_config": {
                "difficulty": [1,
                               2,
                               3,
                               4,
                               5],
                "max_step": [2,
                             4,
                             6,
                             8]
            }
        }
    }
    if pytest.use_hpu:
        if os.getenv("REPLACE_FP16", default=None):
            config_dict["fp16"]["enabled"] = False
            config_dict["fp32"] = {"enabled" : True}
        hpu_flag, msg = is_hpu_supported(config_dict)
        if not hpu_flag:
            pytest.skip(msg)
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10
    ground_truths = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
    model = Curriculum_SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_curriculum_scheduler_fixed_discrete(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        if pytest.use_hpu and os.getenv("REPLACE_FP16", default=None):
            data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device, dtype=torch.float)
        else:
            data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            true_seqlen = 5
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
            print('at step {} the seqlen is {}'.format(n + 1, seqlen))
            assert seqlen == true_seqlen, f"Incorrect curriculum schedule"

    _test_curriculum_scheduler_fixed_discrete(args=args,
                                              model=model,
                                              hidden_dim=hidden_dim)


def test_curriculum_scheduler_fixed_linear(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "curriculum_learning": {
            "enabled": True,
            "curriculum_type": "seqlen",
            "min_difficulty": 2,
            "max_difficulty": 10,
            "schedule_type": "fixed_linear",
            "schedule_config": {
                "total_curriculum_step": 8,
                "difficulty_step": 2
            }
        }
    }
    if pytest.use_hpu:
        if os.getenv("REPLACE_FP16", default=None):
            config_dict["fp16"]["enabled"] = False
            config_dict["fp32"] = {"enabled" : True}
        hpu_flag, msg = is_hpu_supported(config_dict)
        if not hpu_flag:
            pytest.skip(msg)
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10
    ground_truths = {1: 2, 2: 4, 3: 4, 4: 6, 5: 6, 6: 8, 7: 8, 8: 10, 9: 10, 10: 10}
    model = Curriculum_SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_curriculum_scheduler_fixed_linear(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        if pytest.use_hpu and os.getenv("REPLACE_FP16", default=None):
            data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device, dtype=torch.float)
        else:
            data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
                print('at step {} the seqlen is {}'.format(n + 1, seqlen))
                assert seqlen == true_seqlen, f"Incorrect curriculum schedule"

    _test_curriculum_scheduler_fixed_linear(args=args,
                                            model=model,
                                            hidden_dim=hidden_dim)
