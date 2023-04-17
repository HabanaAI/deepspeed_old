import os
import subprocess
import pytest

@pytest.mark.parametrize("ds_config, expected_optimizer",
        [("adam", "FusedAdamW"), ("adam_torch_adam", "AdamW"), ("adam_torch_adam_adamw_mode_false", "Adam"),
         ("adam_adamw_mode_false", "Adam"), ("adamw", "FusedAdamW"), ("adamw_torch_adam", "AdamW")])
def test_TestModel(ds_config, expected_optimizer):
    script_path = os.getenv("DEEPSPEED_FORK_ROOT") + "/hpu_tests/optimizers_configuration.py"
    ds_config_path = os.getenv("DEEPSPEED_FORK_ROOT") + f"/hpu_tests/test_opt_type_{ds_config}.json"

    cmd = ['deepspeed', '--num_nodes', '1', '--num_gpus', '1', '--no_local_rank', script_path, '--deepspeed_config', ds_config_path, '--use_hpu', '--expected_optimizer', expected_optimizer]

    completed_process = subprocess.run(cmd)
    assert completed_process.returncode == 0
