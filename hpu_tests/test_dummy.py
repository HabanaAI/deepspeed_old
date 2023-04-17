import os
import subprocess
import pytest
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

@pytest.mark.parametrize("ds_config", ["zero0", "zero1", "zero2", "zero2_offload", "zero3"])
@pytest.mark.parametrize("activation_checkpoints, partition_activations, contiguous_checkpointing",
                         [(False, False, False),
                          (True, False, False),
                          (True, True, False),
                          (True, True, True)])
@pytest.mark.parametrize("use_config_optimizer, expected_optimizer", [(False, "AdamW"), (True, "FusedAdamW")])
def test_TestModel(ds_config, activation_checkpoints, partition_activations, contiguous_checkpointing, use_config_optimizer, expected_optimizer):
    script_path = os.getenv("DEEPSPEED_FORK_ROOT") + "/hpu_tests/deepspeed_model.py"
    ds_config_path = os.getenv("DEEPSPEED_FORK_ROOT") + f"/hpu_tests/test_dummy_{ds_config}.json"

    cmd = ['deepspeed', '--num_nodes', '1', '--num_gpus', '2', '--no_local_rank', script_path, '--deepspeed_config', ds_config_path, '--use_hpu', '--expected_optimizer']
    if ds_config == "zero2_offload":
        # TODO: SW-122610 - once SW-122430 is done we should change "AdamW" below to "DeepSpeedCPUAdam"
        cmd.append("AdamW")
    else:
        cmd.append(expected_optimizer)
    if use_config_optimizer:
        cmd.append("--use_config_optimizer")
    if activation_checkpoints:
        cmd.append("--activation-checkpoints")
    if partition_activations:
        cmd.append("--partition-activations")
    if contiguous_checkpointing:
        cmd.append("--contiguous-checkpointing")

    completed_process = subprocess.run(cmd)
    assert completed_process.returncode == 0
