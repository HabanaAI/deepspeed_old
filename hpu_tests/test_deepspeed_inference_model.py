import os
import subprocess
import pytest

@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@pytest.mark.parametrize("tp_size", ['1', '2'])
def test_TestModel(enable_cuda_graph, tp_size):
    script_path = os.getenv("DEEPSPEED_FORK_ROOT") + "/hpu_tests/deepspeed_inference_model.py"

    cmd = ['deepspeed', '--num_nodes', '1', '--num_gpus', '2', '--no_local_rank', script_path, '--use_hpu', '--tp_size', tp_size]
    if enable_cuda_graph:
        cmd.append("--enable_cuda_graph")

    completed_process = subprocess.run(cmd)
    assert completed_process.returncode == 0
