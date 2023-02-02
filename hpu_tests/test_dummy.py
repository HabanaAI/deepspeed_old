import os

def test_dummy():
    import deepspeed

    a = 1
    assert a == 1


def test_TestModel():
    import subprocess
    completed_process = subprocess.run(['deepspeed', '--num_nodes', '1', '--num_gpus', '1', '--no_local_rank', 'deepspeed_model.py'])
    assert completed_process.returncode == 0
