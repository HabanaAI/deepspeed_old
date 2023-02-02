import os
import time

import torch
import deepspeed.comm as dist
from torch.multiprocessing import Process

import deepspeed

import pytest

from pathlib import Path

# Worker timeout *after* the first worker has completed.
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120


def get_xdist_worker_id():
    xdist_worker = os.environ.get('PYTEST_XDIST_WORKER', None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace('gw', '')
        return int(xdist_worker_id)
    return None


def get_master_port():
    master_port = os.environ.get('DS_TEST_PORT', '29503')
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is not None:
        master_port = str(int(master_port) + xdist_worker_id)
    return master_port


def set_cuda_visibile():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it from nvidia-smi instead
        import subprocess
        is_rocm_pytorch = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm_pytorch:
            rocm_smi = subprocess.check_output(['rocm-smi', '--showid'])
            gpu_ids = filter(lambda s: 'GPU' in s,
                             rocm_smi.decode('utf-8').strip().split('\n'))
            num_gpus = len(list(gpu_ids))
        else:
            nvidia_smi = subprocess.check_output(['nvidia-smi', '--list-gpus'])
            num_gpus = len(nvidia_smi.decode('utf-8').strip().split('\n'))
        cuda_visible = ",".join(map(str, range(num_gpus)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)


def distributed_test(world_size=2, backend='nccl'):
    """A decorator for executing a function (e.g., a unit test) in a distributed manner.
    This decorator manages the spawning and joining of processes, initialization of
    deepspeed.comm, and catching of errors.

    Usage example:
        @distributed_test(worker_size=[2,3])
        def my_test():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert(rank < world_size)

    Arguments:
        world_size (int or list): number of ranks to spawn. Can be a list to spawn
        multiple tests.
    """
    def dist_wrap(run_func):
        """Second-level decorator for dist_test. This actually wraps the function. """
        def dist_init(local_rank, num_procs, *func_args, **func_kwargs):
            """Initialize deepspeed.comm and execute the user function. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = get_master_port()
            os.environ['LOCAL_RANK'] = str(local_rank)
            # NOTE: unit tests don't support multi-node so local_rank == global rank
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(num_procs)

            # turn off NCCL logging if set
            os.environ.pop('NCCL_DEBUG', None)
            if pytest.use_hpu:
                backend='hccl'
            else:
                set_cuda_visibile()

            deepspeed.init_distributed(dist_backend=backend)
            #dist.init_process_group(backend=backend)
            dist.barrier()

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            run_func(*func_args, **func_kwargs)

            # make sure all ranks finish at the same time
            dist.barrier()
            # tear down after test completes
            dist.destroy_process_group()

        def dist_launcher(num_procs, *func_args, **func_kwargs):
            """Launch processes and gracefully handle failures. """

            # Spawn all workers on subprocesses.
            processes = []
            for local_rank in range(num_procs):
                p = Process(target=dist_init,
                            args=(local_rank,
                                  num_procs,
                                  *func_args),
                            kwargs=func_kwargs)
                p.start()
                processes.append(p)

            # Now loop and wait for a test to complete. The spin-wait here isn't a big
            # deal because the number of processes will be O(#GPUs) << O(#CPUs).
            any_done = False
            while not any_done:
                for p in processes:
                    if not p.is_alive():
                        any_done = True
                        break

            # Wait for all other processes to complete
            for p in processes:
                p.join(DEEPSPEED_UNIT_WORKER_TIMEOUT)

            failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
            for rank, p in failed:
                # If it still hasn't terminated, kill it because it hung.
                if p.exitcode is None:
                    p.terminate()
                    pytest.fail(f'Worker {rank} hung.', pytrace=False)
                if p.exitcode < 0:
                    pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}',
                                pytrace=False)
                if p.exitcode > 0:
                    pytest.fail(f'Worker {rank} exited with code {p.exitcode}',
                                pytrace=False)

        def run_func_decorator(*func_args, **func_kwargs):
            """Entry point for @distributed_test(). """

            if isinstance(world_size, int):
                dist_launcher(world_size, *func_args, **func_kwargs)
            elif isinstance(world_size, list):
                for procs in world_size:
                    dist_launcher(procs, *func_args, **func_kwargs)
                    time.sleep(0.5)
            else:
                raise TypeError(f'world_size must be an integer or a list of integers.')

        return run_func_decorator

    return dist_wrap

def get_hpu_dev_version():
    try:
        command_output = os.popen('sudo hl-smi -L')
        command_output_list = command_output.read()
        device_id = [s for s in command_output_list.split('\n') if 'Device Id' in s][0].split()[-1]
        if ('0x1da31000' in device_id) or ('0x1da31001' in device_id):
            return "Gaudi"
        elif '0x1da31020' in device_id:
            return "Gaudi2"
    except:
        pass
    return None


def is_hpu_supported(config):
    # FP16 is not supported by HPU.
    if config.get('fp16'):
        if config.get('fp16', None).get('enabled', None) == True:
            if get_hpu_dev_version() == 'Gaudi':
                return False, "FP16 datatype is not supported by HPU"
    # Fused ADAM is not supported
    if config.get('optimizer'):
        if config.get('optimizer', None).get('params', None):
            if config.get('optimizer', None).get('params', None).get('torch_adam', None) == False:
                return False, "Fused ADAM optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "Lamb":
                return False, "LAMB optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "OneBitAdam":
                return False, "OneBitAdam optimizer is not supported by HPU"
    # Zero-3 is not supported
    if config.get('zero_optimization'):
        if config.get('zero_optimization', None).get('stage', None) == 3:
            return False, "DeepSpeed Stage3 is not supported by HPU"
    # CPU offload is not supported
    if config.get('zero_optimization'):
        if config.get('zero_optimization', None).get('cpu_offload', None) == True:
            return False, "CPU offload is not supported by HPU"
        if config.get('zero_optimization', None).get('offload_param', None):
            if config.get('zero_optimization', None).get('offload_param', None).get('device', None) == 'cpu':
                return False, "CPU offload param is not supported by HPU"
        if config.get('zero_optimization', None).get('offload_optimizer', None):
            if config.get('zero_optimization', None).get('offload_optimizer', None).get('device', None) == 'cpu':
                return False, "CPU offload optimizer is not supported by HPU"
    # FLOPS profiler is not supported by HPU
    if config.get('flops_profiler'):
        if config.get('flops_profiler', None).get('enabled', None) == True:
            return False, "FLOPS profiler is not supported by HPU"
    # wall_clock_breakdown is not supported by HPU.
    if 'wall_clock_breakdown' in config:
        if config['wall_clock_breakdown'] == True:
            return False, "Wall Clock breakdown is not supported by HPU"
    # sparse gradients is not supported by HPU.
    if 'sparse_gradients' in config:
        if config['sparse_gradients'] == True:
            return False, "sparse_gradients is not supported by HPU"

    return True, ''

def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))
