import os
import time
import pytest
import torch
import deepspeed
from transformers import pipeline
from unit.common import DistributedTest
from unit.hpu import *


@pytest.fixture
def query(model, task):
    if task == "text-generation":
        return "DeepSpeed is"
    elif task == "fill-mask":
        if "roberta" in model:
            return "I am a <mask> model"
        else:
            return "I am a [MASK] model"
    else:
        raise NotImplementedError


@pytest.fixture
def inf_kwargs(task):
    if task == "text-generation":
        return {"do_sample": False, "min_length": 50, "max_length": 50}
    else:
        return {}


@pytest.mark.inference
@pytest.mark.parametrize("model,task",
                         [
                             ("bert-base-cased",
                              "fill-mask"),
                             ("roberta-base",
                              "fill-mask"),
                             ("gpt2",
                              "text-generation"),
                             ("facebook/opt-125m",
                              "text-generation"),
                             ("bigscience/bloom-560m",
                              "text-generation"),
                         ])
@pytest.mark.parametrize("cuda_graphs", [True, False])
@pytest.mark.parametrize("use_cuda_events", [True, False])
class TestModelProfiling(DistributedTest):
    world_size = 1

    def test(self,
             model,
             task,
             query,
             inf_kwargs,
             cuda_graphs,
             use_cuda_events,
             dtype=torch.float16):
        if cuda_graphs and "bert" not in model:
            pytest.skip(f"CUDA Graph not supported for {model}")

        if bool(pytest.use_hpu) == True:
            # FP16 is not supported on Gaudi1.
            if get_hpu_dev_version() == "Gaudi":
                pytest.skip(f"FP16 tests are not supported by Gaudi1.")

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if bool(pytest.use_hpu) == True:
            dev_name = f"hpu:{local_rank}"
            pipe = pipeline(task, model, framework="pt", device=dev_name)
            device = torch.device(f"hpu:{local_rank}")
            pipe.device = device
            pipe.model.to(device)
        else:
            pipe = pipeline(task, model, framework="pt", device=local_rank)
        if bool(pytest.use_hpu) == True:
            import deepspeed.module_inject as module_inject
            import habana_frameworks.torch as htorch
            pipe.model = deepspeed.init_inference(pipe.model,
                                              dtype=dtype,
                                              mp_size=world_size,
                                              replace_with_kernel_inject=False,
                                              replace_method="auto",
                                              injection_policy={"BertLayer": (module_inject.HFBertLayerPolicy,)},
                                              enable_cuda_graph=cuda_graphs)
        else:
            pipe.model = deepspeed.init_inference(pipe.model,
                                              dtype=dtype,
                                              mp_size=world_size,
                                              replace_with_kernel_inject=True,
                                              replace_method="auto",
                                              enable_cuda_graph=cuda_graphs)
        pipe.model.profile_model_time(use_cuda_events=use_cuda_events)

        e2e_times = []
        model_times = []
        for _ in range(10):
            if bool(pytest.use_hpu) == True:
                htorch.hpu.synchronize()
            else:
                torch.cuda.synchronize()
            start = time.perf_counter_ns()

            r = pipe(query, **inf_kwargs)

            if bool(pytest.use_hpu) == True:
                htorch.hpu.synchronize()
            else:
                torch.cuda.synchronize()
            end = time.perf_counter_ns()

            e2e_times.append((end - start) / 1e6)  # convert ns to ms
            model_times.extend(pipe.model.model_times())

        for e2e_t, model_t in zip(e2e_times, model_times):
            assert e2e_t >= model_t
