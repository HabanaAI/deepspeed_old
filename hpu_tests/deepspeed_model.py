import os
import argparse
import torch
import deepspeed
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

class TestLayer(torch.nn.Module):
    def __init__(self, data_size):
        super(TestLayer, self).__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size], dtype=torch.float))
    def forward(self, input):
        output = input * self.w
        return output

class TestModel(torch.nn.Module):
    def __init__(self, data_size):
        super(TestModel, self).__init__()
        self.l1 = TestLayer(data_size)
        self.l2 = TestLayer(data_size)
        self.l3 = TestLayer(data_size)
        self.l4 = TestLayer(data_size)

    def forward(self, input):
        if deepspeed.checkpointing.is_configured():
            l1_out = self.l1(input)
            l2_out = checkpoint(self.l2, l1_out)
            l3_out = checkpoint(self.l3, l2_out)
            l4_out = checkpoint(self.l4, l3_out)
        else:
            l1_out = self.l1(input)
            l2_out = self.l2(l1_out)
            l3_out = self.l3(l2_out)
            l4_out = self.l4(l3_out)
        return l4_out

def deepspeed_model():
    import habana_frameworks.torch.hpu as ht
    import habana_frameworks.torch.core as htcore

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation-checkpoints', action='store_true')
    parser.add_argument('--partition-activations', action='store_true')
    parser.add_argument('--contiguous-checkpointing', action='store_true')
    parser.add_argument('--use_config_optimizer', action='store_true')
    parser.add_argument('--expected_optimizer')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="hccl")

    data_size = 10
    model = TestModel(data_size)
    model.to('hpu')
    optimizer = None
    model_params = None
    if args.use_config_optimizer:
        model_params = model.parameters()
    else:
        optimizer = torch.optim.AdamW(model.parameters())
    model_engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model_params, args=args)
    if opt.__class__.__name__.__contains__("DeepSpeedZeroOptimizer"):
        opt = opt.optimizer
    assert opt.__class__.__name__ == args.expected_optimizer, f"Wrong optimizer type {opt.__class__.__name__} instead of {args.expected_optimizer}"

    _, micro_batch, gradient_accumulation_steps = model_engine.get_batch_info()

    if args.activation_checkpoints:
        deepspeed.checkpointing.configure(None, deepspeed_config=args.deepspeed_config,
                                          partition_activations=args.partition_activations,
                                          contiguous_checkpointing=args.contiguous_checkpointing,
                                          num_checkpoints=3)

    for i in range(2 * micro_batch() * gradient_accumulation_steps()):
        input = torch.ones([micro_batch(), data_size], dtype=torch.float, device="hpu")
        target = torch.zeros([micro_batch(), data_size], dtype=torch.float, device="hpu")
        output = model_engine(input)
        loss = torch.sum(torch.abs(target - output)) / (micro_batch() * data_size * 50)
        model_engine.backward(loss)
        htcore.mark_step()
        model_engine.step()
        htcore.mark_step()
        print(f"complete step {i}, loss = {loss}")
    ht.synchronize()
    expected = torch.tensor([0.0164])
    assert torch.allclose(loss, expected, atol=4e-4), f"incorrect loss value {loss}, expected {expected}"

if __name__ == "__main__":
    deepspeed_model()
