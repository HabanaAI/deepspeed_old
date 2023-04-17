import os
import argparse
import torch
import deepspeed

class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.l1 = torch.nn.Parameter(torch.ones([1], dtype=torch.float))

    def forward(self, input):
        return self.l1(input)

def optimizers_configuration():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--expected_optimizer')
    args = parser.parse_args()
    model = TestModel()
    model.to('hpu')
    _, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), args=args)
    assert opt.__class__.__name__ == args.expected_optimizer, f"Wrong optimizer type {opt.__class__.__name__} instead of {args.expected_optimizer} for DeepSpeed config={args.deepspeed_config}"

if __name__ == "__main__":
    optimizers_configuration()
