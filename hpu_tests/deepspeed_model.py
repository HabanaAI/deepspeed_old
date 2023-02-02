import os
import argparse
import torch
import deepspeed

class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([100], dtype=torch.float))
    def forward(self, input):
        output = input * self.a
        return output

def deepspeed_model():
    import habana_frameworks.torch.hpu as ht
    import habana_frameworks.torch.core as htcore
    model = TestModel()

    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.deepspeed_config = os.getenv("DEEPSPEED_FORK_ROOT") + "/hpu_tests/test_dummy.json"
    args.use_hpu = True
    model.to('hpu')
    optimizer = torch.optim.AdamW(model.parameters())
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, args=args)

    for i in range(20):
        input = torch.randn([16, 100]).float().to('hpu')
        target = torch.randn([16, 100]).float().to('hpu')
        output = model_engine(input)
        loss = torch.sum(torch.abs(target-output))
        loss = model_engine.backward(loss)
        htcore.mark_step()
        model_engine.step()
        htcore.mark_step()
        print(f"complete step {i}")
    ht.synchronize()

if __name__ == "__main__":
    deepspeed_model()
