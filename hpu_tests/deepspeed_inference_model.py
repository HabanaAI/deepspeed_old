import os
import argparse
import torch
import deepspeed

class TestLayer(torch.nn.Module):
    def __init__(self, data_size):
        super(TestLayer, self).__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size], dtype=torch.float))
    def forward(self, input):
        output = input * torch.matmul(input, self.w)
        return output

class TestModel(torch.nn.Module):
    def __init__(self, data_size):
        super(TestModel, self).__init__()
        self.l1 = TestLayer(data_size)
        self.l2 = TestLayer(data_size)
        self.l3 = TestLayer(data_size)
        self.l4 = TestLayer(data_size)

    def forward(self, input):
        l1_out = self.l1(input)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)
        return l4_out

def deepspeed_inference_model():
    import habana_frameworks.torch.hpu as ht
    import deepspeed.module_inject as module_inject

    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_cuda_graph', action='store_true')
    parser.add_argument('--tp_size')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    data_size = 2
    model = TestModel(data_size)
    model.to('hpu')
    model.eval()
    kwargs = dict(dtype=torch.float)
    kwargs["tensor_parallel"] = {"tp_size": int(args.tp_size)}
    kwargs['enable_cuda_graph'] = args.enable_cuda_graph
    kwargs['replace_method'] = "auto"
    kwargs['replace_with_kernel_inject'] = False
    kwargs['injection_policy'] = {"BertLayer": (module_inject.HFBertLayerPolicy,)}
    engine = deepspeed.init_inference(model=model, **kwargs)
    input = torch.ones([data_size, data_size], dtype=torch.float, device="hpu")
    ds_output = engine(input)
    ht.synchronize()
    expected = torch.tensor([32768., 32768.])
    assert torch.allclose(ds_output, expected), f"incorrect result value {ds_output}, expected {expected}"

if __name__ == "__main__":
    deepspeed_inference_model()
