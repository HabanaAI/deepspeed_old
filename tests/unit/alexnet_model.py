import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import deepspeed.comm as dist
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec
import os


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,
                      64,
                      kernel_size=11,
                      stride=4,
                      padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(64,
                      192,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(192,
                      384,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(AlexNet):
    def to_layers(self):
        layers = [*self.features, lambda x: x.view(x.size(0), -1), self.classifier]
        return layers


class AlexNetPipeSpec(PipelineModule):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        specs = [
            LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=5),
            LayerSpec(nn.ReLU, inplace=True),
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),

            lambda x: x.view(x.size(0), -1),
            LayerSpec(nn.Linear, 256, self.num_classes), # classifier
        ]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


# Define this here because we cannot pickle local lambda functions
def cast_to_half(x):
    return x.half()


def cifar_trainset(fp16=False):
    torchvision = pytest.importorskip("torchvision", minversion="0.5.0")
    import torchvision.transforms as transforms

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,
                              0.5,
                              0.5),
                             (0.5,
                              0.5,
                              0.5)),
    ]
    if fp16:
        transform_list.append(torchvision.transforms.Lambda(cast_to_half))

    transform = transforms.Compose(transform_list)

    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    # Only one rank per machine downloads.
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    if os.getenv("CIFAR10_OFFLINE", default = None):
        if os.getenv("CIFAR10_DATASET_PATH", default = None):
            trainset = torchvision.datasets.CIFAR10(root=os.getenv("CIFAR10_DATASET_PATH", default = None),
                                            train=True,
                                            download=False,
                                            transform=transform)
    elif os.getenv("STORE_CIFAR10", default = None):
        if os.getenv("CIFAR10_DATASET_PATH", default = None):
            trainset = torchvision.datasets.CIFAR10(root=os.getenv("CIFAR10_DATASET_PATH", default = None),
                                            train=True,
                                            download=True,
                                            transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='.',
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def train_cifar(model,
                config,
                num_steps=400,
                average_dp_losses=True,
                fp16=True,
                seed=123):
    if bool(pytest.use_hpu) == True:
        import habana_frameworks.torch.hpu as hpu
        device = 'hpu:' + str(hpu.current_device())
        #TODO SW-123528: enable once we added support in HPU
        fork_rng = False
    else:
        device = torch.cuda.current_device()
        fork_rng = True
    with torch.random.fork_rng(enabled=fork_rng):
        ds_utils.set_random_seed(seed)

        # disable dropout
        model.eval()

        trainset = cifar_trainset(fp16=fp16)
        config['local_rank'] = dist.get_rank()

        engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            model_parameters=[p for p in model.parameters()],
            training_data=trainset)

        losses = []
        for step in range(num_steps):
            loss = engine.train_batch()
            losses.append(loss.item())
            if step % 50 == 0 and dist.get_rank() == 0:
                print(f'STEP={step} LOSS={loss.item()}')

        if average_dp_losses:
            loss_tensor = torch.tensor(losses).to(device)
            dist.all_reduce(loss_tensor)
            loss_tensor /= dist.get_world_size()
            losses = loss_tensor.tolist()

    return losses
