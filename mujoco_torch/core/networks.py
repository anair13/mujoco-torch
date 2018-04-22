import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mujoco_torch.utils.logger import logger
import mujoco_torch.utils.pythonplusplus as ppp
import numpy as np

from gym.envs.mujoco import HalfCheetahEnv
import gtimer as gt
import mujoco_py

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()

def get_numpy(tensor):
    if isinstance(tensor, TorchVariable):
        return get_numpy(tensor.data)
    return tensor.numpy()

def np_to_var(np_array, **kwargs):
    return Variable(from_numpy(np_array), **kwargs)


class Convnet(nn.Module):
# class ConvVAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            output_activation=ppp.identity,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.output_activation = output_activation
        self.input_channels = input_channels

        self.dist_mu = None
        self.dist_std = None

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128 # kernel 3

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)

    def forward(self, input):
        x = input.view(-1, self.input_channels, 84, 84)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        h = x.view(-1, 128) # flatten
        return self.output_activation(self.fc1(h))

if __name__ == "__main__":
    env = HalfCheetahEnv()
    c = Convnet(6, output_activation=torch.tanh).cuda()

    # viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=1)
    # env.sim.add_render_context(viewer)

    gt.stamp('start')
    for i in gt.timed_for(range(100)):
        if i % 100 == 0:
            env.reset()
        img = env.sim.render(84, 84, device_id=1).transpose()
        gt.stamp('render')

        torch_img = np_to_var(img).cuda()
        u = c.forward(torch_img)
        gt.stamp('forward')

        env.step(env.action_space.sample())
        gt.stamp('step')

    gt.stamp('end')

    print(gt.report(include_itrs=False, format_options=dict(itr_num_width=10)))
