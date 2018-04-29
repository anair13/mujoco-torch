import torch
from torch import nn
from mujoco_torch.utils.logger import logger
import mujoco_torch.utils.pythonplusplus as ppp
import numpy as np

from gym.envs.mujoco import HalfCheetahEnv
import gtimer as gt
import mujoco_py

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()

def get_numpy(tensor):
    if isinstance(tensor, Variable):
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
    E = 10
    R = 84
    cuda = True

    envs = []


    for e in range(E):
        env = HalfCheetahEnv()
        envs.append(env)
    c = Convnet(6, output_activation=torch.tanh, input_channels=3)
    if cuda:
        c.cuda()

    # viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=1)
    # env.sim.add_render_context(viewer)

    def step(stamp=True):
        imgs = []
        if i % 100 == 0:
            for e in range(E):
                envs[e].reset()
        for e in range(E):
            img = envs[e].sim.render(R, R, device_id=1).transpose()
            imgs.append(img)
        gt.stamp('render') if stamp else 0

        imgs =np.array(imgs)
        torch_img = np_to_var(imgs)
        if cuda:
            torch_img = torch_img.cuda()
            torch.cuda.synchronize()
        gt.stamp('transfer') if stamp else 0

        u = get_numpy(c.forward(torch_img).cpu())
        torch.cuda.synchronize()
        gt.stamp('forward') if stamp else 0

        for e in range(E):
            env.step(u[e, :])
        gt.stamp('step') if stamp else 0

    for i in range(10):
        step(False)

    gt.stamp('start')
    for i in gt.timed_for(range(100)):
        step()
    gt.stamp('end')

    print(gt.report(include_itrs=False, format_options=dict(itr_num_width=10)))
