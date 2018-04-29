from mujoco_torch.doodad.arglauncher import run_variants

import torch
from torch import nn
from mujoco_torch.utils.logger import logger
import mujoco_torch.utils.pythonplusplus as ppp
from torch.autograd import Variable

from mujoco_torch.core.networks import Convnet
from gym.envs.mujoco import HalfCheetahEnv
import gtimer as gt
import mujoco_py

import gtimer as gt
import numpy as np

import cv2

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()

def get_numpy(tensor):
    if isinstance(tensor, Variable):
        return get_numpy(tensor.data)
    return tensor.numpy()

def np_to_var(np_array, **kwargs):
    return Variable(from_numpy(np_array), **kwargs)

def experiment(variant):
    cuda = True
    from gym.envs.mujoco import HalfCheetahEnv
    from mujoco_torch.core.bridge import MjCudaRender
    R = 84
    env = HalfCheetahEnv()
    c = Convnet(6, output_activation=torch.tanh, input_channels=3)
    if cuda:
        c.cuda()

    gt.stamp("start")
    for i in range(100):
        img = env.sim.render(R, R, device_id=1)

    gt.stamp("warmstart")
    for i in gt.timed_for(range(1000)):
        env.step(np.random.rand(6))
        gt.stamp('step')

        img = env.sim.render(R, R, device_id=1)
        gt.stamp('render')

        x = np_to_var(img)
        if cuda:
            x = x.cuda()
            torch.cuda.synchronize()
        gt.stamp('transfer')
        # cv2.imshow("img", img)
        # cv2.waitKey(1)
    gt.stamp("end")

    print(img)

    print(gt.report(include_itrs=False))

variant=dict(
)

run_variants(experiment, [variant])
