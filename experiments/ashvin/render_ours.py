from mujoco_torch.doodad.arglauncher import run_variants

import torch
from torch import nn
from mujoco_torch.utils.logger import logger
import mujoco_torch.utils.pythonplusplus as ppp
from torch.autograd import Variable

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
    from gym.envs.mujoco import HalfCheetahEnv
    from mujoco_torch.core.bridge import MjCudaRender
    renderer = MjCudaRender(84, 84)
    env = HalfCheetahEnv()

    gt.stamp("start")
    for i in range(100):
        tensor, img = renderer.get_cuda_tensor(env.sim, False)

    gt.stamp("warmstart")
    for i in gt.timed_for(range(1000)):
        env.step(np.random.rand(6))
        gt.stamp('step')

        tensor, img = renderer.get_cuda_tensor(env.sim, False)
        gt.stamp('render')
        # cv2.imshow("img", img)
        # cv2.waitKey(1)
    gt.stamp("end")

    print(img)

    print(gt.report(include_itrs=False))

variant=dict(
)

run_variants(experiment, [variant])
