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
    ingpu = True

    if ingpu:
        from mujoco_torch.core.bridge import MjCudaRender
        renderer = MjCudaRender(84, 84)

    R = 84
    env = HalfCheetahEnv()
    c = Convnet(6, output_activation=torch.tanh, input_channels=3)
    if cuda:
        c.cuda()

    def step(stamp=True):
        env.step(np.random.rand(6))
        gt.stamp('step') if stamp else 0

        if ingpu:
            tensor, img = renderer.get_cuda_tensor(env.sim, False)
            gt.stamp('render') if stamp else 0

        else:
            img = env.sim.render(R, R, device_id=1)
            gt.stamp('render') if stamp else 0

            x = np_to_var(img)
            if cuda:
                x = x.cuda()
                torch.cuda.synchronize()
            gt.stamp('transfer') if stamp else 0

        # cv2.imshow("img", img)
        # cv2.waitKey(1)

    gt.stamp("start")
    for i in range(100):
        step(False)

    gt.stamp("warmstart")
    for i in gt.timed_for(range(1000)):
        step()

    gt.stamp("end")

    print(gt.report(include_itrs=False))

variant=dict(
)

run_variants(experiment, [variant])
