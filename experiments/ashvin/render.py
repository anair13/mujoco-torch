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
    ingpu = False
    R = 84
    E = 100
    N = 100

    if ingpu:
        from mujoco_torch.core.bridge import MjCudaRender
        renderer = MjCudaRender(84, 84, E)

    envs = []
    for e in range(E):
        env = HalfCheetahEnv()
        envs.append(env)

    c = Convnet(6, output_activation=torch.tanh, input_channels=3)
    if cuda:
        c.cuda()

    def step(stamp=True):
        for e in range(E):
            env = envs[e]
            env.step(np.random.rand(6))
        gt.stamp('step') if stamp else 0

        if ingpu:
            sims = [env.sim for env in envs]
            env = envs[e]
            tensor, img = renderer.get_batch_cuda_tensor(sims, False)
            tensor = Variable(tensor).float()
            gt.stamp('render') if stamp else 0

        else:
            imgs = []
            for e in range(E):
                env = envs[e]
                img = env.sim.render(R, R, device_id=1)
                imgs.append(img)
            gt.stamp('render') if stamp else 0

            imgs = np.array(imgs)
            tensor = np_to_var(imgs)
            if cuda:
                tensor = tensor.cuda()
                torch.cuda.synchronize()
            gt.stamp('transfer') if stamp else 0

        u = get_numpy(c.forward(tensor).cpu())
        torch.cuda.synchronize()
        gt.stamp('forward') if stamp else 0

        # cv2.imshow("img", img)
        # cv2.waitKey(1)

    gt.stamp("start")
    for i in range(10):
        step(False)

    gt.stamp("warmstart")
    for i in gt.timed_for(range(N)):
        step()

    gt.stamp("end")

    print(gt.report(include_itrs=False))

variant=dict(
)

run_variants(experiment, [variant])
