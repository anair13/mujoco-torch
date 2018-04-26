from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjBatchRenderer
import math
import os
import torch
import numpy as np
import gtimer as gt

from mujoco_torch.doodad.arglauncher import run_variants
from mujoco_torch.core.networks import Convnet, np_to_var, get_numpy
from gym.envs.mujoco import HalfCheetahEnv

def experiment(variant):

    root = 0

    E = 20
    R = 84
    U = 6
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

    def step(i, stamp=True):
        imgs = []
        if i % 100 == 0:
            for e in envs:
                e.reset()
        for e in envs:
            img = e.sim.render(R, R, device_id=0).transpose()
            imgs.append(img)
        gt.stamp('render') if stamp else 0

        imgs = np.array(imgs)

        torch_img = np_to_var(imgs)
        if cuda:
            torch_img = torch_img.cuda()
            torch.cuda.synchronize()
        gt.stamp('transfer') if stamp else 0

        u = get_numpy(c.forward(torch_img).cpu())
        torch.cuda.synchronize()
        gt.stamp('forward') if stamp else 0

        for i, e in enumerate(envs):
            e.step(u[i, :])
        gt.stamp('step') if stamp else 0

    for i in range(10):
        step(i, False)

    gt.stamp('start')
    for i in gt.timed_for(range(100)):
        step(i)
    gt.stamp('end')

    print(gt.report(include_itrs=False, format_options=dict(itr_num_width=10)))

if __name__ == "__main__":
    variants = [dict()]
    #run_variants(experiment, variants)
    experiment(variants)

