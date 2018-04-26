from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjBatchRenderer
import math
import os
import torch
import numpy as np
import gtimer as gt
import GPUtil

from mpi4py import MPI
from mujoco_torch.doodad.arglauncher import run_variants
from mujoco_torch.core.networks import Convnet, np_to_var, get_numpy
from gym.envs.mujoco import HalfCheetahEnv

def experiment(variant):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()
    root = 0
    gpus = GPUtil.getGPUs()
    n_gpu = len(gpus)

    E = 20
    R = 84
    U = 6
    cuda = True

    envs = []

    for e in range(rank, E, n_proc):
        env = HalfCheetahEnv()
        envs.append(env)

    sendcounts = np.array(comm.gather(len(envs), root))

    i_sendcounts = None
    u_sendcounts = None

    if rank == root:
        i_sendcounts = sendcounts*3*R*R
        u_sendcounts = sendcounts*U
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
            img = e.sim.render(R, R, device_id=rank%n_gpu).transpose()
            imgs.append(img)
        comm.Barrier()
        if rank == 0:
            gt.stamp('render') if stamp else 0

        imgs = np.array(imgs)
        r_imgs = None
        if rank == 0:
            #r_imgs = np.empty((E,shape[1],shape[2],shape[3]), dtype='uint8')
            r_imgs = np.empty((E,3,R,R), dtype='uint8')

        comm.Gatherv(sendbuf=imgs, recvbuf=(r_imgs, i_sendcounts), root=root)
        if rank == 0:
            gt.stamp('comm1') if stamp else 0

        u = None
        if rank == 0:
            torch_img = np_to_var(r_imgs)
            if cuda:
                torch_img = torch_img.cuda()
                torch.cuda.synchronize()
            gt.stamp('transfer') if stamp else 0

            u = get_numpy(c.forward(torch_img).cpu())
            torch.cuda.synchronize()
            gt.stamp('forward') if stamp else 0

        r_u = np.empty((len(envs) , U), dtype='float32')
        comm.Scatterv(sendbuf=(u, u_sendcounts), recvbuf=r_u, root=root)
        if rank == 0:
            gt.stamp('comm2') if stamp else 0
        for i, e in enumerate(envs):
            e.step(r_u[i, :])
        comm.Barrier()
        if rank == 0:
            gt.stamp('step') if stamp else 0

    for i in range(10):
        step(i, False)

    if rank == 0:
        gt.stamp('start')
    for i in gt.timed_for(range(100)):
        step(i)
    if rank == 0:
        gt.stamp('end')

        print(gt.report(include_itrs=False, format_options=dict(itr_num_width=10)))

if __name__ == "__main__":
    variants = [dict()]
    #run_variants(experiment, variants)
    experiment(variants)
