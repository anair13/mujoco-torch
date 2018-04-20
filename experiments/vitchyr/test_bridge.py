
from mujoco_torch.doodad.launcher import  run_experiment

def experiment(variant):
    from gym.envs.mujoco import HalfCheetahEnv
    from mujoco_torch.core.bridge import MjCudaRender
    renderer = MjCudaRender(32, 32)
    env = HalfCheetahEnv()

    renderer.get_cuda_tensor(env.sim)

variant=dict(
)

run_experiment(experiment)
