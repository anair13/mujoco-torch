from mujoco_torch.doodad.launcher import  run_experiment

def foo(variant):
    import mujoco_py as mp
    br = mp.MjBatchRenderer(32, 32, use_cuda=True)

variant=dict(

)

run_experiment(
    foo,
    variant=variant,
    mode='local_docker',
    use_gpu=True,
)