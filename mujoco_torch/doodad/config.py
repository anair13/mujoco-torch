# Change these things
CODE_DIRS_TO_MOUNT = [
    '/home/ashvin/code/mujoco-torch',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/ashvin/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/ashvin/data/s3doodad'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/ashvin/code/mujoco-torch/mujoco_torch/doodad/run_experiment_from_doodad.py'
)

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'anair17/mujoco-torch'
INSTANCE_TYPE = 'c4.8xlarge'
SPOT_PRICE = 2.0
SPOT_PRICE_LOOKUP = {'c4.large': 0.1, 'm4.large': 0.1, 'm4.xlarge': 0.2, 'm4.2xlarge': 0.4, 'c4.8xlarge': 2.0, 'c4.4xlarge': 1.0}

GPU_DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-vitchyr-gpu'
GPU_INSTANCE_TYPE = 'g2.2xlarge'
GPU_SPOT_PRICE = 0.5
GPU_AWS_IMAGE_ID = "ami-874378e7"

REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-west-1': "ami-874378e7",
    'us-east-1': "ami-0ef1b374",
}

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
