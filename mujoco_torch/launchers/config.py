CODE_DIRS_TO_MOUNT = [
    '/home/vitchyr/git/classes/spring2018/mujoco-torch',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    # dict(
    #     local_dir='/home/vitchyr/.mujoco/',
    #     mount_point='/root/.mujoco',
    # ),
]
LOCAL_LOG_DIR = '/home/vitchyr/git/railrl/data/local/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/vitchyr/git/railrl/scripts/run_experiment_from_doodad.py'
)
AWS_S3_PATH = 's3://2-12-2017.railrl.vitchyr.rail.bucket/doodad/logs-12-01-2017'

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'anair17/mujoco-torch'
