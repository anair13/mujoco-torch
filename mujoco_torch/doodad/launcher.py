import json
import os
import os.path as osp
import pickle
import random
import sys
import time
import uuid
from collections import namedtuple

import __main__ as main
import datetime
import dateutil.tz
import joblib
import numpy as np

import mujoco_torch.utils.pythonplusplus as ppp
from mujoco_torch.utils.logger import logger as default_logger
from mujoco_torch.doodad import config

GitInfo = namedtuple('GitInfo', ['code_diff', 'commit_hash', 'branch_name'])


ec2_okayed = False
gpu_ec2_okayed = False

try:
    import doodad.mount as mount
    from doodad.utils import REPO_DIR
    CODE_MOUNTS = [
        mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    ]
    for code_dir in config.CODE_DIRS_TO_MOUNT:
        CODE_MOUNTS.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))

    NON_CODE_MOUNTS = []
    for non_code_mapping in config.DIR_AND_MOUNT_POINT_MAPPINGS:
        NON_CODE_MOUNTS.append(mount.MountLocal(**non_code_mapping))
except ImportError:
    print("doodad not detected")

target_mount = None


def run_experiment(
        method_call,
        mode='local',
        exp_prefix='default',
        seed=None,
        variant=None,
        exp_id=0,
        unique_id=None,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        base_log_dir=None,
        local_input_dir_to_mount_point_dict=None,  # TODO(vitchyr): test this
        # Settings for EC2 only
        sync_interval=180,
        region='us-east-1',
        instance_type=None,
        spot_price=None,
        logger=default_logger,
):
    """
    Usage:

    ```
    def foo(variant):
        x = variant['x']
        y = variant['y']
        logger.log("sum", x+y)

    variant = {
        'x': 4,
        'y': 3,
    }
    run_experiment(foo, variant, exp_prefix="my-experiment")
    ```

    Results are saved to
    `base_log_dir/<date>-my-experiment/<date>-my-experiment-<unique-id>`

    By default, the base_log_dir is determined by
    `config.LOCAL_LOG_DIR/`

    :param method_call: a function that takes in a dictionary as argument
    :param mode: A string:
     - 'local'
     - 'local_docker'
     - 'ec2'
     - 'here_no_doodad': Run without doodad call
    :param exp_prefix: name of experiment
    :param seed: Seed for this specific trial.
    :param variant: Dictionary
    :param exp_id: One experiment = one variant setting + multiple seeds
    :param unique_id: If not set, the unique id is generated.
    :param prepend_date_to_exp_prefix: If False, do not prepend the date to
    the experiment directory.
    :param use_gpu:
    :param snapshot_mode: See rllab.logger
    :param snapshot_gap: See rllab.logger
    :param base_log_dir: Will over
    :param sync_interval: How often to sync s3 data (in seconds).
    :param local_input_dir_to_mount_point_dict: Dictionary for doodad.
    :return:
    """
    try:
        import doodad
        import doodad.mode
    except ImportError:
        print("Doodad not set up! Running experiment here.")
        mode = 'here_no_doodad'
    global ec2_okayed
    global gpu_ec2_okayed
    global target_mount

    """
    Sanitize inputs as needed
    """
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    for key, value in ppp.recursive_items(variant):
        # This check isn't really necessary, but it's to prevent myself from
        # forgetting to pass a variant through dot_map_dict_to_nested_dict.
        if "." in key:
            raise Exception(
                "Variants should not have periods in keys. Did you mean to "
                "convert {} into a nested dictionary?".format(key)
            )
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['unique_id'] = str(unique_id)
    variant['exp_prefix'] = str(exp_prefix)
    variant['instance_type'] = str(instance_type)

    try:
        import git
        repo = git.Repo(os.getcwd())
        git_info = GitInfo(
            code_diff=repo.git.diff(None),
            commit_hash=repo.head.commit.hexsha,
            branch_name=repo.active_branch.name,
        )
    except ImportError:
        git_info = None

    run_experiment_kwargs = dict(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_info=git_info,
        script_name=main.__file__,
        logger=logger,
    )
    if mode == 'here_no_doodad':
        run_experiment_kwargs['base_log_dir'] = base_log_dir
        return run_experiment_here(
            method_call,
            **run_experiment_kwargs
        )

    """
    Safety Checks
    """

    if mode == 'ec2':
        if not ec2_okayed and not query_yes_no(
                "EC2 costs money. Are you sure you want to run?"
        ):
            sys.exit(1)
        if not gpu_ec2_okayed and use_gpu:
            if not query_yes_no(
                    "EC2 is more expensive with GPUs. Confirm?"
            ):
                sys.exit(1)
            gpu_ec2_okayed = True
        ec2_okayed = True

    """
    GPU vs normal configs
    """
    if use_gpu:
        docker_image = config.GPU_DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = config.GPU_INSTANCE_TYPE
        else:
            assert instance_type[0] == 'g'
        if spot_price is None:
            spot_price = config.GPU_SPOT_PRICE
    else:
        docker_image = config.DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = config.INSTANCE_TYPE
        if spot_price is None:
            spot_price = config.SPOT_PRICE

    """
    Get the mode
    """
    mode_kwargs = {}
    if use_gpu:
        image_id = config.REGION_TO_GPU_AWS_IMAGE_ID[region]
        if region == 'us-east-1':
            mode_kwargs['extra_ec2_instance_kwargs'] = dict(
                Placement=dict(
                    AvailabilityZone='us-east-1b',
                ),
            )
    else:
        image_id = None
    if hasattr(config, "AWS_S3_PATH"):
        aws_s3_path = config.AWS_S3_PATH
    else:
        aws_s3_path = None
    if "run_id" in variant and variant["run_id"] is not None:
        run_id, exp_id = variant["run_id"], variant["exp_id"]
        s3_log_name = "run{}/id{}".format(run_id, exp_id)
    else:
        s3_log_name = "{}-id{}-s{}".format(exp_prefix, exp_id, seed)
    mode_str_to_doodad_mode = {
        'local': doodad.mode.Local(),
        'local_docker': doodad.mode.LocalDocker(
            image=docker_image,
            gpu=use_gpu,
        ),
        'ec2': doodad.mode.EC2AutoconfigDocker(
            image=docker_image,
            image_id=image_id,
            region=region,
            instance_type=instance_type,
            spot_price=spot_price,
            s3_log_prefix=exp_prefix,
            s3_log_name=s3_log_name,
            gpu=use_gpu,
            aws_s3_path=aws_s3_path,
            **mode_kwargs
        ),
    }

    """
    Get the mounts
    """
    mounts = create_mounts(
        base_log_dir=base_log_dir,
        mode=mode,
        sync_interval=sync_interval,
        local_input_dir_to_mount_point_dict=local_input_dir_to_mount_point_dict,
    )

    """
    Get the outputs
    """
    if mode == 'ec2':
        # Ignored since I'm setting the snapshot dir directly
        base_log_dir_for_script = None
        # The snapshot dir needs to be specified for S3 because S3 will
        # automatically create the experiment director and sub-directory.
        snapshot_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    elif mode == 'local':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode == 'local_docker':
        base_log_dir_for_script = config.OUTPUT_DIR_FOR_DOODAD_TARGET
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode == 'here_no_doodad':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    run_experiment_kwargs['base_log_dir'] = base_log_dir_for_script
    target_mount = doodad.launch_python(
        target=config.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH,
        mode=mode_str_to_doodad_mode[mode],
        mount_points=mounts,
        args={
            'method_call': method_call,
            'output_dir': snapshot_dir_for_script,
            'run_experiment_kwargs': run_experiment_kwargs,
            'mode': mode,
        },
        use_cloudpickle=True,
        target_mount=target_mount,
    )


def create_mounts(
        mode,
        base_log_dir,
        sync_interval=180,
        local_input_dir_to_mount_point_dict=None,
):
    if local_input_dir_to_mount_point_dict is None:
        local_input_dir_to_mount_point_dict = {}
    else:
        raise NotImplementedError("TODO(vitchyr): Implement this")

    mounts = [m for m in CODE_MOUNTS]
    for dir, mount_point in local_input_dir_to_mount_point_dict.items():
        mounts.append(mount.MountLocal(
            local_dir=dir,
            mount_point=mount_point,
            pythonpath=False,
        ))

    if mode != 'local':
        for m in NON_CODE_MOUNTS:
            mounts.append(m)

    if mode == 'ec2':
        output_mount = mount.MountS3(
            s3_path='',
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            sync_interval=sync_interval,
        )
    elif mode == 'local':
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=None,  # For purely local mode, skip mounting.
            output=True,
        )
    elif mode == 'local_docker':
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    mounts.append(output_mount)
    return mounts


def save_experiment_data(dictionary, log_dir):
    with open(log_dir + '/experiment.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resume_torch_algorithm(variant):
    from railrl.torch import pytorch_util as ptu
    load_file = variant.get('params_file', None)
    if load_file is not None and osp.exists(load_file):
        data = joblib.load(load_file)
        algorithm = data['algorithm']
        epoch = data['epoch']+1
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epoch + 1)


def continue_experiment(load_experiment_dir, resume_function):
    path = os.path.join(load_experiment_dir, 'experiment.pkl')
    if osp.exists(path):
        data = joblib.load(path)
        mode = data['mode']
        exp_prefix = data['exp_prefix']
        variant = data['variant']
        variant['params_file'] = load_experiment_dir + '/extra_data.pkl' # load from snapshot directory
        exp_id = data['exp_id']
        seed = data['seed']
        use_gpu = data['use_gpu']
        snapshot_mode = data['snapshot_mode']
        snapshot_gap = data['snapshot_gap']
        diff_string = data['diff_string']
        commit_hash = data['commit_hash']
        base_log_dir = data['base_log_dir']
        log_dir = data['log_dir']
        if mode == 'local':
            run_experiment_here(
                resume_function,
                variant=variant,
                exp_prefix=exp_prefix,
                exp_id=exp_id,
                seed=seed,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
                code_diff=diff_string,
                commit_hash=commit_hash,
                base_log_dir=base_log_dir,
                log_dir=log_dir,
            )
    else:
        raise Exception('invalid experiment_file')


def continue_experiment_simple(load_experiment_dir, resume_function):
    path = os.path.join(load_experiment_dir, 'experiment.pkl')
    data = joblib.load(path)
    run_experiment_here_kwargs = data['run_experiment_here_kwargs']
    run_experiment_here_kwargs['log_dir'] = load_experiment_dir
    run_experiment_here_kwargs['variant']['params_file'] = (
        os.path.join(load_experiment_dir, 'extra_data.pkl')
    )
    run_experiment_here(
        resume_function,
        **run_experiment_here_kwargs
    )


def resume_torch_algorithm_simple(variant):
    from railrl.torch import pytorch_util as ptu
    load_file = variant.get('params_file', None)
    if load_file is not None and osp.exists(load_file):
        data = joblib.load(load_file)
        algorithm = data['algorithm']
        epoch = data['epoch']+1
        if ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epoch + 1)


def run_experiment_here(
        experiment_function,
        variant=None,
        exp_id=0,
        seed=0,
        use_gpu=True,
        # Logger params:
        exp_prefix="default",
        snapshot_mode='last',
        snapshot_gap=1,
        git_info=None,
        script_name=None,
        base_log_dir=None,
        log_dir=None,
        logger=default_logger,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    if variant is None:
        variant = {}
    variant['exp_id'] = str(exp_id)

    if seed is None and 'seed' not in variant:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    reset_execution_environment(logger=logger)

    actual_log_dir = setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        base_log_dir=base_log_dir,
        log_dir=log_dir,
        git_info=git_info,
        script_name=script_name,
        logger=logger,
    )

    set_seed(seed)
    # set_gpu_mode(use_gpu)

    run_experiment_here_kwargs = dict(
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        exp_prefix=exp_prefix,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_info=git_info,
        script_name=script_name,
        base_log_dir=base_log_dir,
    )
    save_experiment_data(
        dict(
            run_experiment_here_kwargs=run_experiment_here_kwargs
        ),
        actual_log_dir
    )
    return experiment_function(variant)


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None, variant=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    if variant and "run_id" in variant and variant["run_id"] is not None:
        run_id, exp_id = variant["run_id"], variant["exp_id"]
        exp_name = "run{}/id{}".format(run_id, exp_id)
    else:
        exp_name = create_exp_name(exp_prefix, exp_id=exp_id, seed=seed)

    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix="default",
        exp_id=0,
        seed=0,
        variant=None,
        base_log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        git_info=None,
        script_name=None,
        logger=default_logger,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

        based_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.

    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param git_info:
    :param script_name: If set, save the script name to this.
    :return:
    """
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, exp_id=exp_id, seed=seed,
                                           base_log_dir=base_log_dir, variant=variant)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(ppp.dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if git_info is not None:
        code_diff, commit_hash, branch_name = git_info
        if code_diff is not None:
            with open(osp.join(log_dir, "code.diff"), "w") as f:
                f.write(code_diff)
        with open(osp.join(log_dir, "git_info.txt"), "w") as f:
            f.write("git hash: {}".format(commit_hash))
            f.write('\n')
            f.write("git branch name: {}".format(branch_name))
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def reset_execution_environment(logger=default_logger):
    """
    Call this between calls to separate experiments.
    :return:
    """
    import importlib
    importlib.reload(logger)


def create_run_experiment_multiple_seeds(n_seeds, experiment, **kwargs):
    """
    Run a function multiple times over different seeds and return the average
    score.
    :param n_seeds: Number of times to run an experiment.
    :param experiment: A function that returns a score.
    :param kwargs: keyword arguements to pass to experiment.
    :return: Average score across `n_seeds`.
    """

    def run_experiment_with_multiple_seeds(variant):
        seed = int(variant['seed'])
        scores = []
        for i in range(n_seeds):
            variant['seed'] = str(seed + i)
            scores.append(run_experiment(
                experiment,
                variant=variant,
                exp_id=i,
                mode='here',
                **kwargs
            ))
        return np.mean(scores)

    return run_experiment_with_multiple_seeds


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
