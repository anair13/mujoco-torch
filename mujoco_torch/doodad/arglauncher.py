"""Wraps launcher_util to make launching experiments one step easier - Ashvin
- Names experiments based on the running filename
- Adds some modes like --1 to run only one variant of a set for testing
- Control the GPU used and other experiment attributes through command line args
"""

from mujoco_torch.doodad import launcher as lu
import argparse # TODO: migrate to argparse if necessary
import sys
from multiprocessing import Process, Pool
import pdb

def run_variants(experiment, variants, run_id=0):
    # preprocess
    for i, v in enumerate(variants):
        v["exp_id"] = i
        v["run_id"] = run_id
        process_variant_cmd(v)

    if "--variants" in sys.argv: # takes either 3-7 or 3,6,7,8,10 as next arg
        i = sys.argv.index("--variants")
        val = sys.argv[i+1]
        ids = []
        if "," in val:
            ids = map(int, val.split(','))
        elif "-" in val:
            start, end = map(int, val.split(','))
            ids = range(start, end)
        else:
            ids = [int(val), ]
        new_variants = []
        for v in variants:
            if v["exp_id"] in ids:
                new_variants.append(v)
        variants = new_variants
    if "--1" in sys.argv: # only run the first experiment for testing
        variants = variants[:1]

    print("Running", len(variants), "variants")

    # run
    parallel = variants[0]["parallel"]
    if parallel:
        parallel_run(experiment, variants, parallel)
    else:
        for variant in variants:
            run_variant(experiment, variant)

    print("Running", len(variants), "variants")

def run_variant(experiment, variant):
    lu.run_experiment(
        experiment,
        variant=variant,
        # run_id=variant["run_id"],
        mode=variant["mode"],
        exp_prefix=variant["exp_prefix"],
        exp_id=variant["exp_id"],
        instance_type=variant["instance_type"],
        # use_gpu=variant["use_gpu"],
        snapshot_mode=variant["snapshot_mode"],
        snapshot_gap=variant["snapshot_gap"],
        base_log_dir=variant["base_log_dir"],
        prepend_date_to_exp_prefix=False,
        spot_price=variant["spot_price"],
    )

def parallel_run(experiment, variants, n_p):
    i = 0
    # import pdb; pdb.set_trace()
    while i < len(variants):
        prs = []
        for p in range(n_p):
            if i < len(variants):
                v = variants[i]
                v["gpu_id"] = v["gpus"][p]
                pr = Process(target=run_variant, args=(experiment, v))
                prs.append(pr)
                pr.start()
            i += 1
        for pr in prs:
            pr.join()

def process_variant_cmd(variant):
    # assumes some format of arguments passed in for quick interface
    if "--sync" in sys.argv:
        variant["sync"] = True
    if "--nosync" in sys.argv:
        variant["sync"] = False

    if "--render" in sys.argv:
        variant["algo_params"]["render"] = True
    if "--norender" in sys.argv:
        variant["algo_params"]["render"] = False

    if "--ec2" in sys.argv:
        variant["mode"] = "ec2"
    if "--local" in sys.argv:
        variant["mode"] = "local"
    if "--localdocker" in sys.argv:
        variant["mode"] = "local_docker"

    if "--parallel" in sys.argv:
        i = sys.argv.index("--parallel")
        parallel = int(sys.argv[i+1])
        variant["parallel"] = parallel
        if "--gpus" in sys.argv:
            i = sys.argv.index("--gpus")
            variant["gpus"] = list(map(int, sys.argv[i+1].split(",")))
            variant["use_gpu"] = True
        else:
            variant["gpus"] = list(range(parallel))
    else:
        variant["parallel"] = False

    if "--snapshot" in sys.argv:
        variant["snapshot_mode"] = 'gap_and_last'
        variant["snapshot_gap"] = 20
    else:
        variant["snapshot_mode"] = 'last'
        variant["snapshot_gap"] = 1

    if "--gpu_id" in sys.argv:
        i = sys.argv.index("--gpu_id")
        variant["gpu_id"] = int(sys.argv[i+1])
        variant["use_gpu"] = True
    if "--gpu" in sys.argv:
        variant["use_gpu"] = True
        variant["instance_type"] = None
    if "use_gpu" in variant and variant["use_gpu"] and "gpu_id" not in variant:
        variant["gpu_id"] = 0

    if "--run" in sys.argv:
        i = sys.argv.index("--run")
        variant["run_id"] = int(sys.argv[i+1])

    if "exp_prefix" not in variant:
        s = "experiments/"
        n = len(s)
        assert sys.argv[0][:n] == s
        variant["exp_prefix"] = sys.argv[0][n:-3]

    if "instance_type" not in variant:
        variant["instance_type"] = "c4.8xlarge"
    if "use_gpu" not in variant:
        variant["use_gpu"] = None

    if "base_log_dir" not in variant:
        variant["base_log_dir"] = None
    if "--mac" in sys.argv:
        variant["base_log_dir"] = "/Users/ashvin/data/s3doodad/"

    variant["spot_price"] = {'c4.large': 0.1, 'c4.xlarge': 0.2, 'c4.2xlarge': 0.4, 'm4.large': 0.1, 'm4.xlarge': 0.2, 'm4.2xlarge': 0.4, 'c4.8xlarge': 2.0, 'c4.4xlarge': 1.0}[variant["instance_type"]]
