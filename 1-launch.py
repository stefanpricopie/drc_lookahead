#!/usr/bin/env python3
import datetime
import os
import platform
import subprocess

completed_process = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True,
                                   stdout=subprocess.PIPE, universal_newlines=True)
# Strip newline character at the end
latest_git_hash = completed_process.stdout.strip()
if latest_git_hash is None:
    raise ValueError("Could not obtain the latest git hash")

# Assuming this remains constant as in your bash script
EMAIL = "stefan.pricopie@postgrad.manchester.ac.uk"
N_RUNS = 100


# Get the current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Assuming BINDIR remains constant as in your bash script
BINDIR = os.path.dirname(os.path.abspath(__file__))
# Modify OUTDIR to include both the current date and the latest git hash
OUTDIR = f"results/{current_date}_{latest_git_hash}"
# Ensure the directory exists
os.makedirs(OUTDIR, exist_ok=True)


def qsub_job(runner, configs, jobname, memory=None, ncores=1):
    # Generate the Config array
    config_array = "configs=(" + " \\\n         \"" + "\" \\\n         \"".join(configs) + "\")"

    # Set memory flag based on the input
    if memory is None:
        memory_flag = ""
    elif memory == 512:
        # For 32GB per core
        memory_flag = "#$ -l mem512"
    elif memory == 1500:
        # 1.5TB RAM = 48GB per core, max 32 cores (Skylake CPU). 7 nodes.
        memory_flag = "#$ -l mem1500"
    elif memory == 2000:
        # 2TB RAM   = 64GB per core, max 32 cores (Icelake CPU), 8TB SSD /tmp. 10 nodes.
        memory_flag = "#$ -l mem2000"
    else:
        raise ValueError(f"Memory value {memory} not recognised")

    # Set the number of cores
    if ncores == 1:
        ncores_flag = ""
    elif isinstance(ncores, int) and ncores > 1:
        ncores_flag = f"#$ -pe smp.pe {ncores}"
    else:
        raise ValueError(f"Number of cores {ncores} not recognised")

    cmd = f"""#!/bin/bash --login
#$ -t 1-{len(configs)}  # Using N_RUNS to specify task range
#$ -N {jobname}
{ncores_flag}
# -l s_rt=06:00:00
{memory_flag}
# -M {EMAIL}
# -m as
#$ -cwd
#$ -j y
#$ -o {OUTDIR}

{config_array}

# Use SGE_TASK_ID to access the specific configuration
CONFIG_INDEX=$(($SGE_TASK_ID - 1))  # Arrays are 0-indexed
CONFIG=${{configs[$CONFIG_INDEX]}}

echo "{runner} $CONFIG"
echo "Job: $JOB_ID, Task: $SGE_TASK_ID, Config: $CONFIG"

{BINDIR}/{runner} $CONFIG
"""
    with subprocess.Popen(["qsub", "-v", "PATH"], stdin=subprocess.PIPE) as proc:
        proc.communicate(input=cmd.encode())


def add_config(configurations, problem, dim, algo, switching_cost, xcs, budget, exponent, num_fantasies=None, outdir=OUTDIR):
    begin_seed = 0
    for seed in range(begin_seed, begin_seed + N_RUNS):
        if num_fantasies is None:
            # Initialize base config string
            config = (f"--problem {problem} --dim {dim} --algo {algo} --switching_cost {switching_cost} --xcs {xcs} "
                      f"--seed {seed} --budget {budget} --exponent {exponent} --outdir {outdir}")
        else:
            config = (f"--problem {problem} --dim {dim} --algo {algo} --switching_cost {switching_cost} --xcs {xcs} "
                      f"--seed {seed} --budget {budget} --exponent {exponent} --outdir {outdir} "
                      f"--lookahead_fantasies {' '.join(map(str, num_fantasies))}")

        configurations.append(config)


def run_local(runner, configs):
    for i, config in enumerate(configs):
        cmd = [runner]
        cmd.extend(config.split())
        subprocess.run(cmd)


def run_job(job, memory, ncores):
    runner = "run.py"       # Your Python script for running a single experiment
    configurations = job()  # Generate the configurations for the job

    if platform.system() == "Linux":
        # assert N_RUNS == 50, "N_RUNS must be 50 for cluster runs"
        # split configurations into jobnames and configs
        qsub_job(runner=runner, configs=configurations, jobname=job.__name__, memory=memory, ncores=ncores)
    elif platform.system() == "Darwin":  # macOS is identified as 'Darwin'
        assert N_RUNS == 1, "N_RUNS must be 1 for local runs"
        run_local(runner=f"{os.getcwd()}/{runner}", configs=configurations)


def run():
    configurations = []

    sc = 15
    xcs = 1
    budget = 10

    for objective, dim in [
        ('ackley', 2),
        ('ackley', 5),
        ('shekel', 5),
        ('shekel', 7),
        ('michalewicz', 2),
        ('michalewicz', 5),
        ('dropwave', 2),
        ('shubert', 2),
    ]:
        for algo in [
            "bo",
            "eipu",
            "mean",
            "bobatch",
        ]:
            add_config(configurations, problem=objective, dim=dim, algo=algo, switching_cost=sc,
                       xcs=xcs, exponent='none', budget=budget)

    return configurations


def lookahead():
    configurations = []

    sc = 15
    xcs = 1
    budget = 10

    for objective, dim in [
        ('ackley', 2),
        ('ackley', 5),
        ('shekel', 5),
        ('shekel', 7),
        ('michalewicz', 2),
        ('michalewicz', 5),
        ('dropwave', 2),
        ('shubert', 2),
    ]:
        add_config(configurations, problem=objective, dim=dim, algo="bo_lookahead", switching_cost=sc,
                   xcs=xcs, exponent='none', budget=budget)

    return configurations


def og_lookahead():
    configurations = []

    sc = 0
    xcs = 1
    budget = 20

    for objective, dim in [
        ('ackley', 2),
        ('ackley', 5),
        ('bukin', 2),
        ('dropwave', 2),
        ('eggholder', 2),
        ('rastrigin', 4),
        ('shekel', 5),
        ('shekel', 7),
        ('shubert', 2),
    ]:
        for num_fantasies in [
            [1],
            [1, 1],
            [1, 1, 1],
            [10],
            [10, 5],
            [10, 5, 3],
        ]:
            add_config(configurations, problem=objective, dim=dim, algo="bo_look_og", switching_cost=sc, xcs=xcs,
                       budget=budget, exponent='none',  num_fantasies=num_fantasies)

    return configurations


def multi_lookahead():
    configurations = []

    sc = 0
    xcs = 1
    budget = 20

    for objective, dim in [
        ('ackley', 2),
        ('ackley', 5),
        ('shekel', 5),
        ('shekel', 7),
        ('michalewicz', 2),
        ('michalewicz', 5),
        ('dropwave', 2),
        ('shubert', 2),
    ]:
        for num_fantasies in [
            [1],
            [2, 1],
            [1, 1, 1],
        ]:
            add_config(configurations, problem=objective, dim=dim, algo="bo_lookahead", switching_cost=sc, xcs=xcs,
                       budget=budget, exponent='none', num_fantasies=num_fantasies)

    return configurations


if __name__ == "__main__":
    job_mem_ncores = [
        # (lookahead, 2000, 1),
        # (run, 512, 1),
        (og_lookahead, 2000, 1),
        # (multi_lookahead, 2000, 4),
    ]

    for job, mem, ncore in job_mem_ncores:
        run_job(job, mem, ncore)
