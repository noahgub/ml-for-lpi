import os
from parsl.addresses import address_by_hostname  # or address_by_interface

def setup_parsl(parsl_provider="local", num_gpus=4, nodes=1, walltime="00:30:00", label="tpd"):
    from parsl.config import Config
    from parsl.providers import SlurmProvider, LocalProvider
    from parsl.executors import HighThroughputExecutor

    if parsl_provider == "local":
        if nodes == 1:
            this_provider = LocalProvider
            provider_args = get_singlenode_local_provider_args()
            htex = HighThroughputExecutor(
                available_accelerators=num_gpus,
                label=label,
                provider=this_provider(**provider_args),
                cpu_affinity="block",
                address=address_by_hostname(),
                worker_port_range=(50000, 60000),
            )

        else:
            if num_gpus > 0:
                this_provider = LocalProvider
                provider_args = get_multinode_local_provider_args(nodes)

                htex = HighThroughputExecutor(
                    available_accelerators=num_gpus,
                    label=label,
                    provider=this_provider(**provider_args),
                    max_workers_per_node=4,
                    cpu_affinity="block",
                )
            else:
                this_provider = LocalProvider
                provider_args = get_multinode_cpu_local_provider_args(nodes)
                htex = HighThroughputExecutor(
                    available_accelerators=0,
                    label=label,
                    provider=this_provider(**provider_args),
                    max_workers_per_node=4,
                    cpu_affinity="block",
                )

    elif parsl_provider == "gpu":
        this_provider = SlurmProvider
        provider_args = get_gpu_provider_args(nodes, walltime)

        htex = HighThroughputExecutor(
            available_accelerators=4, label=label, provider=this_provider(**provider_args), cpu_affinity="block",
            address=address_by_hostname(),
            worker_port_range=(50000, 60000),
        )

    return Config(executors=[htex], retries=0)


def get_singlenode_local_provider_args():
    provider_args = dict(
        worker_init=f"  source /global/homes/n/ngub/srs/.venv/bin/activate; \
                        export PYTHONPATH=$PYTHONPATH:/global/homes/n/ngub/srs/ml-for-lpi; \
                        export BASE_TEMPDIR='/pscratch/sd/n/ngub/tmp/'; \
                        export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'; \
                        export MLFLOW_TRACKING_USERNAME={os.environ['MLFLOW_TRACKING_USERNAME']}; \
                        export MLFLOW_TRACKING_PASSWORD={os.environ['MLFLOW_TRACKING_PASSWORD']};",
        init_blocks=1,
        max_blocks=1,
        nodes_per_block=1,
    )

    return provider_args

def get_multinode_cpu_local_provider_args(nodes):
    from parsl.launchers import SrunLauncher

    provider_args = dict(
        worker_init=f"source /global/common/software/m4490/archis/venvs/ml-for-lpi/bin/activate; \
                        export PYTHONPATH=$PYTHONPATH:/global/homes/a/archis/ml-for-lpi; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'; \
                        module load matlab; \
                        export MLFLOW_TRACKING_USERNAME={os.environ['MLFLOW_TRACKING_USERNAME']}; \
                        export MLFLOW_TRACKING_PASSWORD={os.environ['MLFLOW_TRACKING_PASSWORD']};",
        nodes_per_block=1,
        launcher=SrunLauncher(overrides="-c 64"),
        cmd_timeout=120,
        init_blocks=1,
        max_blocks=nodes,
    )

    return provider_args

def get_multinode_local_provider_args(nodes):
    from parsl.launchers import SrunLauncher

    provider_args = dict(
        worker_init=f"source /global/homes/n/ngub/srs/.venv/bin/activate; \
                        export PYTHONPATH=$PYTHONPATH:/global/homes/n/ngub/srs/ml-for-lpi; \
                        export BASE_TEMPDIR='/pscratch/sd/n/ngub/tmp/'; \
                        export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'; \
                        export MLFLOW_TRACKING_USERNAME={os.environ['MLFLOW_TRACKING_USERNAME']}; \
                        export MLFLOW_TRACKING_PASSWORD={os.environ['MLFLOW_TRACKING_PASSWORD']};",
        nodes_per_block=1,
        launcher=SrunLauncher(overrides="-c 32 --gpus-per-node 4"),
        cmd_timeout=120,
        init_blocks=1,
        max_blocks=nodes,
    )

    return provider_args


def get_gpu_provider_args(nodes, walltime):
    from parsl.launchers import SrunLauncher

    sched_args = ["#SBATCH -C gpu&hbm80g", "#SBATCH --qos=regular"]
    provider_args = dict(
        partition=None,
        account="m5057_g",
        scheduler_options="\n".join(sched_args),
        worker_init=f"export SLURM_CPU_BIND='cores';\
                    export PYTHONPATH=$PYTHONPATH:/global/homes/n/ngub/srs/ml-for-lpi; \
                    source /global/homes/n/ngub/srs/.venv/bin/activate; \
                    export BASE_TEMPDIR='/pscratch/sd/n/ngub/tmp/'; \
                    export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'; \
                    export MLFLOW_TRACKING_USERNAME={os.environ['MLFLOW_TRACKING_USERNAME']}; \
                    export MLFLOW_TRACKING_PASSWORD={os.environ['MLFLOW_TRACKING_PASSWORD']};",
        launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 128"),
        walltime=walltime,
        cmd_timeout=120,
        nodes_per_block=1,
        # init_blocks=1,
        max_blocks=nodes,
    )

    return provider_args