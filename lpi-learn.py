from copy import deepcopy
import logging, os, dill as pickle
import parsl
from parsl import python_app
from itertools import product
import optax

import yaml, mlflow, tempfile, os
import numpy as np, equinox as eqx
import jax
from jax.flatten_util import ravel_pytree


from ml4tpd.parsl_utils import setup_parsl
from adept import ergoExo, utils as adept_utils
from ml4tpd import TPDModule
from ml4tpd.helpers import calc_tpd_broadband_threshold_intensity
from ml4tpd.runners import run_one_val_and_grad, run_adept_fwd

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def train_model(_cfg_path, parsl_provider="gpu", num_nodes=4):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{_cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
    exo = ergoExo()
    modules = exo.setup(orig_cfg, adept_module=TPDModule)
    diff_params, _ = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"], decay_steps=orig_cfg["opt"]["decay_steps"]
    )

    with mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        opt_state = opt.init(eqx.filter(diff_params, eqx.is_array))  # initialize the optimizer state

        epoch_loss_mean = _train_(0, orig_cfg, parsl_provider, num_nodes, exo.mlflow_run_id, modules, opt, opt_state)

    return epoch_loss_mean


def initialize_training_data(cfg):
    nt = cfg["training data"]["num_temperatures"]
    ngsl = cfg["training data"]["num_gradient_scale_lengths"]
    bandwidth = cfg["drivers"]["E0"]["delta_omega_max"] * 2
    temperatures = np.round(np.linspace(2000, 4000, nt), 2)
    gradient_scale_lengths = np.round(np.linspace(200, 600, ngsl), 2)
    all_hps = []

    for te, gsl in product(temperatures, gradient_scale_lengths):
        all_hps.append(
            (te, gsl, round(calc_tpd_broadband_threshold_intensity(te / 1000, gsl, 0.351, bandwidth) * 1e14, 2))
        )
    return all_hps


def _train_(
    start_epoch,
    orig_cfg,
    parsl_provider,
    num_nodes,
    parent_run_id,
    modules,
    opt,
    opt_state,
):
    parsl_config = setup_parsl(parsl_provider, 4, nodes=num_nodes, walltime="8:00:00")
    parsl_run_one_val_and_grad = python_app(run_one_val_and_grad)
    parsl_run_fwd = python_app(run_adept_fwd)
    # parsl_run_one_val_and_grad = run_one_val_and_grad
    diff_params, static_params = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())
    num_batches = 1  # len(all_hps) // batch_size
    rng = np.random.default_rng()
    all_hps = initialize_training_data(cfg=orig_cfg)
    base_cfg = deepcopy(orig_cfg)
    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
        os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
        with parsl.load(parsl_config):
            for i in range(start_epoch, 200):  # 1000 epochs
                epoch_loss = []
                epoch_gradnorm = []
                rng.shuffle(all_hps)
                batch_size = num_nodes * 4
                for j in range(num_batches):
                    with open(opt_state_path := os.path.join(td, f"opt-state-epoch={i}-batch-{j}.pkl"), "wb") as fi:
                        pickle.dump(opt_state, fi)

                    modules["laser"].save(
                        module_path := os.path.join(td, "weights-history", f"weights-e{i:02d}-b{j:02d}.eqx")
                    )

                    mlflow.log_artifact(opt_state_path)
                    mlflow.log_artifact(module_path)

                    step = i * num_batches + j
                    training_data = all_hps[j * batch_size : (j + 1) * batch_size]

                    orig_cfg["drivers"]["E0"]["file"] = module_path

                    val_and_grads = []
                    for k in range(batch_size):
                        _training_data = training_data[k]
                        print(f"{i=}, {j=}, {k=} -- _Training Data: {_training_data}")
                        export = np.random.choice([True, False], p=[0.25, 0.75])

                        tt = _training_data[0]
                        gsl = _training_data[1]
                        # intensity_factor = _training_data[2]

                        # orig_cfg["units"]["intensity factor"] = f"{intensity_factor:.3f}"
                        base_intensity = _training_data[2]
                        orig_cfg["units"]["reference electron temperature"] = f"{_training_data[0]:.3f} eV"
                        orig_cfg["density"]["gradient scale length"] = f"{_training_data[1]:.3f} um"

                        factor = rng.uniform(1.1, 1.3)

                        intensity = factor * base_intensity
                        orig_cfg["units"]["intensity factor"] = f"{factor:.3f}"
                        orig_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
                        orig_cfg["grid"]["dt"] = f"{np.random.uniform(6, 8):.3f} fs"
                        orig_cfg["mlflow"]["run"] = (
                            f"epoch-{i}-batch-{j}-temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
                        )
                        orig_cfg["mlflow"]["export"] = str(export)

                        with open(run_cfg_path := os.path.join(td, f"config-{i=}-{j=}-{k=}.yaml"), "w") as fi:
                            yaml.dump(orig_cfg, fi)

                        val_and_grads.append(
                            parsl_run_one_val_and_grad(
                                parent_run_id=parent_run_id, _run_cfg_path=run_cfg_path, export=export
                            )
                        )

                    vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                    validation_losses = np.array([v for v, _ in vgs])
                    validation_losses = np.nan_to_num(validation_losses, nan=30.0, posinf=30.0, neginf=30.0)
                    validation_losses = np.where(validation_losses > 30.0, 30, validation_losses)
                    val = np.mean(validation_losses)

                    avg_grad = adept_utils.all_reduce_gradients([g for _, g in vgs], batch_size)

                    flat_grad, _ = ravel_pytree(avg_grad["laser"])
                    mlflow.log_metrics(
                        {"batch grad norm": float(np.linalg.norm(flat_grad)), "batch loss": float(val)}, step=step
                    )
                    updates, opt_state = opt.update(avg_grad["laser"], opt_state, diff_params)
                    diff_params = eqx.apply_updates(diff_params, updates)
                    modules["laser"] = eqx.combine(diff_params, static_params)
                    epoch_loss.append(val)
                    epoch_gradnorm.append(np.linalg.norm(flat_grad))

                mlflow.log_metrics(
                    {
                        "epoch loss": (epoch_loss_mean := float(np.mean(epoch_loss))),
                        "epoch grad norm": float(np.mean(epoch_gradnorm)),
                    },
                    step=i,
                )

                if i % 3 == 0:
                    latest_weights_path = os.path.join(td, "weights-history", f"weights-e{i:02d}-latest.eqx")
                    modules["laser"].save(latest_weights_path)
                    validation_losses = []
                    factor = 1.25
                    for idx, (tt, gsl, base_intensity) in enumerate(all_hps):
                        intensity = factor * base_intensity
                        validation_cfg = deepcopy(base_cfg)
                        validation_cfg["save"]["fields"]["t"]["dt"] = "0.25 ps"
                        validation_cfg["drivers"]["E0"]["file"] = latest_weights_path
                        validation_cfg["units"]["reference electron temperature"] = f"{tt:.3f} eV"
                        validation_cfg["density"]["gradient scale length"] = f"{gsl:.3f} um"
                        validation_cfg["units"]["intensity factor"] = factor
                        validation_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
                        validation_cfg["mlflow"]["run"] = f"epoch-{i}-validation-temperature={tt:.1f}-gsl={gsl:.1f}"
                        validation_cfg["mlflow"]["export"] = "True"
                        with open(
                            validation_cfg_path := os.path.join(td, f"validation-config-epoch={i}-hp={idx}.yaml"), "w"
                        ) as fi:
                            yaml.dump(validation_cfg, fi)

                        validation_losses.append(parsl_run_fwd(validation_cfg_path, parent_run_id=parent_run_id))

                    validation_losses = np.array([vl.result() for vl in validation_losses])
                    # vals = np.array([v for v, _ in vgs])
                    vals = np.nan_to_num(validation_losses, nan=30.0, posinf=30.0, neginf=30.0)
                    vals = np.where(vals > 30.0, 30, vals)
                    val = np.mean(vals)

                    # if validation_losses:
                    mlflow.log_metric("val loss", float(val), step=i)

    return epoch_loss_mean


def initialize_resume(run_id: str, tmpdir: str) -> str:
    """
    - Download config using mlflow download artifact
    - find latest epoch and batch number by checking the logged metrics
    - download weights and opt state
    - continue training

    :param run_id: Description
    :type run_id: str
    :return: Description
    :rtype: str
    """

    # Download the config file
    cfg_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=tmpdir)

    # Find latest epoch and batch number
    all_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    all_weights = []
    for artifact in all_artifacts:
        if artifact.path.startswith("weights-e"):
            all_weights.append(artifact.path)

    # the weights will have a name like weights-e02-b05.eqx
    # we want to find e02-b05.eqx in that case
    # sort the weights to find the latest epoch and latest batch

    epochs = set()
    for weight in all_weights:
        epoch_str = weight.split("-")[1]  # e02
        epoch_num = int(epoch_str[1:])  # 02 -> 2
        epochs.add(epoch_num)
    latest_epoch = max(epochs)

    all_weights = []
    for artifact in all_artifacts:
        if artifact.path.startswith(f"weights-e{latest_epoch:02d}-b"):
            all_weights.append(artifact.path)

    batches = set()
    for weight in all_weights:
        batch_str = weight.split("-")[2]  # b05.eqx
        batch_num = int(batch_str[1 : batch_str.index(".")])  # 05 -> 5
        batches.add(batch_num)
    latest_batch = max(batches)

    latest_weights = f"weights-e{latest_epoch:02d}-b{latest_batch:02d}.eqx"

    # download weights
    weights_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=latest_weights, dst_path=tmpdir)
    opt_state_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"opt-state-epoch={latest_epoch}-batch-0.pkl", dst_path=tmpdir
    )

    print(f"Resuming from epoch {latest_epoch} with weights {weights_path} and opt state {opt_state_path}")

    return cfg_path, weights_path, opt_state_path, latest_epoch


def resume_train_model(cfg_path, run_id, start_epoch, weights_path, opt_state_path, parsl_provider="gpu", num_nodes=4):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["drivers"]["E0"]["file"] = str(weights_path)

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

    copy_cfg = deepcopy(orig_cfg)
    exo = ergoExo(mlflow_run_id=run_id)
    modules = exo._setup_(copy_cfg, td=os.path.dirname(weights_path), adept_module=TPDModule, log=False)

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"], decay_steps=orig_cfg["opt"]["decay_steps"]
    )

    with mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        with open(opt_state_path, "rb") as fi:
            opt_state = pickle.load(fi)

        epoch_loss_mean = _train_(
            start_epoch,
            orig_cfg,
            parsl_provider,
            num_nodes,
            exo.mlflow_run_id,
            modules,
            opt,
            opt_state,
        )

    return epoch_loss_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    parser.add_argument("--run_id", type=str, default=None, help="The MLflow run ID to use for resuming")
    parser.add_argument("--provider", type=str, default="gpu", help="The Parsl provider to use")
    parser.add_argument("--nodes", type=int, default=4, help="The number of nodes to use")

    args = parser.parse_args()
    cfg_path = args.config
    parsl_provider = args.provider
    num_nodes = args.nodes
    resume_run_id = args.run_id

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if resume_run_id is not None:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            cfg_path, weights_path, opt_state_path, latest_epoch = initialize_resume(resume_run_id, td)
            resume_train_model(
                cfg_path,
                resume_run_id,
                latest_epoch,
                weights_path,
                opt_state_path,
                parsl_provider=parsl_provider,
                num_nodes=num_nodes,
            )
    else:
        train_model(cfg_path, parsl_provider=parsl_provider, num_nodes=num_nodes)
