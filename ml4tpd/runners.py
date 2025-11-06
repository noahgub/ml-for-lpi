from typing import Dict


def clear_xla_cache():
    """
    Clear XLA compilation cache to prevent memory buildup during optimization.

    This function attempts to clear JAX's compilation cache using multiple methods
    to ensure memory is freed during long optimization runs.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        import jax

        # Clear the compilation cache
        jax.clear_caches()
        logger.info("XLA cache cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear XLA cache: {e}")

    # try:
    #     # Force garbage collection after clearing cache
    #     import gc

    #     gc.collect()
    #     logger.debug("Garbage collection completed after cache clear")
    # except Exception as e:
    #     logger.warning(f"Garbage collection failed: {e}")


def _execute_adept_forward(cfg, parent_run_id):
    from adept import ergoExo
    from ml4tpd import TPDModule

    exo = ergoExo(parent_run_id=parent_run_id, mlflow_nested=True)
    modules = exo.setup(cfg, adept_module=TPDModule)
    run_output, _, _ = exo(modules)
    val = float(run_output[0])
    clear_xla_cache()
    return val


def run_adept_fwd(_cfg_path, parent_run_id=None, seed=None, run_name=None):
    """
    Run a single ADEPT forward pass for the provided configuration.

    Args:
        _cfg_path: Path to YAML configuration.
        parent_run_id: Optional MLflow parent run used for nested logging.
        seed: Optional seed override for the driver phases.
        run_name: Optional MLflow run name override.
        log_params: When True, emit configuration parameters to MLflow.

    Returns:
        The loss value from the forward evaluation.
    """
    import yaml, mlflow
    from adept import utils as adept_utils

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    if seed is not None:
        cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(seed)
    if run_name is not None:
        cfg["mlflow"]["run"] = run_name

    if parent_run_id is None:
        active_run = mlflow.active_run()
        with mlflow.start_run(run_name=cfg["mlflow"]["run"], nested=active_run is not None) as run:
            # if log_params:
            #     adept_utils.log_params(cfg)
            val = _execute_adept_forward(cfg, parent_run_id=run.info.run_id)
            mlflow.log_metric("loss", val)
        return val
    else:
        # if log_params:
        #     adept_utils.log_params(cfg)
        return _execute_adept_forward(cfg, parent_run_id=parent_run_id)


def run_adept_fwd_ensemble(_cfg_path, num_seeds=8):
    import yaml, mlflow

    from adept import utils as adept_utils
    import numpy as np

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    active_run = mlflow.active_run()
    with mlflow.start_run(run_name=cfg["mlflow"]["run"], nested=active_run is not None) as parent_run:
        adept_utils.log_params(cfg)
        vals = []
        for i in range(num_seeds):
            seed = int(np.random.randint(0, 2**10))
            run_name = f"seed-{i}"
            val = run_adept_fwd(
                _cfg_path=_cfg_path,
                parent_run_id=parent_run.info.run_id,
                seed=seed,
                run_name=run_name,
                log_params=False,
            )
            vals.append(val)
        mean_val = float(np.mean(vals))
        mlflow.log_metric("loss", mean_val)
    return mean_val


def run_one_val_and_grad(parent_run_id, _run_cfg_path, export=False):
    import os, yaml
    from equinox import partition

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from adept import ergoExo
    from ml4tpd import TPDModule

    with open(_run_cfg_path, "r") as fi:
        _run_cfg = yaml.safe_load(fi)

    exo = ergoExo(parent_run_id=parent_run_id, mlflow_nested=True)
    modules = exo.setup(_run_cfg, adept_module=TPDModule)
    diff_modules, static_modules = {}, {}
    diff_modules["laser"], static_modules["laser"] = partition(modules["laser"], modules["laser"].get_partition_spec())
    val, grad, (sol, ppo, _) = exo.val_and_grad(diff_modules, args={"static_modules": static_modules}, export=export)

    return val, grad


def calc_loss_and_grads_with_retry(modules: Dict, epoch: int, orig_cfg: Dict, max_retries=3):
    """
    Wrapper around calc_loss_and_grads with XLA/CUDA error handling and retries.

    Args:
        modules: Dict: The modules
        epoch: int: The epoch
        orig_cfg: Dict: The original config
        max_retries: int: Maximum number of retry attempts

    Returns:
        val, flat_grad, grad: Tuple: The value, the flattened gradient, and the pytree gradient
    """
    import time
    import random
    import gc
    import logging

    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        try:
            return calc_loss_and_grads(modules, epoch, orig_cfg)

        except Exception as e:
            error_msg = str(e).lower()

            # Check if this is an XLA/CUDA memory error
            if any(
                keyword in error_msg
                for keyword in [
                    "out of memory",
                    "cuda",
                    "xla",
                    "memory",
                    "allocation",
                    "resource exhausted",
                    "device memory",
                ]
            ):
                logger.warning(f"XLA/CUDA memory error on attempt {attempt + 1}/{max_retries}: {e}")

                if attempt < max_retries - 1:
                    # Aggressive cache and memory cleanup
                    logger.info(f"Performing aggressive memory cleanup before retry {attempt + 2}")

                    # Clear XLA cache
                    clear_xla_cache()

                    # Additional JAX-specific cleanup
                    try:
                        import jax

                        # Clear all JAX caches
                        jax.clear_caches()
                        # Clear backend caches
                        for backend_name in ["gpu", "cuda", "cpu"]:
                            try:
                                backend = jax.lib.xla_bridge.get_backend(backend_name)
                                if hasattr(backend, "clear_compile_cache"):
                                    backend.clear_compile_cache()
                            except:
                                pass
                    except Exception as jax_e:
                        logger.warning(f"JAX cleanup failed: {jax_e}")

                    # Force garbage collection multiple times
                    for _ in range(3):
                        gc.collect()
                        time.sleep(0.1)

                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(1, 3)
                    logger.info(f"Waiting {delay:.2f} seconds before retry")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts with XLA/CUDA error")
                    raise
            else:
                # Non-memory related error, re-raise immediately
                logger.error(f"Non-memory error occurred: {e}")
                raise


def calc_loss_and_grads(modules: Dict, epoch: int, orig_cfg: Dict):
    """
    This is a wrapper around the run_one_val_and_grad function.

    It logs the loss and the gradient norm to mlflow.

    Args:
        modules: Dict: The modules
        epoch: int: The epoch
        orig_cfg: Dict: The original config

    Returns:
        val, flat_grad, grad: Tuple: The value, the flattened gradient, and the pytree gradient
    """

    import logging

    logger = logging.getLogger(__name__)

    import os
    import tempfile, yaml

    import numpy as np
    import mlflow
    from jax.flatten_util import ravel_pytree

    if "BASE_TEMPDIR" in os.environ:
        BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
    else:
        BASE_TEMPDIR = None

    # Pre-emptive memory check
    try:
        import jax

        # Check available memory before computation
        devices = jax.devices()
        for device in devices:
            if hasattr(device, "memory_stats"):
                stats = device.memory_stats()
                logger.debug(f"Device {device} memory: {stats}")
    except Exception as e:
        logger.debug(f"Could not check device memory: {e}")

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        module_path = os.path.join(_td, f"laser.eqx")
        modules["laser"].save(module_path)
        orig_cfg["drivers"]["E0"]["file"] = module_path
        orig_cfg["grid"]["dt"] = f"{np.random.uniform(0.1, 3):.3f}fs"
        with open(_cfg_path := os.path.join(_td, "config.yaml"), "w") as fi:
            yaml.dump(orig_cfg, fi)

        with mlflow.start_run(nested=True, run_name=f"epoch-{epoch}") as nested_run:
            pass

        val, grad = run_one_val_and_grad(run_id=nested_run.info.run_id, _cfg_path=_cfg_path)
        mlflow.log_artifacts(_td, run_id=nested_run.info.run_id)

    flat_grad, _ = ravel_pytree(grad["laser"])
    loss = float(val)
    grad_norm = float(np.linalg.norm(flat_grad))

    mlflow.log_metrics({"loss": loss, "grad norm": grad_norm}, step=epoch)

    return val, flat_grad, grad["laser"]
