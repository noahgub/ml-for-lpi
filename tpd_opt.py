from typing import Dict

import logging, os
from contextlib import redirect_stdout, redirect_stderr
from scipy.optimize import OptimizeResult

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def clear_xla_cache():
    """
    Clear XLA compilation cache to prevent memory buildup during optimization.
    
    This function attempts to clear JAX's compilation cache using multiple methods
    to ensure memory is freed during long optimization runs.
    """
    try:
        import jax
        # Clear the compilation cache
        jax.clear_caches()
        logger.info("XLA cache cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear XLA cache: {e}")
    
    try:
        # Force garbage collection after clearing cache
        import gc
        gc.collect()
        logger.debug("Garbage collection completed after cache clear")
    except Exception as e:
        logger.warning(f"Garbage collection failed: {e}")


def run_one_val_and_grad(run_id: str, _cfg_path: str):
    """
    Runs a single val and grad step.

    This function calculates the total electrostatic energy
    in the box and the gradient of the total electrostatic energy with respect to
    the laser parameters.

    Args:
        run_id: str: The run id
        _cfg_path: str: The config path

    Returns:
        val, grad: Tuple: The value and the gradient
    """
    import yaml
    import equinox as eqx

    from adept import ergoExo

    from ml4tpd import SRSModule

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(cfg, adept_module=SRSModule)
    diff_modules, static_modules = {}, {}
    diff_modules["laser"], static_modules["laser"] = eqx.partition(
        modules["laser"], modules["laser"].get_partition_spec()
    )
    val, grad, (sol, ppo, _) = exo.val_and_grad(diff_modules, args={"static_modules": static_modules})

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
    
    for attempt in range(max_retries):
        try:
            return calc_loss_and_grads(modules, epoch, orig_cfg)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if this is an XLA/CUDA memory error
            if any(keyword in error_msg for keyword in [
                'out of memory', 'cuda', 'xla', 'memory', 'allocation',
                'resource exhausted', 'device memory'
            ]):
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
                        for backend_name in ['gpu', 'cuda', 'cpu']:
                            try:
                                backend = jax.lib.xla_bridge.get_backend(backend_name)
                                if hasattr(backend, 'clear_compile_cache'):
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
                    delay = (2 ** attempt) + random.uniform(1, 3)
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
    import tempfile, yaml
    import numpy as np
    import mlflow
    from jax.flatten_util import ravel_pytree

    # Pre-emptive memory check
    try:
        import jax
        # Check available memory before computation
        devices = jax.devices()
        for device in devices:
            if hasattr(device, 'memory_stats'):
                stats = device.memory_stats()
                logger.debug(f"Device {device} memory: {stats}")
    except Exception as e:
        logger.debug(f"Could not check device memory: {e}")

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        module_path = os.path.join(_td, f"laser.eqx")
        modules["laser"].save(module_path)
        orig_cfg["drivers"]["E0"]["file"] = module_path
        orig_cfg["grid"]["dt"] = f"{np.random.uniform(5, 15):.3f}fs"
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


def optax_loop(orig_cfg: Dict, modules: Dict):
    """
    Performs the optimization loop using optax.

    Args:
        parent_run_id: str: The parent run id
        orig_cfg: Dict: The original config
        modules: Dict: The modules


    """
    import optax
    import equinox as eqx

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"], decay_steps=orig_cfg["opt"]["decay_steps"]
    )
    opt = optax.adam(learning_rate=lr_sched)
    opt_state = opt.init(eqx.filter(modules["laser"], eqx.is_array))  # initialize the optimizer state

    # Configuration for cache cleanup
    cache_cleanup_interval = orig_cfg.get("opt", {}).get("cache_cleanup_interval", 50)
    
    for i in range(200):  # 1000 epochs
        _, _, laser_grad = calc_loss_and_grads(modules, i, orig_cfg)
        updates, opt_state = opt.update(laser_grad, opt_state, modules["laser"])
        modules["laser"] = eqx.apply_updates(modules["laser"], updates)
        
        # Periodic XLA cache cleanup to prevent memory buildup
        if (i + 1) % cache_cleanup_interval == 0:
            logger.info(f"Performing XLA cache cleanup at epoch {i + 1}")
            clear_xla_cache()
    
    # Final cache cleanup after optax optimization completes
    logger.info("Performing final XLA cache cleanup after optax optimization")
    clear_xla_cache()


def scipy_loop(orig_cfg: Dict, modules: Dict) -> OptimizeResult:
    """
    Performs the optimization loop using scipy.

    The main reason this is different than the optax loop is because scipy prefers numpy arrays so
    the pytrees need to be flattened

    Args:
        parent_run_id: str: The parent run id
        orig_cfg: Dict: The original config
        modules: Dict: The modules

    Returns:
        result: The result of the optimization
    """
    import time
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree
    import numpy as np
    import equinox as eqx

    class Fitter:
        def __init__(self, _modules):
            self.model_cfg = _modules["laser"].model_cfg
            x0, self.static_params = eqx.partition(_modules["laser"], _modules["laser"].get_partition_spec())
            self.flattened_x0, self.unravel_pytree = ravel_pytree(x0)
            self.epoch = 0
            # Cache cleanup configuration
            self.cache_cleanup_interval = 1 #orig_cfg.get("opt", {}).get("cache_cleanup_interval", 10)
            # self.parent_run_id = parent_run_id

        def loss_fn(self, flattened_x):
            diff_params = self.unravel_pytree(flattened_x)
            modules["laser"] = eqx.combine(diff_params, self.static_params)
            for k in self.model_cfg.keys():
                modules["laser"].model_cfg[k] = self.model_cfg[k]
            val, flat_grad, _ = calc_loss_and_grads(modules, self.epoch, orig_cfg)
            self.epoch += 1

            # Periodic XLA cache cleanup during scipy optimization
            # if self.epoch % self.cache_cleanup_interval == 0:
            
            logger.info(f"Performing XLA cache cleanup at iteration {self.epoch}")
            clear_xla_cache()
            time.sleep(60)

            return float(val), np.array(flat_grad)

        def fit(self):
            result = minimize(
                self.loss_fn,
                np.array(self.flattened_x0, dtype=np.float32),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 100, "disp": True},
            )
            
            # Final cache cleanup after optimization completes
            logger.info("Performing final XLA cache cleanup after scipy optimization")
            clear_xla_cache()
            
            return result

    fitter = Fitter(modules)
    result = fitter.fit()

    return result


def run_opt(_cfg_path: str):
    """
    Sets up and runs the parent run which is the optimization loop

    Args:
        _cfg_path: str: Path to the config file


    """
    import jax
    from copy import deepcopy
    from adept import ergoExo
    from adept import utils as adept_utils
    from ml4tpd import TPDModule

    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    import yaml, mlflow, tempfile, os

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    if cfg["opt"]["method"] == "optax":
        optimization_loop = optax_loop
    elif cfg["opt"]["method"] == "scipy":
        optimization_loop = scipy_loop
    else:
        raise NotImplementedError(f"Optimization method {cfg['opt']['method']} not implemented.")

    _tt = cfg["units"]["reference electron temperature"]
    _gsl = cfg["density"]["gradient scale length"]
    _intensity = cfg["units"]["laser intensity"]

    # cfg["mlflow"]["run"] = f"temperature={_tt}-gsl={_gsl}-intensity={_intensity}"
    experiment = cfg["mlflow"]["experiment"]
    mlflow.set_experiment(experiment)
    print(f"Experiment: {experiment}")

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        # adept_utils.log_params(cfg)

        parent_run_id = mlflow_run.info.run_id
        orig_cfg = deepcopy(cfg)

    exo = ergoExo(mlflow_run_id=parent_run_id, mlflow_nested=False)
    modules = exo.setup(cfg, adept_module=TPDModule)

    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        optimization_loop(orig_cfg, modules)

    # Final cleanup after entire optimization run
    logger.info("Performing final XLA cache cleanup after optimization run")
    clear_xla_cache()

    return mlflow_run


def run_opt_with_retry(config_path, max_retries=3):
    from botocore.exceptions import ClientError
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            return run_opt(config_path)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"AWS ClientError on attempt {attempt + 1}: {error_code} - {error_message}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise

if __name__ == "__main__":
    import argparse, mlflow

    parser = argparse.ArgumentParser(description="Run TPD training.")
    parser.add_argument("--config", type=str, help="The config file")
    parser.add_argument("--run_id", type=str, help="The run id")
    args = parser.parse_args()

    if args.run_id is not None:
        run_id = args.run_id
        cfg_path = os.path.join(mlflow.get_run(run_id).info.artifact_uri, "config.yaml")
    else:
        cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    mlflow_run = run_opt(cfg_path)
