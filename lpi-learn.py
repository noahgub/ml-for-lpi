from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple
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


def train_model(_cfg_path, parsl_provider="gpu", num_nodes=4, num_colors=32):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{_cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)
    
    orig_cfg["drivers"]["E0"]["num_colors"] = num_colors
    print(f"Experiment: learn-tpd-100ps-{num_colors}-colors")
    orig_cfg["mlflow"]["experiment"] = f"learn-tpd-100ps-{num_colors}-colors"

    orig_cfg["parsl"]["provider"] = parsl_provider 
    orig_cfg["parsl"]["nodes"] = num_nodes
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
    """Generate (Te, GSL, baseline intensity) tuples from config-defined ranges."""
    nt = cfg["training data"]["num_temperatures"]
    ngsl = cfg["training data"]["num_gradient_scale_lengths"]
    bandwidth = cfg["drivers"]["E0"]["delta_omega_max"] * 2
    temperatures = np.round(np.linspace(2000, 4000, nt), 2)
    gradient_scale_lengths = np.round(np.linspace(200, 600, ngsl), 2)
    all_hps = []

    for te, gsl in product(temperatures, gradient_scale_lengths):
        all_hps.append(
            (te, gsl, round(0.5*calc_tpd_broadband_threshold_intensity(te / 1000, gsl, 0.351, bandwidth) * 1e14, 2))
        )
    return all_hps


@dataclass
class BatchResult:
    mean_loss: float
    grad_norm: float
    total_samples: int
    positive_samples: int
    nan_samples: int


@dataclass
class EpochResult:
    loss_mean: float
    grad_norm_mean: float
    total_samples: int
    positive_samples: int
    nan_samples: int

    @property
    def nan_fraction(self) -> float:
        return float(self.nan_samples / self.total_samples) if self.total_samples else 0.0

    @property
    def positive_fraction(self) -> float:
        return float(self.positive_samples / self.total_samples) if self.total_samples else 0.0


class TrainingLoop:
    """Controller that runs training epochs, handles validation, and updates intensity limits."""

    MAX_USEFUL_LOSS = 1e6

    def __init__(
        self,
        *,
        start_epoch: int,
        orig_cfg: dict,
        base_cfg: dict,
        modules,
        diff_params,
        static_params,
        opt,
        opt_state,
        parent_run_id: str,
        parsl_run_one_val_and_grad,
        parsl_run_fwd,
        all_hps,
        num_nodes: int,
    ):
        self.start_epoch = start_epoch
        self.orig_cfg = orig_cfg
        self.base_cfg = base_cfg
        self.modules = modules
        self.diff_params = diff_params
        self.static_params = static_params
        self.opt = opt
        self.opt_state = opt_state
        self.parent_run_id = parent_run_id
        self.parsl_run_one_val_and_grad = parsl_run_one_val_and_grad
        self.parsl_run_fwd = parsl_run_fwd
        self.all_hps = all_hps
        self.num_batches = 1  # len(all_hps) // batch_size
        self.batch_size = num_nodes * 4
        self.max_epochs = 200
        self.validation_interval = 3

        self.training_tmax_ps = 25.0
        self.training_save_dt_ps = 5.0
        self.validation_tmax_ps = 100.0
        self.validation_save_dt_ps = 5.0

        self.rng = np.random.default_rng()
        self.hp_to_base_intensity = {(hp[0], hp[1]): hp[2] for hp in all_hps}
        self.hp_indices = {(hp[0], hp[1]): idx for idx, hp in enumerate(all_hps)}
        self.unique_temperatures = sorted({hp[0] for hp in all_hps})
        self.unique_gradient_scale_lengths = sorted({hp[1] for hp in all_hps})

        validation_cfg = orig_cfg.get("validation", {})

        # Support both direct pairs and separate temperature/GSL lists (backwards compatible)
        validation_pairs_cfg = validation_cfg.get("pairs")
        if validation_pairs_cfg is not None:
            # New format: direct (Te, GSL) pairs
            self.validation_pairs_population = []
            for pair in validation_pairs_cfg:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    tt, gsl = float(pair[0]), float(pair[1])
                    if (tt, gsl) in self.hp_to_base_intensity:
                        self.validation_pairs_population.append((tt, gsl))
                    else:
                        logger.warning(f"Validation pair ({tt}, {gsl}) not in training set, skipping")

            if not self.validation_pairs_population:
                logger.warning("No valid validation pairs specified, using all training pairs")
                self.validation_pairs_population = [(hp[0], hp[1]) for hp in all_hps]

            default_pairs_per_epoch = len(self.validation_pairs_population)
            self.validation_pairs_per_epoch = max(
                1, int(validation_cfg.get("pairs_per_epoch", default_pairs_per_epoch))
            )
            self.use_direct_pairs = True
        else:
            # Old format: product of temperatures and GSLs
            self.validation_temperature_population = self._build_validation_population(
                available=self.unique_temperatures,
                requested=validation_cfg.get("temperatures"),
                label="temperatures",
            )
            self.validation_gsl_population = self._build_validation_population(
                available=self.unique_gradient_scale_lengths,
                requested=validation_cfg.get("gradient_scale_lengths"),
                label="gradient scale lengths",
            )

            default_temp_subset = len(self.validation_temperature_population)
            default_gsl_subset = len(self.validation_gsl_population)
            temp_subset = validation_cfg.get("temperatures_per_epoch", default_temp_subset)
            gsl_subset = validation_cfg.get("gradient_scale_lengths_per_epoch", default_gsl_subset)
            self.validation_temperatures_per_epoch = (
                max(1, int(temp_subset)) if self.validation_temperature_population else 0
            )
            self.validation_gsls_per_epoch = max(1, int(gsl_subset)) if self.validation_gsl_population else 0
            self.use_direct_pairs = False

        intensity_scan_cfg = validation_cfg.get("intensity_scan", {})
        self.validation_intensity_default_samples = max(1, int(intensity_scan_cfg.get("num_samples", 1)))
        intensity_budget = intensity_scan_cfg.get("max_total_samples")
        self.validation_intensity_total_budget = max(1, int(intensity_budget)) if intensity_budget else None

        # Dynamic intensity factor scheduling - per (Te, GSL) pair
        intensity_schedule_cfg = orig_cfg.get("intensity_schedule", {})
        self.factor_min_initial = float(intensity_schedule_cfg.get("min_initial", 0.5))
        self.factor_min_target = float(intensity_schedule_cfg.get("min_target", 1.8))
        self.factor_max_initial = float(intensity_schedule_cfg.get("max_initial", 1.2))
        self.factor_max_target = float(intensity_schedule_cfg.get("max_target", 3.5))

        # Per-(Te, GSL) intensity factor ranges
        self.hp_factor_ranges = {
            (hp[0], hp[1]): {
                "min": self.factor_min_initial,
                "max": self.factor_max_initial,
            }
            for hp in all_hps
        }

        # Thresholds for adaptation
        self.loss_target_low = float(intensity_schedule_cfg.get("loss_target_low", 0.0))
        self.loss_target_high = float(intensity_schedule_cfg.get("loss_target_high", 20.0))
        self.nan_threshold = float(intensity_schedule_cfg.get("nan_threshold", 0.2))
        self.growth_rate = float(intensity_schedule_cfg.get("growth_rate", 1.08))
        self.shrink_rate = float(intensity_schedule_cfg.get("shrink_rate", 0.95))
        self.grad_clip_norm = float(orig_cfg.get("opt", {}).get("grad_clip_norm", 10000.0))

    def run(self, workdir: str) -> float:
        """Execute the full training curriculum and return the final epoch loss."""
        epoch_loss_mean = 0.0
        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_result, sample_results = self._run_epoch(epoch, workdir)
            val_loss = self._maybe_run_validation(epoch, workdir)
            self._update_intensity_schedule(epoch, sample_results)
            self._log_epoch_metrics(epoch, epoch_result, val_loss)
            epoch_loss_mean = epoch_result.loss_mean
        return epoch_loss_mean

    def _run_epoch(self, epoch: int, workdir: str):
        """Run one epoch of batched jobs and aggregate losses, gradients, and stability data."""
        epoch_losses = []
        epoch_gradnorms = []
        epoch_total_samples = 0
        epoch_positive_samples = 0
        epoch_nan_samples = 0
        sample_results = []  # List of (hp_key, factor, loss, is_nan)

        self.rng.shuffle(self.all_hps)

        for batch_idx in range(self.num_batches):
            training_data = self.all_hps[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
            current_batch_size = len(training_data)
            if current_batch_size == 0:
                continue

            module_path = self._save_batch_state(epoch, batch_idx, workdir)
            batch_result, batch_sample_results = self._run_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                training_data=training_data,
                module_path=module_path,
            )
            epoch_losses.append(batch_result.mean_loss)
            epoch_gradnorms.append(batch_result.grad_norm)
            epoch_total_samples += batch_result.total_samples
            epoch_positive_samples += batch_result.positive_samples
            epoch_nan_samples += batch_result.nan_samples
            sample_results.extend(batch_sample_results)

        loss_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        grad_norm_mean = float(np.mean(epoch_gradnorms)) if epoch_gradnorms else 0.0
        epoch_result = EpochResult(
            loss_mean=loss_mean,
            grad_norm_mean=grad_norm_mean,
            total_samples=epoch_total_samples,
            positive_samples=epoch_positive_samples,
            nan_samples=epoch_nan_samples,
        )
        return epoch_result, sample_results

    def _save_batch_state(self, epoch: int, batch_idx: int, workdir: str) -> str:
        """Persist optimizer state and weights for reproducibility and later export."""
        opt_state_path = os.path.join(workdir, f"opt-state-epoch={epoch}-batch-{batch_idx}.pkl")
        with open(opt_state_path, "wb") as fi:
            pickle.dump(self.opt_state, fi)

        weights_dir = os.path.join(workdir, "weights-history")
        module_path = os.path.join(weights_dir, f"weights-e{epoch:02d}-b{batch_idx:02d}.eqx")
        self.modules["laser"].save(module_path)

        mlflow.log_artifact(opt_state_path)
        mlflow.log_artifact(module_path)
        return module_path

    def _run_batch(self, *, epoch: int, batch_idx: int, training_data, module_path: str):
        """Launch Parsl tasks for one batch, accumulate gradients, and apply an optimizer step."""
        self.orig_cfg["drivers"]["E0"]["file"] = module_path

        val_and_grads = []
        sample_metadata = []  # Track (hp_key, factor) for each sample
        for sample_idx, training_example in enumerate(training_data):
            print(f"{epoch=}, {batch_idx=}, {sample_idx=} -- _Training Data: {training_example}")
            run_cfg_path, export, factor = self._write_training_sample_config(
                epoch=epoch,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                training_example=training_example,
                module_path=module_path,
            )
            tt, gsl, _ = training_example
            hp_key = (tt, gsl)
            sample_metadata.append((hp_key, factor))

            val_and_grads.append(
                self.parsl_run_one_val_and_grad(
                    parent_run_id=self.parent_run_id, _run_cfg_path=run_cfg_path, export=export
                )
            )

        vgs = [vg.result() for vg in val_and_grads]
        raw_validation_losses = np.array([v for v, _ in vgs])
        validation_losses = np.nan_to_num(raw_validation_losses, nan=30.0, posinf=30.0, neginf=30.0)
        validation_losses = np.where(validation_losses > 30.0, 30, validation_losses)
        mean_loss = float(np.mean(validation_losses))

        finite_mask = np.isfinite(raw_validation_losses)
        nan_samples = int(np.sum(~finite_mask))
        positive_samples = int(np.sum(finite_mask & (raw_validation_losses > 0.0)))
        overflow_mask = finite_mask & (np.abs(raw_validation_losses) >= self.MAX_USEFUL_LOSS)
        overflow_samples = int(np.sum(overflow_mask))

        # Build sample results for scheduling updates
        sample_results = []
        for (hp_key, factor), raw_loss in zip(sample_metadata, raw_validation_losses):
            is_nan = not np.isfinite(raw_loss)
            sample_results.append((hp_key, factor, raw_loss, is_nan))

        valid_grad_entries = [
            grad
            for (loss, grad), is_overflow in zip(vgs, overflow_mask)
            if np.isfinite(loss) and not is_overflow and self._tree_all_finite(grad["laser"])
        ]
        dropped_grads = len(vgs) - len(valid_grad_entries)
        if valid_grad_entries:
            avg_grad = adept_utils.all_reduce_gradients(valid_grad_entries, len(valid_grad_entries))
        else:
            zero_grad = jax.tree_map(lambda x: np.zeros_like(x), self.diff_params)
            avg_grad = {"laser": zero_grad}

        grad_norm_unclipped = optax.global_norm(avg_grad["laser"])
        if grad_norm_unclipped > self.grad_clip_norm:
            scale = self.grad_clip_norm / (grad_norm_unclipped + 1e-8)
            avg_grad["laser"] = jax.tree_map(lambda g: g * scale, avg_grad["laser"])

        flat_grad, _ = ravel_pytree(avg_grad["laser"])
        mlflow.log_metrics(
            {
                "batch grad norm": float(np.linalg.norm(flat_grad)),
                "batch loss": mean_loss,
                "batch grad norm unclipped": float(grad_norm_unclipped),
            },
            step=epoch * self.num_batches + batch_idx,
        )
        if dropped_grads:
            mlflow.log_metrics(
                {"batch dropped grad count": dropped_grads, "batch overflow loss count": overflow_samples},
                step=epoch * self.num_batches + batch_idx,
            )

        updates, self.opt_state = self.opt.update(avg_grad["laser"], self.opt_state, self.diff_params)
        self.diff_params = eqx.apply_updates(self.diff_params, updates)
        self.modules["laser"] = eqx.combine(self.diff_params, self.static_params)

        batch_result = BatchResult(
            mean_loss=mean_loss,
            grad_norm=float(np.linalg.norm(flat_grad)),
            total_samples=len(training_data),
            positive_samples=positive_samples,
            nan_samples=nan_samples + overflow_samples,
        )
        return batch_result, sample_results

    def _write_training_sample_config(
        self,
        *,
        epoch: int,
        batch_idx: int,
        sample_idx: int,
        training_example,
        module_path: str,
    ) -> Tuple[str, bool, float]:
        """Write a single training run configuration and return its path, export flag, and factor."""
        tt, gsl, base_intensity = training_example
        hp_key = (tt, gsl)
        export = bool(self.rng.choice([True, False], p=[0.25, 0.75]))

        self._apply_training_temporal_settings()
        factor = self._sample_training_factor(hp_key)
        intensity = factor * base_intensity

        self.orig_cfg["units"]["reference electron temperature"] = f"{tt:.3f} eV"
        self.orig_cfg["density"]["gradient scale length"] = f"{gsl:.3f} um"
        self.orig_cfg["units"]["intensity factor"] = f"{factor:.3f}"
        self.orig_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
        self.orig_cfg["grid"]["dt"] = f"{self.rng.uniform(4, 6):.3f} fs"
        self.orig_cfg["grid"]["dx"] = f"{self.rng.uniform(65, 80):.1f} nm"
        self.orig_cfg["mlflow"]["run"] = (
            f"epoch-{epoch}-batch-{batch_idx}-temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
        )
        self.orig_cfg["mlflow"]["export"] = str(export)

        config_dir = os.path.dirname(os.path.dirname(module_path))
        run_cfg_path = os.path.join(
            config_dir,
            f"config-epoch={epoch}-batch={batch_idx}-sample={sample_idx}.yaml",
        )
        with open(run_cfg_path, "w") as fi:
            yaml.dump(self.orig_cfg, fi)

        return run_cfg_path, export, factor

    def _sample_training_factor(self, hp_key) -> float:
        """Draw a training intensity factor from the per-HP dynamic range."""
        hp_range = self.hp_factor_ranges[hp_key]
        return float(self.rng.uniform(hp_range["min"], hp_range["max"]))

    def _update_intensity_schedule(self, epoch: int, sample_results: list) -> None:
        """Update per-HP intensity factor ranges based on sample-level results."""
        # Group results by HP key
        hp_results = {}
        for hp_key, factor, loss, is_nan in sample_results:
            if hp_key not in hp_results:
                hp_results[hp_key] = []
            hp_results[hp_key].append((factor, loss, is_nan))

        # Update each HP's range independently
        for hp_key, results in hp_results.items():
            if hp_key not in self.hp_factor_ranges:
                continue

            hp_range = self.hp_factor_ranges[hp_key]

            # Calculate statistics for this HP
            losses = [loss for _, loss, is_nan in results if not is_nan]
            nan_count = sum(1 for _, _, is_nan in results if is_nan)
            nan_fraction = nan_count / len(results) if results else 0.0

            # Decide whether to grow or shrink this HP's range
            should_grow = False
            should_shrink = False

            if losses:
                mean_loss = np.mean(losses)
                # losses are small and indicate small gradients
                if mean_loss < self.loss_target_low:
                    should_grow = True
                # Model is struggling with high losses or instability
                if nan_fraction > self.nan_threshold * 1.5 or mean_loss > self.loss_target_high:
                    should_shrink = True
            elif nan_fraction > self.nan_threshold:
                # Only NaNs, definitely shrink
                should_shrink = True

            # Update this HP's bounds
            if should_grow:
                # Expand the range: increase max, increase min (push range upward)
                hp_range["max"] = min(hp_range["max"] * self.growth_rate, self.factor_max_target)
                hp_range["min"] = min(
                    hp_range["min"] * self.growth_rate,
                    self.factor_min_target,
                    hp_range["max"] - 0.1,  # ensure min stays below max
                )
            elif should_shrink:
                # Reduce the upper and lower bound when model struggles
                hp_range["max"] = max(hp_range["max"] * self.shrink_rate, self.factor_max_initial)
                hp_range["min"] = max(hp_range["min"] * self.shrink_rate, self.factor_min_initial)

            # Ensure bounds remain valid
            hp_range["min"] = float(np.clip(hp_range["min"], self.factor_min_initial, self.factor_min_target))
            hp_range["max"] = float(np.clip(hp_range["max"], self.factor_max_initial, self.factor_max_target))
            # Ensure min < max with a small margin
            if hp_range["min"] >= hp_range["max"]:
                hp_range["min"] = max(self.factor_min_initial, hp_range["max"] - 0.1)

    def _maybe_run_validation(self, epoch: int, workdir: str):
        """Run validation at specified intervals with static intensity factors."""
        if epoch % self.validation_interval != 0:
            return None

        latest_weights_path = os.path.join(workdir, "weights-history", f"weights-e{epoch:02d}-latest.eqx")
        self.modules["laser"].save(latest_weights_path)

        validation_tasks = []
        selected_hp_keys = self._select_validation_hp_pairs()

        for hp_key in selected_hp_keys:
            tt, gsl = hp_key
            base_intensity = self.hp_to_base_intensity.get(hp_key)
            if base_intensity is None:
                continue
            hp_index = self.hp_indices.get(hp_key, 0)
            factors = self._build_validation_factors(hp_key)
            for scan_idx, factor in enumerate(factors):
                validation_cfg_path = self._write_validation_config(
                    epoch=epoch,
                    hp_index=hp_index,
                    scan_index=scan_idx,
                    tt=tt,
                    gsl=gsl,
                    base_intensity=base_intensity,
                    factor=factor,
                    weights_path=latest_weights_path,
                    workdir=workdir,
                )

                validation_tasks.append(self.parsl_run_fwd(validation_cfg_path, parent_run_id=self.parent_run_id))

        validation_losses = []
        for future in validation_tasks:
            raw_loss = future.result()
            validation_losses.append(raw_loss)

        if not validation_losses:
            return None

        vals = np.array(validation_losses)
        vals = np.nan_to_num(vals, nan=30.0, posinf=30.0, neginf=30.0)
        vals = np.where(vals > 30.0, 30, vals)
        return float(np.mean(vals))

    def _select_validation_hp_pairs(self):
        """Choose a reduced set of (Te, GSL) pairs to probe during validation."""
        if self.use_direct_pairs:
            # New format: directly specified pairs
            return self._sample_population(self.validation_pairs_population, self.validation_pairs_per_epoch)
        else:
            # Old format: product of temperatures and GSLs
            selected_temps = self._sample_population(
                self.validation_temperature_population, self.validation_temperatures_per_epoch
            )
            selected_gsls = self._sample_population(self.validation_gsl_population, self.validation_gsls_per_epoch)
            hp_pairs = [(tt, gsl) for tt in selected_temps for gsl in selected_gsls]
            if hp_pairs:
                return hp_pairs
            # Fallback: sample from all available pairs
            all_keys = [(hp[0], hp[1]) for hp in self.all_hps]
            desired = max(1, self.validation_temperatures_per_epoch * self.validation_gsls_per_epoch)
            desired = min(desired, len(all_keys))
            return self._sample_population(all_keys, desired)

    def _sample_population(self, population, sample_size: int):
        """Sample without replacement from a population of floats/tuples."""
        items = list(population)
        if not items:
            return []
        if sample_size >= len(items):
            return list(items)
        indices = self.rng.choice(len(items), size=sample_size, replace=False)
        return [items[i] for i in indices]

    def _build_validation_population(self, *, available, requested, label: str):
        """Return the ordered list of values to consider for validation."""
        population = list(available)
        if not population:
            return []
        if requested is None:
            return population

        cleaned = []
        seen = set()
        for raw in requested:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-numeric validation %s value: %r", label, raw)
                continue
            if value not in seen:
                cleaned.append(value)
                seen.add(value)

        if not cleaned:
            logger.warning("No valid entries found for validation %s; using full available range.", label)
            return population

        available_map = {float(val): val for val in population}
        filtered = []
        missing = []
        for value in cleaned:
            if value in available_map:
                filtered.append(available_map[value])
            else:
                missing.append(value)

        if missing:
            missing_str = ", ".join(f"{val:g}" for val in missing)
            logger.warning("Skipping validation %s outside the training set: %s", label, missing_str)

        if not filtered:
            logger.warning("Requested validation %s were unavailable; falling back to all options.", label)
            return population
        return filtered

    def _build_validation_factors(self, hp_key) -> List[float]:
        """Return validation intensity factors from the per-HP dynamic range."""
        hp_range = self.hp_factor_ranges.get(hp_key)
        if hp_range is None:
            # Fallback to initial values if HP not found
            hp_range = {"min": self.factor_min_initial, "max": self.factor_max_initial}
        factors = np.linspace(hp_range["min"], hp_range["max"], self.validation_intensity_default_samples)
        return [float(f) for f in factors]

    def _write_validation_config(
        self,
        *,
        epoch: int,
        hp_index: int,
        scan_index: int,
        tt: float,
        gsl: float,
        base_intensity: float,
        factor: float,
        weights_path: str,
        workdir: str,
    ) -> str:
        """Emit a validation config to probe the candidate factor for a given condition."""
        validation_cfg = deepcopy(self.base_cfg)
        intensity = factor * base_intensity

        validation_cfg.setdefault("grid", {})["tmax"] = self._format_time(self.validation_tmax_ps)
        t_field_cfg = validation_cfg.setdefault("save", {}).setdefault("fields", {}).setdefault("t", {})
        t_field_cfg["dt"] = self._format_time(self.validation_save_dt_ps)
        t_field_cfg["tmax"] = self._format_time(self.validation_tmax_ps)
        validation_cfg["grid"]["dt"] = f"{self.rng.uniform(2, 4):.3f} fs"
        validation_cfg["grid"]["dx"] = f"{self.rng.uniform(65, 80):.1f} nm"
        validation_cfg["drivers"]["E0"]["file"] = weights_path
        validation_cfg["units"]["reference electron temperature"] = f"{tt:.3f} eV"
        validation_cfg["density"]["gradient scale length"] = f"{gsl:.3f} um"
        validation_cfg["units"]["intensity factor"] = f"{factor:.3f}"
        validation_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
        validation_cfg["mlflow"]["run"] = (
            f"epoch-{epoch}-validation-temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
        )
        validation_cfg["mlflow"]["export"] = "True"

        validation_cfg_path = os.path.join(
            workdir, f"validation-config-epoch={epoch}-hp={hp_index}-scan={scan_index}.yaml"
        )
        with open(validation_cfg_path, "w") as fi:
            yaml.dump(validation_cfg, fi)

        return validation_cfg_path

    @staticmethod
    def _tree_all_finite(tree) -> bool:
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    @staticmethod
    def _format_time(value: float, unit: str = "ps") -> str:
        value_str = f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value_str}{unit}"

    def _apply_training_temporal_settings(self) -> None:
        """Ensure training configs run short windows and save sparsely."""
        tmax_str = self._format_time(self.training_tmax_ps)
        save_dt_str = self._format_time(self.training_save_dt_ps)
        self.orig_cfg.setdefault("grid", {})["tmax"] = tmax_str
        t_field = self.orig_cfg.setdefault("save", {}).setdefault("fields", {}).setdefault("t", {})
        t_field["dt"] = save_dt_str
        t_field["tmax"] = tmax_str

    def _log_epoch_metrics(self, epoch: int, epoch_result: EpochResult, val_loss: float) -> None:
        """Log epoch aggregates and optional validation loss to MLflow."""
        nan_fraction = epoch_result.nan_fraction
        positive_fraction = epoch_result.positive_fraction

        # Calculate aggregate statistics across all HP ranges
        all_mins = [hp_range["min"] for hp_range in self.hp_factor_ranges.values()]
        all_maxs = [hp_range["max"] for hp_range in self.hp_factor_ranges.values()]
        all_widths = [hp_range["max"] - hp_range["min"] for hp_range in self.hp_factor_ranges.values()]

        metrics = {
            "epoch loss": epoch_result.loss_mean,
            "epoch grad norm": epoch_result.grad_norm_mean,
            "epoch nan fraction": nan_fraction,
            "epoch positive fraction": positive_fraction,
            "factor_min_mean": float(np.mean(all_mins)),
            "factor_min_min": float(np.min(all_mins)),
            "factor_min_max": float(np.max(all_mins)),
            "factor_max_mean": float(np.mean(all_maxs)),
            "factor_max_min": float(np.min(all_maxs)),
            "factor_max_max": float(np.max(all_maxs)),
            "factor_range_width_mean": float(np.mean(all_widths)),
        }
        if val_loss is not None:
            metrics["val loss"] = val_loss

        mlflow.log_metrics(metrics, step=epoch)


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
    """Spin up Parsl, create a TrainingLoop, and run the configured number of epochs."""
    parsl_config = setup_parsl(parsl_provider, 4, nodes=num_nodes, walltime="8:00:00")
    parsl_run_one_val_and_grad = python_app(run_one_val_and_grad)
    parsl_run_fwd = python_app(run_adept_fwd)
    diff_params, static_params = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())
    all_hps = initialize_training_data(cfg=orig_cfg)
    base_cfg = deepcopy(orig_cfg)
    trainer = TrainingLoop(
        start_epoch=start_epoch,
        orig_cfg=orig_cfg,
        base_cfg=base_cfg,
        modules=modules,
        diff_params=diff_params,
        static_params=static_params,
        opt=opt,
        opt_state=opt_state,
        parent_run_id=parent_run_id,
        parsl_run_one_val_and_grad=parsl_run_one_val_and_grad,
        parsl_run_fwd=parsl_run_fwd,
        all_hps=all_hps,
        num_nodes=num_nodes,
    )
    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
        os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)
        with parsl.load(parsl_config):
            epoch_loss_mean = trainer.run(td)
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


def resume_train_model(cfg_path, run_id, start_epoch, weights_path, opt_state_path, parsl_provider="gpu", num_nodes=4, num_colors=32):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["drivers"]["E0"]["file"] = str(weights_path)
    orig_cfg["drivers"]["E0"]["num_colors"] = num_colors
    orig_cfg["mlflow"]["experiment"] = f"learn-tpd-100ps-{num_colors}-colors"
    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
    print(f"Experiment: learn-tpd-100ps-{num_colors}-colors")

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
    parser.add_argument("--num_colors", type=int, default=32, help="The number of colors to use")

    args = parser.parse_args()
    cfg_path = args.config
    parsl_provider = args.provider
    num_nodes = args.nodes
    resume_run_id = args.run_id
    num_colors = args.num_colors

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
                num_colors=num_colors
            )
    else:
        train_model(cfg_path, parsl_provider=parsl_provider, num_nodes=num_nodes, num_colors=num_colors)
