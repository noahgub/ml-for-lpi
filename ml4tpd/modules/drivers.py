from typing import Dict
import jax
import numpy as np
from astropy.units import Quantity as _Q

from jax import numpy as jnp, tree_util as jtu, nn as jnn, Array
from equinox import tree_at, Module, nn as eqx_nn
from jax.random import normal, PRNGKey, uniform, split

from adept.lpse2d import ArbitraryDriver
from adept._lpse2d.modules import driver

from . import nn


def calc_tpd_threshold_intensity(Te: float, Ln: float, w0: float) -> float:
    """
    Calculate the TPD threshold intensity

    :param Te:
    :return: intensity
    """

    c = 2.99792458e10
    me_keV = 510.998946  # keV/c^2
    me_cgs = 9.10938291e-28
    e = 4.8032068e-10

    vte = np.sqrt(Te / me_keV) * c
    I_threshold = 4 * 4.134 * 1 / (8 * np.pi) * (me_cgs * c / e) ** 2 * w0 * vte**2 / (Ln / 100) * 1e-7

    return I_threshold


class SmoothArbitraryDriver(ArbitraryDriver):
    bounded: bool
    num_gaussians: int
    learn_amps: bool
    learn_phases: bool
    centers: Array
    widths: Array
    heights: Array

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.num_gaussians = 8  # cfg["drivers"]["E0"]["params"]["amplitudes"]["num_gaussians"]
        self.bounded = cfg["drivers"]["E0"]["params"]["amplitudes"]["bounded"]
        init = cfg["drivers"]["E0"]["params"]["amplitudes"]["init"]
        self.learn_amps = cfg["drivers"]["E0"]["params"]["amplitudes"]["learned"]
        self.learn_phases = cfg["drivers"]["E0"]["params"]["phases"]["learned"]

        self.centers, self.widths, self.heights = self.initialize_gaussian_params(
            self.num_gaussians, dw_max=cfg["drivers"]["E0"]["delta_omega_max"], init=init, bounded=self.bounded
        )

    @staticmethod
    def initialize_gaussian_params(num_gaussians: int, dw_max: float, init: str, bounded: bool = False):
        if init == "random":
            if bounded:
                centers = np.random.uniform(-1, 1, num_gaussians).astype(np.float32)
                widths = np.random.uniform(-1, 1, num_gaussians).astype(np.float32)
                heights = np.random.uniform(-1, 1, num_gaussians).astype(np.float32)
            else:
                centers = np.random.uniform(-dw_max, dw_max, num_gaussians).astype(np.float32)
                widths = np.random.uniform(0.1 * dw_max, 4 * dw_max, num_gaussians).astype(np.float32)
                heights = np.random.uniform(0, 1, num_gaussians).astype(np.float32)
        elif init == "uniform":
            if bounded:
                raise NotImplementedError("Uniform initialization not implemented for bounded parameters.")
            else:
                centers = np.zeros(num_gaussians, dtype=np.float32)
                widths = np.full((num_gaussians,), 100.0 * dw_max).astype(np.float32)
                heights = np.ones((num_gaussians,)).astype(np.float32)

        return centers, widths, heights

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)

        if self.learn_amps:
            filter_spec = tree_at(lambda tree: tree.centers, filter_spec, replace=True)
            filter_spec = tree_at(lambda tree: tree.widths, filter_spec, replace=True)
            filter_spec = tree_at(lambda tree: tree.heights, filter_spec, replace=True)
        if self.learn_phases:
            filter_spec = tree_at(lambda tree: tree.phases, filter_spec, replace=True)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        if self.bounded:
            centers = jnp.tanh(self.centers) * self.delta_omega.max()
            widths = (0.51 + 0.5 * jnp.tanh(self.widths)) * (8 * self.delta_omega.max())
            heights = 0.5 + 0.5 * jnp.tanh(self.heights)  # ensure heights are positive
        else:
            centers = self.centers
            widths = self.widths
            heights = self.heights

        intensities = jnp.sum(
            heights[:, None] * jnp.exp(-0.5 * ((self.delta_omega[None, :] - centers[:, None]) / widths[:, None]) ** 2),
            axis=0,
        )

        intensities = intensities / jnp.sum(intensities)
        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


class ZeroLiner(ArbitraryDriver):
    threshold: float

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        old_driver = driver.load(cfg, ArbitraryDriver)
        self.intensities = old_driver.intensities
        self.phases = old_driver.phases
        self.threshold = cfg["drivers"]["E0"]["intensity_threshold"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.scale_intensities(self.intensities)
        intensities = intensities / jnp.sum(intensities)
        intensities = jnp.where(intensities < self.threshold, 0.0, intensities)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


class LearnedSpacer(ArbitraryDriver):
    first_delta_omega: float
    dw_spacing: float
    delta_omega_max: float

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.first_delta_omega = np.random.uniform(
            cfg["drivers"]["E0"]["delta_omega_min"], cfg["drivers"]["E0"]["delta_omega_max"]
        )
        self.dw_spacing = np.random.uniform(
            0, cfg["drivers"]["E0"]["delta_omega_max"] - cfg["drivers"]["E0"]["delta_omega_min"]
        )
        self.delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        delta_omega = jnp.arange(self.first_delta_omega, self.delta_omega_max, self.dw_spacing)
        intensities = jnp.ones_like(delta_omega)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


def reinitialize_nns(model: Module, key: PRNGKey) -> Module:
    weight_keys = split(key, num=len(model.layers))
    bias_keys = split(key, num=len(model.layers))

    for i in range(len(model.layers)):
        model = tree_at(
            lambda tree: tree.layers[i].weight,
            model,
            replace=uniform(weight_keys[i], model.layers[i].weight.shape, minval=-1, maxval=1),
        )

        model = tree_at(
            lambda tree: tree.layers[i].bias,
            model,
            replace=uniform(bias_keys[i], model.layers[i].bias.shape, minval=-1, maxval=1),
        )

    return model


class TPDLearner(ArbitraryDriver):
    amp_model: Module
    phase_model: Module
    inputs: np.array

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        cfg["drivers"]["E0"]["params"]["nn"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        # self.model = nn.GenerativeModel(**cfg["drivers"]["E0"]["params"]["nn"])
        if cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "relu":
            act_fun = jnn.relu
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "tanh":
            act_fun = jnp.tanh
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "sigmoid":
            act_fun = jnn.sigmoid
        else:
            # print("Using default activation function")
            # act_fun = lambda x: x
            raise NotImplementedError(
                f"Activation function {cfg['drivers']['E0']['params']['nn']['activation']} not supported"
            )

        Te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
        Ln = _Q(cfg["density"]["gradient scale length"]).to("um").value
        I0 = _Q(cfg["units"]["laser intensity"]).to("W/cm^2").value

        Te = (Te - 3.0) / 1.0
        Ln = (Ln - 400.0) / 200
        I0 = (np.log10(I0) - 14.5) / 0.5
        self.inputs = np.array((Te, Ln, I0))

        test_inputs = np.random.uniform(-1, 1, (10, 3))

        nn_params = {
            "in_size": cfg["drivers"]["E0"]["params"]["nn"]["in_size"],
            "out_size": cfg["drivers"]["E0"]["num_colors"],
            "width_size": cfg["drivers"]["E0"]["params"]["nn"]["width_size"],
            "depth": cfg["drivers"]["E0"]["params"]["nn"]["depth"],
            "key": PRNGKey(seed=np.random.randint(2**20)),
            "activation": act_fun,
        }
        init_scale = 2.0

        self.amp_model = eqx_nn.MLP(**nn_params)

        nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
        self.phase_model = eqx_nn.MLP(**nn_params)

        max_attempts = 10
        attempts = 0
        check = False
        while attempts < max_attempts and not check:
            ints_and_phases = {
                "amps": jax.vmap(self.amp_model)(jnp.array(test_inputs)),
                "phases": jax.vmap(self.phase_model)(jnp.array(test_inputs)),
            }
            ints, phases = self.process_amplitudes_phases(ints_and_phases)
            if self._check_diversity(ints, phases):
                check = True
            else:
                # # Reinitialize models if check fails
                # self.amp_model = self._reinitialize_weights(
                #     self.amp_model, PRNGKey(seed=np.random.randint(2**20)), init_scale
                # )
                # self.phase_model = self._reinitialize_weights(
                #     self.phase_model, PRNGKey(seed=np.random.randint(2**20)), init_scale
                # )
                nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
                self.amp_model = eqx_nn.MLP(**nn_params)
                nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
                self.phase_model = eqx_nn.MLP(**nn_params)
                attempts += 1

    def _reinitialize_weights(self, mlp, key, scale):
        """Reinitialize MLP weights with higher variance"""
        import jax.random as jr

        keys = jr.split(key, len(mlp.layers))
        new_layers = []

        for i, (layer, layer_key) in enumerate(zip(mlp.layers, keys)):
            if hasattr(layer, "weight") and hasattr(layer, "bias"):
                fan_in = layer.weight.shape[1]
                fan_out = layer.weight.shape[0]

                # Scale differently for output layer to prevent saturation
                layer_scale = scale * 1.5 if i == len(mlp.layers) - 1 else scale
                std = layer_scale * jnp.sqrt(2.0 / (fan_in + fan_out))

                new_weight = jr.normal(layer_key, layer.weight.shape) * std
                new_bias = jnp.zeros_like(layer.bias)

                new_layer = tree_at(lambda l: (l.weight, l.bias), layer, (new_weight, new_bias))
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)

        return tree_at(lambda m: m.layers, mlp, tuple(new_layers))

        # --- end diversity check ---

    def _check_diversity(self, ints, phases, entropy_thresh=1.0, var_thresh=0.05):
        # ints is a jnp array of shape (num_spectra, num_colors)
        # phases is a jnp array of shape (num_spectra, num_colors)
        # we want to check that the intensities and phases are diverse enough across the num_spectra
        num_spectra = ints.shape[0]
        if num_spectra < 2:
            return True  # Not enough spectra to check diversity
        entropy = -jnp.sum(ints * jnp.log(ints + 1e-10), axis=1)  # entropy for each spectrum
        entropy_mean = jnp.mean(entropy)
        entropy_std = jnp.std(entropy)
        entropy_coeff_var = entropy_std / (entropy_mean + 1e-10)
        phase_var = jnp.var(phases, axis=1)  # variance for each spectrum
        phase_var_mean = jnp.mean(phase_var)
        phase_var_std = jnp.std(phase_var)
        phase_var_coeff_var = phase_var_std / (phase_var_mean + 1e-10)
        return True if entropy_coeff_var < entropy_thresh and phase_var_coeff_var < var_thresh else False

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        amp_model_filter_spec = jtu.tree_map(lambda _: False, self.amp_model)
        for i in range(len(self.amp_model.layers)):
            amp_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                amp_model_filter_spec,
                replace=(True, True),
            )

        phase_model_filter_spec = jtu.tree_map(lambda _: False, self.phase_model)
        for i in range(len(self.phase_model.layers)):
            phase_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                phase_model_filter_spec,
                replace=(True, True),
            )

        filter_spec = tree_at(lambda tree: tree.phase_model, filter_spec, replace=phase_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.amp_model, filter_spec, replace=amp_model_filter_spec)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints_and_phases = {
            "amps": self.amp_model(jnp.array(self.inputs)),
            "phases": self.phase_model(jnp.array(self.inputs)),
        }
        ints, phases = self.process_amplitudes_phases(ints_and_phases)
        args["drivers"]["E0"] = {"delta_omega": self.delta_omega, "phases": phases, "intensities": ints} | self.envelope
        return state, args

    def process_amplitudes_phases(self, ints_and_phases):
        if self.model_cfg["amplitudes"]["learned"]:
            ints = ints_and_phases["amps"]
        else:
            ints = self.intensities

        ints = self.scale_intensities(ints)
        ints = ints / jnp.sum(ints)

        if self.model_cfg["phases"]["learned"]:
            _phases_ = ints_and_phases["phases"]
        else:
            _phases_ = self.phases

        phases = 3 * jnp.pi * jnp.tanh(_phases_)

        return ints, phases


class TPDGaussianLearner(ArbitraryDriver):
    height_model: Module
    width_model: Module
    centers_model: Module
    phase_model: Module
    inputs: np.array
    num_gaussians: int

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.num_gaussians = 8
        cfg["drivers"]["E0"]["params"]["nn"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        if cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "relu":
            act_fun = jnn.relu
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "tanh":
            act_fun = jnp.tanh
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "sigmoid":
            act_fun = jnn.sigmoid
        else:
            raise NotImplementedError(
                f"Activation function {cfg['drivers']['E0']['params']['nn']['activation']} not supported"
            )

        nn_params = {
            "in_size": cfg["drivers"]["E0"]["params"]["nn"]["in_size"],
            "out_size": self.num_gaussians,
            "width_size": cfg["drivers"]["E0"]["params"]["nn"]["width_size"],
            "depth": cfg["drivers"]["E0"]["params"]["nn"]["depth"],
            "key": PRNGKey(seed=np.random.randint(2**20)),
            "activation": act_fun,
        }

        self.initialize_nns(cfg, nn_params)
        self.inputs = np.zeros(2)

        if "file" not in cfg["drivers"]["E0"]:
            test_inputs = np.random.uniform(-1, 1, (10, 2))
            max_attempts = 32
            attempts = 0
            check = False
            while attempts < max_attempts and not check:
                ints, phases = jax.vmap(self.forward)(jnp.array(test_inputs))
                print(f"{np.amax(ints[0]-ints[1])=}")
                print(f"{0.2*np.amax(ints[0])=}")
                print()
                reinit_scale = 2.0
                if np.amax(ints[0] - ints[1]) > 0.2 * np.amax(ints[0]):  # self._check_diversity(ints, phases):
                    check = True
                else:
                    # self.initialize_nns(cfg, nn_params)
                    self.centers_model = self._reinitialize_weights(
                        self.centers_model, PRNGKey(seed=np.random.randint(2**20)), reinit_scale
                    )
                    self.width_model = self._reinitialize_weights(
                        self.width_model, PRNGKey(seed=np.random.randint(2**20)), reinit_scale
                    )
                    self.height_model = self._reinitialize_weights(
                        self.height_model, PRNGKey(seed=np.random.randint(2**20)), reinit_scale
                    )
                    self.phase_model = self._reinitialize_weights(
                        self.phase_model, PRNGKey(seed=np.random.randint(2**20)), reinit_scale
                    )
                    attempts += 1
                    reinit_scale *= 1.5

            if not check:
                raise ValueError("Could not initialize diverse Gaussian parameters after several attempts.")
            else:
                print(f"{ints[0]=}")
                print(f"{ints[1]=}")

    def initialize_nns(self, cfg, nn_params):
        self.height_model = eqx_nn.MLP(**nn_params)
        nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
        self.width_model = eqx_nn.MLP(**nn_params)
        nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
        self.centers_model = eqx_nn.MLP(**nn_params)
        nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
        nn_params["out_size"] = cfg["drivers"]["E0"]["num_colors"]
        self.phase_model = eqx_nn.MLP(**nn_params)

    def _reinitialize_weights(self, mlp, key, scale):
        """Reinitialize MLP weights with higher variance"""
        import jax.random as jr

        keys = jr.split(key, len(mlp.layers))
        new_layers = []

        for i, (layer, layer_key) in enumerate(zip(mlp.layers, keys)):
            if hasattr(layer, "weight") and hasattr(layer, "bias"):
                fan_in = layer.weight.shape[1]
                fan_out = layer.weight.shape[0]

                # Scale differently for output layer to prevent saturation
                layer_scale = scale * 1.5 if i == len(mlp.layers) - 1 else scale
                std = layer_scale * jnp.sqrt(2.0 / (fan_in + fan_out))

                new_weight = jr.normal(layer_key, layer.weight.shape) * std
                new_bias = jnp.zeros_like(layer.bias)

                new_layer = tree_at(lambda l: (l.weight, l.bias), layer, (new_weight, new_bias))
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)

        return tree_at(lambda m: m.layers, mlp, tuple(new_layers))

        # --- end diversity check ---

    def _check_diversity(self, ints, phases, entropy_thresh=1.0, var_thresh=0.05):
        # ints is a jnp array of shape (num_spectra, num_colors)
        # phases is a jnp array of shape (num_spectra, num_colors)
        # we want to check that the intensities and phases are diverse enough across the num_spectra
        num_spectra = ints.shape[0]
        if num_spectra < 2:
            return True  # Not enough spectra to check diversity
        entropy = -jnp.sum(ints * jnp.log(ints + 1e-10), axis=1)  # entropy for each spectrum
        entropy_mean = jnp.mean(entropy)
        entropy_std = jnp.std(entropy)
        entropy_coeff_var = entropy_std / (entropy_mean + 1e-10)
        phase_var = jnp.var(phases, axis=1)  # variance for each spectrum
        phase_var_mean = jnp.mean(phase_var)
        phase_var_std = jnp.std(phase_var)
        phase_var_coeff_var = phase_var_std / (phase_var_mean + 1e-10)
        print(f"Entropy CV: {entropy_coeff_var}, Phase Var CV: {phase_var_coeff_var}")
        return True if entropy_coeff_var < entropy_thresh and phase_var_coeff_var < var_thresh else False

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        height_model_filter_spec = jtu.tree_map(lambda _: False, self.height_model)
        for i in range(len(self.height_model.layers)):
            height_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                height_model_filter_spec,
                replace=(True, True),
            )

        width_model_filter_spec = jtu.tree_map(lambda _: False, self.width_model)
        for i in range(len(self.width_model.layers)):
            width_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                width_model_filter_spec,
                replace=(True, True),
            )

        centers_model_filter_spec = jtu.tree_map(lambda _: False, self.centers_model)
        for i in range(len(self.centers_model.layers)):
            centers_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                centers_model_filter_spec,
                replace=(True, True),
            )

        phase_model_filter_spec = jtu.tree_map(lambda _: False, self.phase_model)
        for i in range(len(self.phase_model.layers)):
            phase_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                phase_model_filter_spec,
                replace=(True, True),
            )

        filter_spec = tree_at(lambda tree: tree.phase_model, filter_spec, replace=phase_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.height_model, filter_spec, replace=height_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.width_model, filter_spec, replace=width_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.centers_model, filter_spec, replace=centers_model_filter_spec)

        return filter_spec

    def forward(self, inputs):
        ints = jnp.zeros_like(self.delta_omega)
        centers = jnp.tanh(self.centers_model(jnp.array(inputs))) * self.delta_omega.max()
        widths = (0.51 + 0.5 * jnp.tanh(self.width_model(jnp.array(4 * inputs)))) * (8 * self.delta_omega.max())
        heights = 0.5 + 0.5 * jnp.tanh(self.height_model(jnp.array(inputs)))  # ensure heights are positive

        for i in range(self.num_gaussians):
            ints += heights[i] * jnp.exp(-0.5 * ((self.delta_omega - centers[i]) / widths[i]) ** 2)
        ints = ints / jnp.sum(ints)
        phases = jnp.pi * jnp.tanh(self.phase_model(jnp.array(inputs)))

        return ints, phases

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints, phases = self.forward(args["nn_inputs"])
        args["drivers"]["E0"] = {"delta_omega": self.delta_omega, "phases": phases, "intensities": ints} | self.envelope
        return state, args

    # def initialize_inputs(self, cfg):
    #     Te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    #     Ln = _Q(cfg["density"]["gradient scale length"]).to("um").value
    #     I0 = _Q(cfg["units"]["laser intensity"]).to("W/cm^2").value

    #     Te = (Te - 3.0) / 1.0
    #     Ln = (Ln - 400.0) / 200
    #     I0 = (np.log10(I0) - 14.5) / 0.5
    #     inputs = np.array((Te, Ln))


class GenerativeDriver(ArbitraryDriver):
    input_width: int
    model: Module

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.input_width = cfg["drivers"]["E0"]["params"]["nn"]["input_width"]
        cfg["drivers"]["E0"]["params"]["nn"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        self.model = nn.GenerativeModel(**cfg["drivers"]["E0"]["params"]["nn"])

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        model_filter_spec = jtu.tree_map(lambda _: False, self.model)
        if self.model_cfg["amplitudes"]["learned"]:
            amp_model_filter_spec = jtu.tree_map(lambda _: False, self.model.amp_decoder)
            for i in range(len(self.model.amp_decoder.layers)):
                amp_model_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    amp_model_filter_spec,
                    replace=(True, True),
                )

        if self.model_cfg["phases"]["learned"]:
            phase_model_filter_spec = jtu.tree_map(lambda _: False, self.model.phase_decoder)
            for i in range(len(self.model.phase_decoder.layers)):
                phase_model_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    phase_model_filter_spec,
                    replace=(True, True),
                )

        filter_spec = tree_at(lambda tree: tree.model, filter_spec, replace=model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.model.amp_decoder, filter_spec, replace=amp_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.model.phase_decoder, filter_spec, replace=phase_model_filter_spec)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        inputs = normal(PRNGKey(seed=np.random.randint(2**20)), shape=(self.input_width,))
        ints_and_phases = self.model(inputs)
        ints, phases = self.process_amplitudes_phases(ints_and_phases)
        args["drivers"]["E0"] = {"delta_omega": self.delta_omega, "phases": phases, "intensities": ints} | self.envelope
        return state, args

    def process_amplitudes_phases(self, ints_and_phases):
        if self.model_cfg["amplitudes"]["learned"]:
            ints = ints_and_phases["amps"]
        else:
            ints = self.intensities

        ints = self.scale_intensities(ints)
        ints = ints / jnp.sum(ints)

        if self.model_cfg["phases"]["learned"]:
            _phases_ = ints_and_phases["phases"]
        else:
            _phases_ = self.phases

        phases = jnp.pi * jnp.tanh(_phases_)
        return ints, phases
