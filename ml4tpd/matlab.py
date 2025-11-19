import os
import subprocess
import tempfile

import mat73
import mlflow
import numpy as np
import scienceplots  # noqa: F401  # Needed for the registered matplotlib style
import xarray as xr
import yaml
from adept import utils as adept_utils
from matplotlib import pyplot as plt
from scipy.io import loadmat

plt.style.use(["science", "grid"])


def run_matlab(_cfg_path, shape="uniform", bandwidth_run_id=None):
    cfg = _load_config(_cfg_path)
    base_tempdir = os.environ.get("BASE_TEMPDIR")
    os.environ["PATH"] += ":/global/common/software/nersc9/texlive/2024/bin/x86_64-linux"

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]):
        adept_utils.log_params(cfg)
        vals = []
        num_seeds = 1

        for seed_index in range(num_seeds):
            seed_params = _prepare_seed_params(cfg, seed_index, bandwidth_run_id)

                    tax = np.arange(epwEnergy.shape[0]) * np.squeeze(data["dt"])
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
                    ax.semilogy(tax, epwEnergy, label="EPW Energy")
                    ax.set_xlabel("Time (ps)")
                    ax.set_ylabel("Log EPW Energy")
                    fig.savefig(os.path.join(td, "epw_energy.png"))
                    plt.close()

                    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
                    ax.semilogy(tax, divEmax, label="max Phi")
                    ax.set_xlabel("Time (ps)")
                    ax.set_ylabel("Maximum Potential")
                    fig.savefig(os.path.join(td, "max_divE.png"))
                    plt.close()

                    os.makedirs(laser_dir := os.path.join(td, "laser"), exist_ok=True)
                    os.makedirs(epw_dir := os.path.join(td, "epw"), exist_ok=True)

                    # save fields to xarrays
                    x_matlab_y_adept = np.squeeze(data["x"])
                    y_matlab_x_adept = np.squeeze(data["y"])
                    t = np.squeeze(data["outputTimes"])
                    t_skip = int(t.size // 8)
                    t_skip = t_skip if t_skip > 1 else 1
                    tslice = slice(0, -1, t_skip)

                    # # save laser fields
                    E0x = np.array([data["E0_save"][i]["x"] for i in range(len(data["E0_save"]))])
                    E0y = np.array([data["E0_save"][i]["y"] for i in range(len(data["E0_save"]))])

                    laser_ds = xr.Dataset(
                        {
                            "E0x": (("time", "x", "y"), E0x),
                            "E0y": (("time", "x", "y"), E0y),
                        },
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                            "time": ("time", t, {"units": "ps"}),
                        },
                    )

                    np.abs(laser_ds["E0x"][tslice].T).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0x.png"))
                    plt.close()

                    np.abs(laser_ds["E0y"][tslice].T).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0y.png"))
                    plt.close()

                    # plot lineout of E0y
                    np.abs(laser_ds["E0y"][tslice].isel(y=E0y.shape[2] // 2)).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0y_lineout.png"))
                    plt.close()
                    laser_ds.to_netcdf(os.path.join(laser_dir, "laser_fields.nc"))

                    fig, ax = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
                    ax[0].plot(
                        data["laser"]["bandwidth"]["frequencyShift"], data["laser"]["bandwidth"]["intensity"], "o"
                    )
                    ax[0].set_xlabel("Frequency Shift")
                    ax[0].set_ylabel("Intensity")
                    ax[1].semilogy(
                        data["laser"]["bandwidth"]["frequencyShift"], data["laser"]["bandwidth"]["intensity"], "o"
                    )
                    ax[1].set_xlabel("Frequency Shift")
                    ax[1].set_ylabel("Intensity")
                    ax[2].plot(data["laser"]["bandwidth"]["frequencyShift"], data["laser"]["bandwidth"]["phase"], "o")
                    ax[2].set_xlabel("Frequency Shift")
                    ax[2].set_ylabel("Phase (rad)")
                    fig.savefig(os.path.join(laser_dir, "laser_bandwidth.png"))
                    plt.close()

                    # save epw fields
                    phi = np.array([data["divE_save"][i] for i in range(len(data["divE_save"]))])
                    phi_da = xr.DataArray(
                        phi,
                        dims=("time", "x", "y"),
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                            "time": ("time", t, {"units": "ps"}),
                        },
                        name="phi",
                    )

                    # # save epw fields
                    # phi = np.array([data["divE_save"][i] for i in range(len(data["divE_save"]))])
                    # phi_da = xr.DataArray(
                    #     phi,
                    #     dims=("time", "x", "y"),
                    #     coords={
                    #         "x": ("x", x_matlab_y_adept, {"units": "um"}),
                    #         "y": ("y", y_matlab_x_adept, {"units": "um"}),
                    #         "time": ("time", t, {"units": "ps"}),
                    #     },
                    #     name="phi",
                    # )

                    # plot phi(kx, ky) over time just like the previous plot
                    kx = np.fft.fftshift(np.fft.fftfreq(phi.shape[1], d=(x_matlab_y_adept[1] - x_matlab_y_adept[0]) * 1e-4))
                    ky = np.fft.fftshift(np.fft.fftfreq(phi.shape[2], d=(y_matlab_x_adept[1] - y_matlab_x_adept[0]) * 1e-4))
                    phi_k = np.fft.fftshift(np.fft.fft2(phi, axes=(1, 2)), axes=(1, 2))
                    phi_k_da = xr.DataArray(
                        np.abs(phi_k),
                        dims=("time", "kx", "ky"),
                        coords={
                            "kx": ("kx", kx, {"units": "1/cm"}),
                            "ky": ("ky", ky, {"units": "1/cm"}),
                            "time": ("time", t, {"units": "ps"}),
                        },
                        name="phi_k",
                    )
                    phi_k_da[tslice].plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(epw_dir, "phi_kx_ky.png"))
                    plt.close()

                    os.makedirs(density_dir := os.path.join(td, "density"), exist_ok=True)
                    os.makedirs(nelf_dir := os.path.join(td, "nelf"), exist_ok=True)

                    # os.makedirs(density_dir := os.path.join(td, "density"), exist_ok=True)
                    # os.makedirs(nelf_dir := os.path.join(td, "nelf"), exist_ok=True)

                    # background_density_da = xr.DataArray(
                    #     data["backgroundDensity"],
                    #     dims=("x", "y"),
                    #     coords={
                    #         "x": ("x", x_matlab_y_adept, {"units": "um"}),
                    #         "y": ("y", y_matlab_x_adept, {"units": "um"}),
                    #     },
                    #     name="background_density",
                    # )
                    # nelf_da = xr.DataArray(
                    #     data["Nelf"],
                    #     dims=("x", "y"),
                    #     coords={
                    #         "x": ("x", x_matlab_y_adept, {"units": "um"}),
                    #         "y": ("y", y_matlab_x_adept, {"units": "um"}),
                    #         # "time": ("time", t, {"units": "ps"}),
                    #     },
                    #     name="nelf",
                    # )
                    # background_density_da.T.plot()
                    # plt.savefig(os.path.join(density_dir, "background_density.png"))
                    # plt.close()
                    # background_density_da.to_netcdf(os.path.join(density_dir, "background_density.nc"))

                    # nelf_da.T.plot()
                    # plt.savefig(os.path.join(nelf_dir, "nelf.png"))
                    # plt.close()
                    # nelf_da.to_netcdf(os.path.join(nelf_dir, "nelf.nc"))

                    mlflow.log_artifacts(td)

                mlflow.log_params(params)
            with mlflow.start_run(run_name=f"seed-{seed_index}", nested=True, log_system_metrics=True):
                metrics = _execute_seed_run(seed_params, shape, base_tempdir)
                # logged_params = {k: seed_params[k] for k in ("intensity", "Ln", "seed")}
                mlflow.log_params(seed_params)
                mlflow.log_metrics(metrics)

            vals.append(np.log10(metrics["epw_energy"]))
            if (np.log10(metrics["epw_energy"]) < -25) or (np.log10(metrics["epw_energy"]) > 10):
                break

        mlflow.log_metric("loss", np.mean(vals))


def _load_config(cfg_path):
    with open(cfg_path, "r") as fi:
        return yaml.safe_load(fi)


def _prepare_seed_params(cfg, seed_index, bandwidth_run_id=None):
    seed = int(np.random.randint(0, 2**10))
    cfg["drivers"]["E0"]["params"]["phases"]["seed"] = seed
    cfg["mlflow"]["run"] = f"seed-{seed_index}"

    intensity = float(cfg["units"]["laser intensity"].split(" ")[0])
    gsl = float(cfg["density"]["gradient scale length"].split(" ")[0])
    temperature = float(cfg["units"]["reference electron temperature"].split(" ")[0]) / 1000  # keV

    return {
        "intensity": intensity,
        "Ln": gsl,
        "temperature": temperature,
        "seed": seed,
        "bandwidth_run_id": bandwidth_run_id,
    }


def _execute_seed_run(seed_params, shape, base_tempdir):
    with tempfile.TemporaryDirectory(dir=base_tempdir) as td:
        _run_matlab_simulation(seed_params, shape, td)

        try:
            import mat73

            data = mat73.loadmat(os.path.join(td, "output.mat"))["output"]  # , simplify_cells=True)["output"]

            epw_energy, max_phi, time_axis = _extract_time_series(data)
            metrics = _build_metrics(epw_energy, max_phi)

            _plot_epw_energy(epw_energy, time_axis, td)
            _plot_max_phi(max_phi, time_axis, td)

            axes = _build_axes(data)
            _save_laser_artifacts(data, axes, td)
            _save_epw_artifacts(data, axes, td)
            _save_density_artifacts(data, axes, td)
        except Exception as e:
            print("post-processing failed:", e)

        mlflow.log_artifacts(td)

    return {}


def _run_matlab_simulation(seed_params, shape, output_dir):
    # download bandwidth data if bandwidth_run_id is provided
    if seed_params["bandwidth_run_id"] is not None:
        bandwidth_file = os.path.join(
            os.path.abspath("/global/homes/a/archis/lpse-matlab/"), f"{seed_params['bandwidth_run_id']}.csv"
        )
        mlflow.artifacts.download_artifacts(
            run_id=seed_params["bandwidth_run_id"],
            artifact_path="driver/used_driver.csv",
            dst_path=os.path.abspath("/global/homes/a/archis/lpse-matlab/"),
        )
        # move file from "/global/homes/a/archis/lpse-matlab/used_driver.csv") to run_id.csv
        os.rename(
            os.path.join(os.path.abspath("/global/homes/a/archis/lpse-matlab/"), "used_driver.csv"),
            bandwidth_file,
        )

    matlab_cmd = [
        "matlab",
        "-batch",
        f"addpath('{os.path.abspath('/global/homes/a/archis/lpse-matlab/')}');"
        + "log_lpse("
        + f"{seed_params['intensity']}, "
        + f"{seed_params['Ln']}, "
        + f"{seed_params['temperature']}, "
        + f"'{str(shape).lower()}', "
        + f"{seed_params['seed']}, "
        + f"'{output_dir}', "
        + f"'{seed_params['bandwidth_run_id']}.csv'"
        + ")",
    ]
    subprocess.run(matlab_cmd)


def _extract_time_series(data):
    epw_energy = np.squeeze(data["metrics"]["epwEnergy"])
    max_phi = np.squeeze(data["metrics"]["max"]["divE"])
    time_axis = np.arange(epw_energy.shape[0]) * np.squeeze(data["dt"])
    return epw_energy, max_phi, time_axis


def _build_metrics(epw_energy, max_phi):
    return {
        "epw_energy": float(epw_energy[-1]),
        "max_phi": float(max_phi[-1]),
        "log10_epw_energy": float(np.log10(epw_energy[-1])),
        "log10_max_phi": float(np.log10(max_phi[-1])),
    }


def _plot_epw_energy(epw_energy, time_axis, output_dir):
    _plot_time_history(
        time_axis,
        epw_energy,
        ylabel="EPW Energy",
        label="EPW Energy",
        output_path=os.path.join(output_dir, "epw_energy.png"),
    )


def _plot_max_phi(max_phi, time_axis, output_dir):
    _plot_time_history(
        time_axis,
        max_phi,
        ylabel="Maximum Potential",
        label="max Phi",
        output_path=os.path.join(output_dir, "max_divE.png"),
    )


def _plot_time_history(time_axis, values, ylabel, label, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.semilogy(time_axis, values, label=label)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    _save_fig(fig, output_path)


def _is_1d_simulation(data):
    """Determine if simulation is 1D or 2D based on y-coordinate size."""
    y_coords = np.squeeze(data["y"])
    return y_coords.size == 1


def _build_axes(data):
    x_coords = np.squeeze(data["x"])
    y_coords = np.squeeze(data["y"])
    time_coords = np.squeeze(data["outputTimes"])
    t_skip = max(int(time_coords.size // 8), 1)
    tslice = slice(0, -1, t_skip)
    is_1d = _is_1d_simulation(data)
    return {
        "x": x_coords,
        "y": y_coords,
        "time": time_coords,
        "tslice": tslice,
        "is_1d": is_1d,
    }


def _save_laser_artifacts(data, axes, output_dir):
    laser_dir = os.path.join(output_dir, "laser")
    os.makedirs(laser_dir, exist_ok=True)

    if axes["is_1d"]:
        # 1D case: data has shape (time, x)
        E0y = np.array([np.squeeze(_arr[0]["y"]) for _arr in data["E0_save"]])

        laser_ds = xr.Dataset(
            {
                "E0y": (("time", "x"), E0y),
            },
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "time": ("time", axes["time"], {"units": "ps"}),
            },
        )

        # _save_xarray_line_panel(np.abs(laser_ds["E0x"][axes["tslice"]]), os.path.join(laser_dir, "E0x.png"))
        _save_xarray_line_panel(np.abs(laser_ds["E0y"][axes["tslice"]]), os.path.join(laser_dir, "E0y.png"))
    else:
        # 2D case: data has shape (time, x, y)
        E0x = np.array([_arr[0]["x"][:, 0] for _arr in data["E0_save"]])
        E0y = np.array([_arr[0]["y"][:, 0] for _arr in data["E0_save"]])

        laser_ds = xr.Dataset(
            {
                "E0x": (("time", "x", "y"), E0x),
                "E0y": (("time", "x", "y"), E0y),
            },
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "y": ("y", axes["y"], {"units": "um"}),
                "time": ("time", axes["time"], {"units": "ps"}),
            },
        )

        _save_xarray_panel(np.abs(laser_ds["E0x"][axes["tslice"]].T), os.path.join(laser_dir, "E0x.png"))
        _save_xarray_panel(np.abs(laser_ds["E0y"][axes["tslice"]].T), os.path.join(laser_dir, "E0y.png"))
        _save_xarray_panel(
            np.abs(laser_ds["E0y"][axes["tslice"]].isel(y=E0y.shape[2] // 2)),
            os.path.join(laser_dir, "E0y_lineout.png"),
        )

    laser_ds.to_netcdf(os.path.join(laser_dir, "laser_fields.nc"))

    # some simulations wont have any bandwidth, we want to check for that
    if "frequencyShift" in data["laser"]["bandwidth"]:
        _plot_laser_bandwidth(data, laser_dir)


def _save_xarray_panel(data_array, output_path):
    data_array.plot(col="time", col_wrap=4)
    plt.savefig(output_path)
    plt.close()


def _save_xarray_line_panel(data_array, output_path):
    """Save 1D line plots with multiple time steps."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), tight_layout=True)
    axes = axes.flatten()

    for i, time_idx in enumerate(range(len(data_array.time))):
        if i >= len(axes):
            break
        data_array.isel(time=time_idx).plot(ax=axes[i])
        axes[i].set_title(f"t = {float(data_array.time[time_idx]):.2f} ps")

    plt.savefig(output_path)
    plt.close()


def _plot_laser_bandwidth(data, laser_dir):
    fig, ax = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
    freq_shift = data["laser"]["bandwidth"]["frequencyShift"]
    intensity = data["laser"]["bandwidth"]["intensity"]
    phase = data["laser"]["bandwidth"]["phase"]

    ax[0].plot(freq_shift, intensity, "o")
    ax[0].set_xlabel("Frequency Shift")
    ax[0].set_ylabel("Intensity")

    ax[1].semilogy(freq_shift, intensity, "o")
    ax[1].set_xlabel("Frequency Shift")
    ax[1].set_ylabel("Intensity")

    ax[2].plot(freq_shift, phase, "o")
    ax[2].set_xlabel("Frequency Shift")
    ax[2].set_ylabel("Phase (rad)")

    _save_fig(fig, os.path.join(laser_dir, "laser_bandwidth.png"))


def _save_epw_artifacts(data, axes, output_dir):
    epw_dir = os.path.join(output_dir, "epw")
    os.makedirs(epw_dir, exist_ok=True)

    phi = np.array(data["divE_save"])[:, 0]

    if axes["is_1d"]:
        # 1D case: phi has shape (time, x)
        phi_da = xr.DataArray(
            phi,
            dims=("time", "x"),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "time": ("time", axes["time"], {"units": "ps"}),
            },
            name="phi",
        )
        _save_xarray_line_panel(np.abs(phi_da[axes["tslice"]]), os.path.join(epw_dir, "phi.png"))
        phi_da.to_netcdf(os.path.join(epw_dir, "epw_fields.nc"))

        # 1D Fourier transform
        phi_k_da = _build_phi_fourier_1d(phi, axes)
        _save_xarray_line_panel(phi_k_da[axes["tslice"]], os.path.join(epw_dir, "phi_kx.png"))

        plt.clf()
        _save_xarray_line_panel(np.log10(np.abs(phi_k_da[axes["tslice"]])), os.path.join(epw_dir, "phi_kx_log10.png"))
    else:
        # 2D case: phi has shape (time, x, y)
        phi_da = xr.DataArray(
            phi,
            dims=("time", "x", "y"),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "y": ("y", axes["y"], {"units": "um"}),
                "time": ("time", axes["time"], {"units": "ps"}),
            },
            name="phi",
        )
        _save_xarray_panel(np.abs(phi_da[axes["tslice"]].T), os.path.join(epw_dir, "phi.png"))
        phi_da.to_netcdf(os.path.join(epw_dir, "epw_fields.nc"))

        # 2D Fourier transform
        phi_k_da = _build_phi_fourier(phi, axes)
        phi_k_da[axes["tslice"]].T.plot(col="time", col_wrap=4)
        plt.savefig(os.path.join(epw_dir, "phi_kx_ky.png"))

        plt.clf()
        np.log10(np.abs(phi_k_da[axes["tslice"]])).T.plot(col="time", col_wrap=4)
        plt.savefig(os.path.join(epw_dir, "phi_kx_ky_log10.png"))

    plt.close()


def _build_phi_fourier(phi, axes):
    """Build 2D Fourier transform of phi."""
    c = 299.792458
    w0 = 5366.52868179
    kx = np.fft.fftshift(np.fft.fftfreq(phi.shape[1], d=(axes["x"][1] - axes["x"][0]) / (2 * np.pi))) * c / w0
    ky = np.fft.fftshift(np.fft.fftfreq(phi.shape[2], d=(axes["y"][1] - axes["y"][0]) / (2 * np.pi))) * c / w0
    phi_k = np.fft.fftshift(np.fft.fft2(phi, axes=(1, 2)), axes=(1, 2))

    return xr.DataArray(
        np.abs(phi_k),
        dims=("time", "kx", "ky"),
        coords={
            "kx": ("kx", kx, {"units": "c/$\\omega_0$"}),
            "ky": ("ky", ky, {"units": "c/$\\omega_0$"}),
            "time": ("time", axes["time"], {"units": "ps"}),
        },
        name="phi_k",
    )


def _build_phi_fourier_1d(phi, axes):
    """Build 1D Fourier transform of phi."""
    c = 299.792458
    w0 = 5366.52868179
    kx = np.fft.fftshift(np.fft.fftfreq(phi.shape[1], d=(axes["x"][1] - axes["x"][0]) / (2 * np.pi))) * c / w0
    phi_k = np.fft.fftshift(np.fft.fft(phi, axis=1), axes=1)

    return xr.DataArray(
        np.abs(phi_k),
        dims=("time", "kx"),
        coords={
            "kx": ("kx", kx, {"units": "c/$\\omega_0$"}),
            "time": ("time", axes["time"], {"units": "ps"}),
        },
        name="phi_k",
    )


def _save_density_artifacts(data, axes, output_dir):
    density_dir = os.path.join(output_dir, "density")
    nelf_dir = os.path.join(output_dir, "nelf")
    os.makedirs(density_dir, exist_ok=True)
    os.makedirs(nelf_dir, exist_ok=True)

    if axes["is_1d"]:
        # 1D case: data has shape (x,)
        background_density_da = xr.DataArray(
            np.squeeze(data["backgroundDensity"]),
            dims=("x",),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
            },
            name="background_density",
        )
        background_density_da.plot()
        plt.savefig(os.path.join(density_dir, "background_density.png"))
        plt.close()
        background_density_da.to_netcdf(os.path.join(density_dir, "background_density.nc"))

        nelf_da = xr.DataArray(
            np.squeeze(data["Nelf"]),
            dims=("x",),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
            },
            name="nelf",
        )
        nelf_da.plot()
        plt.savefig(os.path.join(nelf_dir, "nelf.png"))
        plt.close()
        nelf_da.to_netcdf(os.path.join(nelf_dir, "nelf.nc"))
    else:
        # 2D case: data has shape (x, y)
        background_density_da = xr.DataArray(
            data["backgroundDensity"],
            dims=("x", "y"),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "y": ("y", axes["y"], {"units": "um"}),
            },
            name="background_density",
        )
        background_density_da.T.plot()
        plt.savefig(os.path.join(density_dir, "background_density.png"))
        plt.close()
        background_density_da.to_netcdf(os.path.join(density_dir, "background_density.nc"))

        nelf_da = xr.DataArray(
            data["Nelf"],
            dims=("x", "y"),
            coords={
                "x": ("x", axes["x"], {"units": "um"}),
                "y": ("y", axes["y"], {"units": "um"}),
            },
            name="nelf",
        )
        nelf_da.T.plot()
        plt.savefig(os.path.join(nelf_dir, "nelf.png"))
        plt.close()
        nelf_da.to_netcdf(os.path.join(nelf_dir, "nelf.nc"))


def _save_fig(fig, output_path):
    fig.savefig(output_path)
    plt.close(fig)
