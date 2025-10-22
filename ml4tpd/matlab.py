import os

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_matlab(_cfg_path, bandwidth=False):
    import os
    import tempfile

    os.environ["PATH"] += ":/global/common/software/nersc9/texlive/2024/bin/x86_64-linux"
    import yaml, mlflow, subprocess

    from adept import utils as adept_utils
    import numpy as np
    from scipy.io import loadmat
    from matplotlib import pyplot as plt
    import scienceplots
    import xarray as xr

    plt.style.use(["science", "grid"])

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    with mlflow.start_run(run_name=_cfg["mlflow"]["run"]) as parent_run:
        adept_utils.log_params(_cfg)
        vals = []
        num_seeds = 4 if bandwidth else 1
        for i in range(num_seeds):
            _cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(np.random.randint(0, 2**10))
            _cfg["mlflow"]["run"] = f"seed-{i}"
            intensity = float(_cfg["units"]["laser intensity"].split(" ")[0])
            gsl = float(_cfg["density"]["gradient scale length"].split(" ")[0])
            seed = _cfg["drivers"]["E0"]["params"]["phases"]["seed"]

            with mlflow.start_run(run_name=f"seed-{i}", nested=True, log_system_metrics=True) as mlflow_run:
                with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
                    matlab_cmd = [
                        "matlab",
                        "-batch",
                        f"addpath('{os.path.abspath('/global/common/software/m4490/lpse-matlab/')}');"
                        + f"log_lpse({intensity}, {gsl}, {str(bandwidth).lower()}, {seed}, '{td}')",
                    ]
                    subprocess.run(matlab_cmd)

                    data = loadmat(os.path.join(td, "output.mat"), simplify_cells=True)["output"]
                    epwEnergy = np.squeeze(data["metrics"]["epwEnergy"])
                    divEmax = np.squeeze(data["metrics"]["max"]["divE"])
                    nelfmax = np.squeeze(data["metrics"]["max"]["Nelf"])
                    params = {"intensity": intensity, "Ln": gsl, "seed": seed}
                    metrics = {
                        "epw_energy": float(epwEnergy[-1]),
                        "max_phi": float(divEmax[-1]),
                        "log10_epw_energy": float(np.log10(epwEnergy[-1])),
                        "log10_max_phi": float(np.log10(divEmax[-1])),
                    }

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

                    # save laser fields
                    # E0x = np.array([data["E0_save"][i]["x"] for i in range(len(data["E0_save"]))])
                    # E0y = np.array([data["E0_save"][i]["y"] for i in range(len(data["E0_save"]))])

                    # laser_ds = xr.Dataset(
                    #     {
                    #         "E0x": (("time", "x", "y"), E0x),
                    #         "E0y": (("time", "x", "y"), E0y),
                    #     },
                    #     coords={
                    #         "x": ("x", x_matlab_y_adept, {"units": "um"}),
                    #         "y": ("y", y_matlab_x_adept, {"units": "um"}),
                    #         "time": ("time", t, {"units": "ps"}),
                    #     },
                    # )

                    # np.abs(laser_ds["E0x"][tslice].T).plot(col="time", col_wrap=4)
                    # plt.savefig(os.path.join(laser_dir, "E0x.png"))
                    # plt.close()

                    # np.abs(laser_ds["E0y"][tslice].T).plot(col="time", col_wrap=4)
                    # plt.savefig(os.path.join(laser_dir, "E0y.png"))
                    # plt.close()

                    # # plot lineout of E0y
                    # np.abs(laser_ds["E0y"][tslice].isel(y=E0y.shape[2] // 2)).plot(col="time", col_wrap=4)
                    # plt.savefig(os.path.join(laser_dir, "E0y_lineout.png"))
                    # plt.close()
                    # laser_ds.to_netcdf(os.path.join(laser_dir, "laser_fields.nc"))

                    # save epw fields
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

                    # np.abs(phi_da[tslice].T).plot(col="time", col_wrap=4)
                    # plt.savefig(os.path.join(epw_dir, "phi.png"))
                    # phi_da.to_netcdf(os.path.join(epw_dir, "epw_fields.nc"))
                    # plt.close()

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
                mlflow.log_metrics(metrics)

            vals.append(np.log10(metrics["epw_energy"]))

        mlflow.log_metric("loss", np.mean(vals))