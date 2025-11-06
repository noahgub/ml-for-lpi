from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import os


def calc_tpd_broadband_threshold_intensity(
    Te_keV: float = 1.0, L_um: float = 10.0, lambda0: float = 0.8, tau0_over_tauc: float = 1.0
) -> float:
    tau0_over_tauc *= 1.5 # fudge factor for flat spectra
    return 232 * Te_keV**0.75 / L_um ** (2 / 3) / lambda0 ** (4 / 3) * (tau0_over_tauc) ** 0.5


def calc_tpd_threshold_intensity(Te: float, Ln: float, w0: float = 5366.528681791605) -> float:
    """
    Calculate the TPD monochromatic threshold intensity

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

def calc_srs_broadband_threshold_intensity(Te_keV: float = 1.0, L_um: float = 10.0, lambda0: float = 0.8, tau0_over_tauc: float = 1.0) -> float:
    return 1.0

def calc_srs_threshold_intensity(Te: float, Ln: float, w0: float = 5366.528681791605) -> float:
    
    c = 2.99792458e10
    me_cgs = 9.10938291e-28
    e = 4.8032068e-10
    lambda_cm = 3.51e-5
    w0 = 2 * np.pi * c / lambda_cm

    I_threshold = (
        2 ** (1/3)
        / np.sqrt(3)
        * (c / (w0*(Ln * 1e-4)) ) ** (4/3)
        * (me_cgs) ** 2
        * w0 ** 2
        * c ** 3
        / (8 * np.pi * e**2)
        * 1e-7 # cgs -> W/cm^2
    )

    return I_threshold / 1e14


def calc_coherence(lpse_module, used_driver, density):
    def _calc_e0_(t, y, light_wave):
        return lpse_module.diffeqsolve_quants["terms"].vector_field.light.calc_ey_at_one_point(t, y, light_wave)

    calc_e0 = vmap(_calc_e0_, in_axes=(0, None, None))
    t0 = 2 * np.pi / (lpse_module.diffeqsolve_quants["terms"].vector_field.light.w0)

    Ntau = 128

    tau = np.linspace(-150 * t0, 150 * t0, Ntau)
    ey = np.zeros((Ntau, 2), dtype=np.complex64)

    for it, ttau in enumerate(tau):
        tt = np.random.uniform(0, 1e3, int(1e5))
        e0_tt = calc_e0(tt, density, used_driver["E0"])
        e0_tt_tau = calc_e0(tt + ttau, density, used_driver["E0"])
        ey[it, 0] = np.mean(np.abs(e0_tt) ** 2.0)
        ey[it, 1] = np.mean(e0_tt_tau * np.conjugate(e0_tt))

    gtau = ey[:, 1] / ey[:, 0]

    return tau, gtau


def plot_coherence(lpse_module, used_driver, td, density):
    tau, gtau = calc_coherence(lpse_module, used_driver, density)
    tau_0 = 2 * np.pi / (lpse_module.diffeqsolve_quants["terms"].vector_field.light.w0)

    metrics = {"tau_cf": np.trapz(np.abs(gtau) ** 2.0, tau)}

    integrand = np.abs(gtau) ** 2.0
    half_integrand = integrand[len(integrand) // 2 :]
    decreasing_order_args = np.argsort(half_integrand)[::-1]

    for i in range(1, 4):
        slc = slice(len(integrand) // 2 - decreasing_order_args[-i], len(integrand) // 2 + decreasing_order_args[-i])
        _tau, int = tau[slc], integrand[slc]
        metrics[f"tau_c{str(i)}"] = np.trapz(int, _tau)
        metrics[f"bound_{str(i)}"] = _tau[-1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax.plot(tau / tau_0, np.abs(gtau))
    ax.set_xlabel(r"Time delay (in units of $\tau_0$)")
    ax.set_ylabel("Coherence")
    ax.grid()
    ax.set_title(rf"$\tau_c = {round(metrics['tau_cf'] * 1000, 2)}$ fs")
    fig.savefig(os.path.join(td, "driver", "coherence.png"), bbox_inches="tight")

    plt.close()

    return metrics


def plot_bandwidth(e0, td):
    dw_over_w = e0["delta_omega"]  # / cfg["units"]["derived"]["w0"] - 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 5), tight_layout=True)
    ax[0].plot(dw_over_w, e0["intensities"], "o")
    ax[0].grid()
    ax[0].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[0].set_ylabel("$|E|$", fontsize=14)
    ax[1].semilogy(dw_over_w, e0["intensities"], "o")
    ax[1].grid()
    ax[1].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[1].set_ylabel("$|E|$", fontsize=14)
    ax[2].plot(dw_over_w, e0["phases"], "o")
    ax[2].grid()
    ax[2].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[2].set_ylabel(r"$\angle E$", fontsize=14)
    plt.savefig(os.path.join(td, "driver", "driver_that_was_used.png"), bbox_inches="tight")
    plt.close()


def postprocess_bandwidth(used_driver, lpse_module, td, density):
    import pickle
    if_calc_coherence = False
    os.makedirs(os.path.join(td, "driver"), exist_ok=True)
    with open(os.path.join(td, "driver", "used_driver.pkl"), "wb") as fi:
        pickle.dump(used_driver, fi)

    # write used_driver["E0"]["intensities"] and used_driver["E0"]["phases"] and used_driver["E0"]["delta_omega"] to a csv file
    with open(os.path.join(td, "driver", "used_driver.csv"), "w") as fi:
        fi.write("delta_omega, intensity, phase\n")
        for dw, inten, phase in zip(
            used_driver["E0"]["delta_omega"], used_driver["E0"]["intensities"], used_driver["E0"]["phases"]
        ):
            fi.write(f"{dw}, {inten}, {phase}\n")

    plot_bandwidth(used_driver["E0"], td)
    if if_calc_coherence:
        metrics = plot_coherence(lpse_module, used_driver, td, density)
    else:
        metrics = {}
    return metrics
