# -*- coding: utf-8 -*-
"""Shared helpers for varflow and optimization examples."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygfunction as gt
from scipy.interpolate import interp1d


def create_borefield(*, N_1: int, N_2: int, B: float, H: float, D: float, r_b: float):
    return gt.borefield.Borefield.rectangle_field(N_1, N_2, B, B, H, D, r_b)


def precompute_g_function(*, borefield, r_b: float):
    options = {"nSegments": 8, "disp": False}

    ks_min, ks_max = 1.0, 4.0
    cv_min, cv_max = 1.0, 4.0

    alpha_min = ks_min / (cv_max * 1.0e6)
    alpha_max = ks_max / (cv_min * 1.0e6)

    t_min = 3600.0
    t_max = 100.0 * 8760.0 * 3600.0

    t_eskilson_min = alpha_min * t_min / (r_b**2)
    t_eskilson_max = alpha_max * t_max / (r_b**2)

    t_eskilson_grid = np.logspace(np.log10(t_eskilson_min), np.log10(t_eskilson_max), 150)

    alpha_ref = 1.0e-6
    t_grid = t_eskilson_grid * (r_b**2) / alpha_ref

    g_func = gt.gfunction.gFunction(
        borefield, alpha_ref, time=t_grid, boundary_condition="UBWT", options=options
    )

    g_of_eskilson = interp1d(t_eskilson_grid, g_func.gFunc, kind="linear", fill_value="extrapolate")

    return t_eskilson_grid, g_of_eskilson


def simulate_Tb(*, time, q_b_per_m, LoadAgg, T_g=None, T_s=None, dt=None, pre_steps=0, power_start=0.0):
    T_b = np.zeros(len(time), dtype=float)

    if T_s is None:
        if T_g is None:
            raise ValueError("Either T_s or T_g must be provided")
        T_s = T_g

    for i in range(pre_steps):
        t = dt * (i + 1)
        LoadAgg.next_time_step(t)
        LoadAgg.set_current_load(power_start)
        LoadAgg.temporal_superposition()

    for i, (t, q_b_i) in enumerate(zip(time, q_b_per_m)):
        if dt is not None:
            t_abs = dt * (pre_steps + i + 1)
            LoadAgg.next_time_step(t_abs)
        else:
            LoadAgg.next_time_step(t)
        LoadAgg.set_current_load(q_b_i)
        deltaT_b = LoadAgg.temporal_superposition()
        T_b[i] = T_s - deltaT_b

    return T_b


def build_network_grid(
    m_grid,
    borefield,
    pos,
    r_in,
    r_out,
    k_p,
    k_s,
    k_g,
    epsilon,
    mu_f,
    rho_f,
    k_f,
    cp_f,
    n_pipes=2,
    config="parallel",
):
    network_grid = []
    for m_bh in m_grid:
        if not np.isfinite(m_bh) or m_bh <= 0:
            network_grid.append(None)
            continue
        m_flow_pipe = m_bh / n_pipes
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon
        )
        R_f = 1.0 / (h_f * 2.0 * np.pi * r_in)
        R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in, r_out, k_p)
        R_fp = R_f + R_p
        UTubes = [
            gt.pipes.MultipleUTube(
                pos, r_in, r_out, borehole, k_s, k_g, R_fp, nPipes=n_pipes, config=config
            )
            for borehole in borefield
        ]
        network_grid.append(gt.networks.Network(borefield, UTubes))
    return network_grid


def compute_fluid_temperatures_with_network_grid(
    Q_tot,
    T_b,
    m_flow_total,
    m_flow_borehole_ts,
    m_grid,
    network_grid,
    cp_f,
    m_grid_index=None,
):
    T_f_in = T_b.copy()
    T_f_out = T_b.copy()
    mask = m_flow_total > 0
    for i in np.where(mask)[0]:
        if m_grid_index is None:
            idx = np.abs(m_grid - m_flow_borehole_ts[i]).argmin()
        else:
            idx = int(m_grid_index[i])
        network = network_grid[idx]
        if network is not None:
            m_flow_network_i = float(m_flow_total[i])
            T_f_in[i] = network.get_network_inlet_temperature(
                Q_tot[i], T_b[i], m_flow_network_i, cp_f, nSegments=8
            )
            T_f_out[i] = network.get_network_outlet_temperature(
                T_f_in[i], T_b[i], m_flow_network_i, cp_f, nSegments=8
            )
    return T_f_in, T_f_out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.nanmean(diff * diff)))


def compute_Rb_grid_for_plot(m_grid, network_grid, n_boreholes, cp_f):
    Rb_grid = np.full_like(m_grid, np.nan, dtype=float)
    for i, net in enumerate(network_grid):
        if net is not None:
            m_flow_network = m_grid[i] * n_boreholes
            Rb_grid[i] = gt.networks.network_thermal_resistance(net, m_flow_network, cp_f)
    return Rb_grid


def synthetic_load_and_flow(
    x,
    *,
    max_w_per_m: float = 30.0,
    cp_f: float = 4180.0,
    H: float = 100.0,
):
    """
    Monthly step load: max extraction in January, max recharge ~6 months later.
    Analytical mass flow calculation based on target deltaT_target.

    Returns:
      - q_b_per_m: load per meter (W/m)
      - m_flow_borehole: analytisch für ΔT=4K (kg/s/m)
    """
    hours = np.asarray(x, dtype=float)
    hours_in_year = 24.0 * 365.0
    day_of_year = (hours % hours_in_year) / 24.0

    month_load_factors = np.array(
        [1, 2 / 3, 1 / 3, 0, -1 / 3, -2 / 3, -1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3],
        dtype=float,
    )
    month_deltaT = np.array(
        [2, 3, 4, 0, 4, 3, 2, 3, 4, 0, 4, 3], dtype=float
    )

    month_starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=float)
    month_index = np.searchsorted(month_starts[1:], day_of_year, side="right")

    q_b_per_m = month_load_factors[month_index] * max_w_per_m

    m_flow_borehole = np.zeros_like(q_b_per_m, dtype=float)
    for i, idx in enumerate(month_index):
        deltaT = month_deltaT[idx]
        if deltaT > 0 and q_b_per_m[i] != 0:
            m_flow_borehole[i] = abs(q_b_per_m[i]) / (cp_f * deltaT) * H
        else:
            m_flow_borehole[i] = 0.0

    return q_b_per_m, m_flow_borehole


def plot_borefield_cross_section(
    *,
    borefield,
    pos,
    r_in: float,
    r_out: float,
    n_pipes: int,
    out_path: Path,
):
    fig_cs, ax_cs = plt.subplots(1, 1, figsize=(6, 6))

    sample_borehole = borefield[0]
    R_fp = 0.01
    u_tube = gt.pipes.MultipleUTube(
        pos, r_in, r_out, sample_borehole, 1.0, 1.0, R_fp, nPipes=n_pipes, config="parallel"
    )
    u_tube.visualize_pipes()

    ax_cs.set_aspect("equal", "box")
    fig_cs.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_cs)

    fig_field = plt.figure(figsize=(9, 4))
    gt.boreholes.visualize_field(
        borefield,
        viewTop=True,
        view3D=True,
        labels=True,
        showTilt=True,
    )
    fig_field.tight_layout()
    out_field = out_path.with_name("borefield_layout.png")
    plt.savefig(out_field, dpi=150)
    plt.close(fig_field)


def plot_g_function(*, t_eskilson_grid, g_func, t_eskilson_req, out_path: Path):
    fig_g, ax_g = plt.subplots(1, 1, figsize=(9, 4))
    ax_g.plot(t_eskilson_grid, g_func, color="tab:blue", label="g-function (precalculation)")
    ax_g.set_xscale("log")
    ax_g.set_xlabel(r"$t_{\mathrm{eskilson}}$")
    ax_g.set_ylabel("g-Funktion")
    ax_g.grid(True, which="both", alpha=0.3)

    t_req_min = np.nanmin(t_eskilson_req)
    t_req_max = np.nanmax(t_eskilson_req)
    ax_g.axvspan(t_req_min, t_req_max, color="tab:orange", alpha=0.2, label="used range in simulation")
    ax_g.axvline(t_req_min, color="tab:orange", alpha=0.6, linestyle="--")
    ax_g.axvline(t_req_max, color="tab:orange", alpha=0.6, linestyle="--")
    ax_g.legend()

    fig_g.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_g)


def plot_input(*, hours, q_b_per_m, m_flow_borehole_ts, out_path: Path):
    fig_in, (ax_q, ax_m) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_q.set_ylabel(r"$q_b$ [W/m]")
    gt.utilities._format_axes(ax_q)
    ax_q.plot(hours, q_b_per_m)
    ax_q.grid(True, which="both", alpha=0.2)

    ax_m.set_xlabel(r"$t$ [hours]")
    ax_m.set_ylabel(r"$\dot{m}$ [kg/s]")
    gt.utilities._format_axes(ax_m)
    ax_m.plot(hours, m_flow_borehole_ts)
    ax_m.grid(True, which="both", alpha=0.2)

    fig_in.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_in)


def plot_results(*, hours, T_b, R_b_ts, T_f_in, T_f_out, out_path: Path):
    fig_res, (ax_r1, ax_r2, ax_r3, ax_r4) = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    ax_r1.set_ylabel(r"$T_b$ [degC]")
    gt.utilities._format_axes(ax_r1)
    ax_r1.plot(hours, T_b)
    ax_r1.grid(True, which="both", alpha=0.2)

    ax_r2.set_ylabel(r"$R_b$ [K/W]")
    gt.utilities._format_axes(ax_r2)
    ax_r2.plot(hours, R_b_ts)
    ax_r2.grid(True, which="both", alpha=0.2)

    ax_r3.set_ylabel(r"$T_f$ [degC]")
    gt.utilities._format_axes(ax_r3)
    ax_r3.plot(hours, T_f_in, label=r"$T_{f,in}$")
    ax_r3.plot(hours, T_f_out, label=r"$T_{f,out}$")
    ax_r3.grid(True, which="both", alpha=0.2)
    ax_r3.legend()

    delta_T = np.abs(T_f_in - T_f_out)
    ax_r4.set_xlabel(r"$t$ [hours]")
    ax_r4.set_ylabel(r"$\Delta T$ [degC]")
    gt.utilities._format_axes(ax_r4)
    ax_r4.plot(hours, delta_T)
    ax_r4.grid(True, which="both", alpha=0.2)

    fig_res.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_res)


def m_flow_transition_per_borehole(Re_crit, r_in, mu_f, n_pipes):
    D_h = 2.0 * r_in
    m_dot_pipe = Re_crit * mu_f * (np.pi * D_h) / 4.0
    m_dot_borehole = m_dot_pipe * n_pipes
    return m_dot_borehole


def plot_rb_vs_mflow(
    *,
    m_grid,
    Rb_grid,
    m_sim,
    Rb_sim,
    r_in: float,
    mu_f: float,
    n_pipes: int,
    out_path: Path,
):
    fig_rb, ax_rb = plt.subplots(1, 1, figsize=(9, 4))

    ax_rb.plot(m_grid, Rb_grid, linestyle="-", color="tab:blue", label=r"$R_b$ (Grid)")

    if len(m_sim) < 25:
        ax_rb.plot(m_sim, Rb_sim, "o", color="tab:orange", label=r"$R_b$ (Simulation)")

    m_flow_bh_transition = m_flow_transition_per_borehole(2300, r_in, mu_f, n_pipes)
    m_min = float(np.nanmin(m_sim))
    m_max = float(np.nanmax(m_sim))
    ax_rb.axvspan(m_min, m_max, color="tab:green", alpha=0.15, label="simulation area")
    ax_rb.axvline(m_flow_bh_transition, color="tab:red", linestyle="--", label="laminar/turbulent (Re=2300)")

    ax_rb.set_xlabel(r"$\dot{m}$ [kg/s] (pro Bohrloch)")
    ax_rb.set_ylabel(r"$R_b$ [K/W]")
    ax_rb.set_ylim(0.0, 0.5)
    ax_rb.grid(True, which="both", alpha=0.2)
    ax_rb.legend(loc="best")

    fig_rb.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_rb)


def plot_sensitivity_param(df_param: pd.DataFrame, param: str, out_path: Path, note: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x_vals = df_param["value"]
    x_label = param
    if param == "cv_s":
        x_vals = x_vals / 1.0e6
        x_label = "cv_s [MJ/m³/K]"
    ax.plot(x_vals, df_param["rmse_out"], label="RMSE Tf_out")
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMSE [degC]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    if note:
        fig.text(0.5, 0.002, note, fontsize=9, ha="center", va="bottom", color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_combined(combined: dict, out_path: Path, note: str | None = None):
    fig, ax_out = plt.subplots(1, 1, figsize=(9, 6.5))
    ax_tg = ax_out.twiny()
    ax_ps = ax_out.twiny()
    ax_tg.xaxis.set_label_position("bottom")
    ax_tg.xaxis.tick_bottom()
    ax_tg.spines["bottom"].set_position(("outward", 55))
    ax_ps.xaxis.set_label_position("bottom")
    ax_ps.xaxis.tick_bottom()
    ax_ps.spines["bottom"].set_position(("outward", 105))
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    markers = ["o", "s", "^", "D", "v", "P"]
    handles = []
    labels = []
    tg_color = None
    ps_color = None
    for idx, (param, df_param) in enumerate(combined.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        x_vals = df_param["value"]
        if param == "cv_s":
            x_vals = x_vals / 1.0e6
        if param == "T_g":
            target_ax = ax_tg
            tg_color = color
        elif param == "power_start":
            target_ax = ax_ps
            ps_color = color
        else:
            target_ax = ax_out
        line = target_ax.plot(
            x_vals,
            df_param["rmse_out"],
            label=param,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4,
        )
        handles.append(line[0])
        labels.append(param)

    ax_out.set_xlabel("Parameter values (k_s, k_g, cv_s)")
    ax_out.set_ylabel("RMSE Tf_out [degC]")
    ax_out.grid(True, which="both", alpha=0.2)
    ax_out.legend(handles, labels, loc="best", framealpha=0.9, facecolor="white")

    ax_tg.set_xlabel("T_g [degC]")
    ax_ps.set_xlabel("power_start [W/m]")

    if tg_color is not None:
        ax_tg.tick_params(axis="x", colors=tg_color)
        ax_tg.xaxis.label.set_color(tg_color)
        ax_tg.spines["bottom"].set_color(tg_color)
    if ps_color is not None:
        ax_ps.tick_params(axis="x", colors=ps_color)
        ax_ps.xaxis.label.set_color(ps_color)
        ax_ps.spines["bottom"].set_color(ps_color)

    if note:
        fig.text(0.5, 0.002, note, fontsize=9, ha="center", va="bottom", color="gray")

    fig.subplots_adjust(bottom=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_2d(
    x_grid,
    y_grid,
    z_rmse,
    out_path: Path,
    note: str | None = None,
    ridge_line: tuple[float, float, float, float] | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    contour = ax.contourf(x_grid, y_grid, z_rmse, levels=15)
    fig.colorbar(contour, ax=ax, label="RMSE Tf_out [degC]")

    name = out_path.stem
    if "k_s_cv_s" in name:
        ax.set_xlabel("k_s [W/mK]")
        ax.set_ylabel("cv_s [MJ/m³/K]")
        ax.set_title("Sensitivity sweep: k_s vs cv_s")
    elif "T_g_power_start" in name:
        ax.set_xlabel("T_g [degC]")
        ax.set_ylabel("power_start [W/m]")
        ax.set_title("Sensitivity sweep: T_g vs power_start")
    elif "T_g_k_s" in name:
        ax.set_xlabel("T_g [degC]")
        ax.set_ylabel("k_s [W/mK]")
        ax.set_title("Sensitivity sweep: T_g vs k_s")
    elif "power_start_k_s" in name:
        ax.set_xlabel("power_start [W/m]")
        ax.set_ylabel("k_s [W/mK]")
        ax.set_title("Sensitivity sweep: power_start vs k_s")

    if ridge_line is not None:
        x_min, y_min, x_max, y_max = ridge_line
        ax.plot([x_min, x_max], [y_min, y_max], color="black", linewidth=1.5, label="ridge")
    if note:
        fig.text(0.5, 0.02, note, fontsize=9, ha="center", va="bottom", color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit_timeseries(df_out: pd.DataFrame, out_path: Path, note: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(df_out["timestamp"], df_out["Tf_out"], label="measured", alpha=0.8)
    ax.plot(df_out["timestamp"], df_out["Tf_out_sim"], label="simulated", alpha=0.8)
    ax.set_ylabel("Tf_out [degC]")
    ax.legend()
    if note:
        fig.text(0.02, 0.02, note, fontsize=8, ha="left", va="bottom", color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_config(path: Path | str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged
