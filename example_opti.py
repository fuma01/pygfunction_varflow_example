# -*- coding: utf-8 -*-
"""Example: parameter estimation from synthetic borefield measurements.

This script loads measurements (Tf_in, Tf_out, m_flow) and estimates
T_s, k_s, cv_s, k_g and optionally power_start to match Tf_out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pygfunction as gt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time as tt


@dataclass
class OptiConfig:
    measurement_path: Path
    output_dir: Path
    date_start: str | None
    max_iter: int
    optimize_power_start: bool
    no_penalty: bool
    sensitivity_sweep: bool
    sweep_points: int
    sweep_2d: bool
    sweep_2d_points: int


@dataclass
class GeometryConfig:
    D: float = 4.0
    H: float = 150.0
    r_b: float = 0.075
    N_1: int = 2
    N_2: int = 4
    B: float = 7.5
    r_out: float = 0.016
    r_in: float = 0.014
    D_s: float = 0.04
    epsilon: float = 1.0e-6
    n_pipes: int = 2


@dataclass
class FluidConfig:
    fluid_name: str = "MEA"
    fluid_temp: float = 12.0


def parse_args() -> OptiConfig:
    parser = argparse.ArgumentParser(description="Estimate ground parameters from measurements.")
    parser.add_argument(
        "--measurements",
        default="outputs/varflow_measurements.csv",
        help="CSV file with timestamp, Tf_in, Tf_out, m_flow",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--date-start",
        default=None,
        help="Optional commissioning date (YYYY-MM-DD) before measurements start",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Maximum optimizer iterations",
    )
    parser.add_argument(
        "--optimize-power-start",
        action="store_true",
        help="Optimize power_start when date_start is provided",
    )
    parser.add_argument(
        "--no-penalty",
        action="store_true",
        help="Disable penalty term in the objective",
    )
    parser.add_argument(
        "--sensitivity-sweep",
        action="store_true",
        help="Run 1D sensitivity sweeps for all parameters and exit",
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=9,
        help="Number of points per parameter in sensitivity sweep",
    )
    parser.add_argument(
        "--sweep-2d",
        action="store_true",
        help="Run 2D sweep for k_s vs cv_s on Tf_out and exit",
    )
    parser.add_argument(
        "--sweep-2d-points",
        type=int,
        default=9,
        help="Grid size per axis for 2D sweep",
    )
    args = parser.parse_args()

    return OptiConfig(
        measurement_path=Path(args.measurements),
        output_dir=Path(args.output_dir),
        date_start=args.date_start,
        max_iter=args.max_iter,
        optimize_power_start=args.optimize_power_start,
        no_penalty=args.no_penalty,
        sensitivity_sweep=args.sensitivity_sweep,
        sweep_points=args.sweep_points,
        sweep_2d=args.sweep_2d,
        sweep_2d_points=args.sweep_2d_points,
    )


def load_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def infer_dt_seconds(df: pd.DataFrame) -> float:
    diffs = df["timestamp"].diff().dropna().dt.total_seconds()
    dt = float(diffs.median())
    if not np.allclose(diffs, dt, rtol=0, atol=1e-6):
        raise ValueError("Irregular time steps detected in measurements.")
    return dt


def compute_q_b_per_m(df: pd.DataFrame, cp_f: float, H: float) -> np.ndarray:
    delta_t = df["Tf_out"].to_numpy() - df["Tf_in"].to_numpy()
    return df["m_flow"].to_numpy() * cp_f * delta_t / H


def create_borefield(geo: GeometryConfig):
    borefield = gt.borefield.Borefield.rectangle_field(
        geo.N_1, geo.N_2, geo.B, geo.B, geo.H, geo.D, geo.r_b
    )
    return borefield


def precompute_g_function(borefield, r_b: float):
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


def simulate_Tb(
    *,
    time: np.ndarray,
    q_b_per_m: np.ndarray,
    LoadAgg,
    T_s: float,
    dt: float,
    pre_steps: int,
    power_start: float,
):
    T_b = np.zeros_like(q_b_per_m, dtype=float)

    for i in range(pre_steps):
        t = dt * (i + 1)
        LoadAgg.next_time_step(t)
        LoadAgg.set_current_load(power_start)
        LoadAgg.temporal_superposition()

    for i, (t, q_b_i) in enumerate(zip(time, q_b_per_m)):
        t_abs = dt * (pre_steps + i + 1)
        LoadAgg.next_time_step(t_abs)
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


def run_sensitivity_sweep(
    *,
    df: pd.DataFrame,
    dt: float,
    time: np.ndarray,
    q_b_per_m: np.ndarray,
    borefield,
    g_of_eskilson,
    geo: GeometryConfig,
    cp_f: float,
    rho_f: float,
    mu_f: float,
    k_f: float,
    pos,
    m_grid: np.ndarray,
    m_grid_index: np.ndarray,
    pre_steps: int,
    starts: dict,
    bounds: dict,
    optimize_power_start: bool,
    sweep_points: int,
    out_dir: Path,
):
    m_flow_borehole_ts = df["m_flow"].to_numpy()
    n_boreholes = len(borefield)
    m_flow_total = m_flow_borehole_ts * n_boreholes
    Q_tot = n_boreholes * geo.H * q_b_per_m
    tf_out_meas = df["Tf_out"].to_numpy()

    cache = {}

    def get_network_grid(k_s: float, k_g: float):
        cache_key = (round(k_s, 6), round(k_g, 6))
        if cache_key in cache:
            return cache[cache_key]
        network_grid = build_network_grid(
            m_grid,
            borefield,
            pos,
            geo.r_in,
            geo.r_out,
            0.4,
            k_s,
            k_g,
            geo.epsilon,
            mu_f,
            rho_f,
            k_f,
            cp_f,
            n_pipes=geo.n_pipes,
            config="parallel",
        )
        cache[cache_key] = network_grid
        return network_grid

    sweep_params = ["T_s", "k_s", "cv_s", "k_g"]
    if optimize_power_start:
        sweep_params.append("power_start")

    summary_rows = []
    combined = {}
    eval_counter = 0
    for param in sweep_params:
        values = np.linspace(bounds[param][0], bounds[param][1], sweep_points)
        rows = []
        for val in values:
            t_eval_start = tt.perf_counter()
            params = dict(starts)
            params[param] = float(val)
            if not optimize_power_start:
                params["power_start"] = starts["power_start"]

            alpha = params["k_s"] / params["cv_s"]
            load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
            time_req = load_agg.get_times_for_simulation()
            t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
            g_needed = g_of_eskilson(t_eskilson_req)
            load_agg.initialize(g_needed / (2 * np.pi * params["k_s"]))

            T_b = simulate_Tb(
                time=time,
                q_b_per_m=q_b_per_m,
                LoadAgg=load_agg,
                T_s=params["T_s"],
                dt=dt,
                pre_steps=pre_steps,
                power_start=params["power_start"],
            )

            network_grid = get_network_grid(params["k_s"], params["k_g"])
            _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
                Q_tot,
                T_b,
                m_flow_total,
                m_flow_borehole_ts,
                m_grid,
                network_grid,
                cp_f,
                m_grid_index=m_grid_index,
            )

            rmse_out = rmse(T_f_out_sim, tf_out_meas)
            eval_counter += 1
            t_eval = tt.perf_counter() - t_eval_start
            params_str = ", ".join(f"{k}={v:.6g}" for k, v in params.items())
            print(
                f"Sweep {eval_counter:04d} | param={param} value={float(val):.6g} | "
                f"rmse_out={rmse_out:.6g} | t={t_eval:.3f}s | {params_str}"
            )
            rows.append({"param": param, "value": float(val), "rmse_out": rmse_out})

        df_param = pd.DataFrame(rows)
        df_param.to_csv(out_dir / f"sensitivity_{param}.csv", index=False)
        combined[param] = df_param
        summary_rows.extend(rows)

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
        fig.tight_layout()
        fig.savefig(out_dir / f"sensitivity_{param}.png", dpi=150)
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_dir / "sensitivity_summary.csv", index=False)

    fig, ax_out = plt.subplots(1, 1, figsize=(9, 4))
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (param, df_param) in enumerate(combined.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        x_vals = df_param["value"]
        if param == "cv_s":
            x_vals = x_vals / 1.0e6
        ax_out.plot(
            x_vals,
            df_param["rmse_out"],
            label=param,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4,
        )

    ax_out.set_xlabel("Parameter value (cv_s in MJ/m³/K)")
    ax_out.set_ylabel("RMSE Tf_out [degC]")
    ax_out.grid(True, which="both", alpha=0.2)
    ax_out.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_combined.png", dpi=150)
    plt.close(fig)


def run_sensitivity_sweep_2d(
    *,
    df: pd.DataFrame,
    dt: float,
    time: np.ndarray,
    q_b_per_m: np.ndarray,
    borefield,
    g_of_eskilson,
    geo: GeometryConfig,
    cp_f: float,
    rho_f: float,
    mu_f: float,
    k_f: float,
    pos,
    m_grid: np.ndarray,
    m_grid_index: np.ndarray,
    pre_steps: int,
    starts: dict,
    bounds: dict,
    optimize_power_start: bool,
    sweep_points: int,
    out_dir: Path,
):
    m_flow_borehole_ts = df["m_flow"].to_numpy()
    n_boreholes = len(borefield)
    m_flow_total = m_flow_borehole_ts * n_boreholes
    Q_tot = n_boreholes * geo.H * q_b_per_m
    tf_out_meas = df["Tf_out"].to_numpy()

    k_s_vals = np.linspace(bounds["k_s"][0], bounds["k_s"][1], sweep_points)
    cv_s_vals = np.linspace(bounds["cv_s"][0], bounds["cv_s"][1], sweep_points)

    cache = {}

    def get_network_grid(k_s: float, k_g: float):
        cache_key = (round(k_s, 6), round(k_g, 6))
        if cache_key in cache:
            return cache[cache_key]
        network_grid = build_network_grid(
            m_grid,
            borefield,
            pos,
            geo.r_in,
            geo.r_out,
            0.4,
            k_s,
            k_g,
            geo.epsilon,
            mu_f,
            rho_f,
            k_f,
            cp_f,
            n_pipes=geo.n_pipes,
            config="parallel",
        )
        cache[cache_key] = network_grid
        return network_grid

    z_rmse = np.zeros((len(cv_s_vals), len(k_s_vals)), dtype=float)
    eval_counter = 0
    for i, cv_s in enumerate(cv_s_vals):
        for j, k_s in enumerate(k_s_vals):
            t_eval_start = tt.perf_counter()
            params = dict(starts)
            params["cv_s"] = float(cv_s)
            params["k_s"] = float(k_s)
            if not optimize_power_start:
                params["power_start"] = starts["power_start"]

            alpha = params["k_s"] / params["cv_s"]
            load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
            time_req = load_agg.get_times_for_simulation()
            t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
            g_needed = g_of_eskilson(t_eskilson_req)
            load_agg.initialize(g_needed / (2 * np.pi * params["k_s"]))

            T_b = simulate_Tb(
                time=time,
                q_b_per_m=q_b_per_m,
                LoadAgg=load_agg,
                T_s=params["T_s"],
                dt=dt,
                pre_steps=pre_steps,
                power_start=params["power_start"],
            )

            network_grid = get_network_grid(params["k_s"], params["k_g"])
            _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
                Q_tot,
                T_b,
                m_flow_total,
                m_flow_borehole_ts,
                m_grid,
                network_grid,
                cp_f,
                m_grid_index=m_grid_index,
            )

            rmse_out = rmse(T_f_out_sim, tf_out_meas)
            z_rmse[i, j] = rmse_out
            eval_counter += 1
            t_eval = tt.perf_counter() - t_eval_start
            print(
                f"Sweep2D {eval_counter:04d} | k_s={k_s:.6g} cv_s={cv_s:.6g} | "
                f"rmse_out={rmse_out:.6g} | t={t_eval:.3f}s"
            )

    ks_grid, cvs_grid = np.meshgrid(k_s_vals, cv_s_vals)
    df_grid = pd.DataFrame(
        {
            "k_s": ks_grid.ravel(),
            "cv_s": cvs_grid.ravel(),
            "rmse_out": z_rmse.ravel(),
        }
    )
    df_grid.to_csv(out_dir / "sensitivity_2d_k_s_cv_s.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    contour = ax.contourf(ks_grid, cvs_grid / 1.0e6, z_rmse, levels=15)
    fig.colorbar(contour, ax=ax, label="RMSE Tf_out [degC]")
    ax.set_xlabel("k_s [W/mK]")
    ax.set_ylabel("cv_s [MJ/m³/K]")
    ax.set_title("Sensitivity sweep: k_s vs cv_s")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_2d_k_s_cv_s.png", dpi=150)
    plt.close(fig)


def objective_factory(
    *,
    df: pd.DataFrame,
    q_b_per_m: np.ndarray,
    time: np.ndarray,
    dt: float,
    borefield,
    g_of_eskilson,
    geo: GeometryConfig,
    cp_f: float,
    rho_f: float,
    mu_f: float,
    k_f: float,
    k_p: float,
    pos,
    m_grid: np.ndarray,
    pre_steps: int,
    optimize_power_start: bool,
    steps: dict,
    starts: dict,
    penalty_weight: float,
):
    cache = {}
    last_eval = {
        "loss": None,
        "params": None,
        "timings": None,
        "nfev": 0,
    }
    eval_counts = {"nfev": 0}
    n_boreholes = len(borefield)
    m_flow_borehole_ts = df["m_flow"].to_numpy()
    m_flow_total = m_flow_borehole_ts * n_boreholes
    Q_tot = n_boreholes * geo.H * q_b_per_m
    m_grid_index = np.abs(m_grid[None, :] - m_flow_borehole_ts[:, None]).argmin(axis=1)

    def get_network_grid(k_s: float, k_g: float):
        cache_key = (round(k_s, 6), round(k_g, 6))
        if cache_key in cache:
            return cache[cache_key]
        network_grid = build_network_grid(
            m_grid,
            borefield,
            pos,
            geo.r_in,
            geo.r_out,
            k_p,
            k_s,
            k_g,
            geo.epsilon,
            mu_f,
            rho_f,
            k_f,
            cp_f,
            n_pipes=geo.n_pipes,
            config="parallel",
        )
        cache[cache_key] = network_grid
        return network_grid

    def objective(x: np.ndarray):
        eval_counts["nfev"] += 1
        if optimize_power_start:
            T_s, k_s, cv_s, k_g, power_start = x
        else:
            T_s, k_s, cv_s, k_g = x
            power_start = starts["power_start"]

        t0 = tt.perf_counter()
        alpha = k_s / cv_s

        load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
        time_req = load_agg.get_times_for_simulation()
        t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
        g_needed = g_of_eskilson(t_eskilson_req)
        load_agg.initialize(g_needed / (2 * np.pi * k_s))
        t1 = tt.perf_counter()

        T_b = simulate_Tb(
            time=time,
            q_b_per_m=q_b_per_m,
            LoadAgg=load_agg,
            T_s=T_s,
            dt=dt,
            pre_steps=pre_steps,
            power_start=power_start,
        )
        t2 = tt.perf_counter()

        network_grid = get_network_grid(k_s, k_g)
        t3 = tt.perf_counter()
        _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
            Q_tot,
            T_b,
            m_flow_total,
            m_flow_borehole_ts,
            m_grid,
            network_grid,
            cp_f,
            m_grid_index=m_grid_index,
        )
        t4 = tt.perf_counter()

        resid = T_f_out_sim - df["Tf_out"].to_numpy()
        mse = np.nanmean(resid**2)

        penalty = 0.0
        for name, step in steps.items():
            if name == "power_start" and not optimize_power_start:
                continue
            penalty += ((locals()[name] - starts[name]) / step) ** 2
        loss = mse + penalty_weight * penalty
        last_eval["loss"] = loss
        last_eval["params"] = {
            "T_s": T_s,
            "k_s": k_s,
            "cv_s": cv_s,
            "k_g": k_g,
            "power_start": power_start,
        }
        last_eval["timings"] = {
            "total_iter": t4 - t0,
            "calc_Tb": t2 - t1,
            "compute_fluid_temps": t4 - t3,
        }
        last_eval["nfev"] = eval_counts["nfev"]
        params_str = ", ".join(f"{k}={v:.6g}" for k, v in last_eval["params"].items())
        print(f"Eval {eval_counts['nfev']:04d} | loss={loss:.6g} | run_t={t4 - t0:.3f}s | {params_str}")
        return loss

    return objective, last_eval


def main():
    t_start = tt.perf_counter()
    opti = parse_args()
    opti.output_dir.mkdir(parents=True, exist_ok=True)

    geo = GeometryConfig()
    fluid_cfg = FluidConfig()
    t1 = tt.perf_counter()

    df = load_measurements(opti.measurement_path)
    dt = infer_dt_seconds(df)
    t2 = tt.perf_counter()

    fluid = gt.media.Fluid(fluid_cfg.fluid_name, fluid_cfg.fluid_temp)
    cp_f = fluid.cp
    rho_f = fluid.rho
    mu_f = fluid.mu
    k_f = fluid.k
    t3 = tt.perf_counter()

    q_b_per_m = compute_q_b_per_m(df, cp_f=cp_f, H=geo.H)
    time = np.arange(1, len(df) + 1) * dt
    t4 = tt.perf_counter()

    measurement_start = df["timestamp"].iloc[0]
    if opti.date_start is not None:
        commissioning = pd.Timestamp(opti.date_start)
        if commissioning >= measurement_start:
            pre_steps = 0
        else:
            pre_steps = int(np.round((measurement_start - commissioning).total_seconds() / dt))
    else:
        pre_steps = 0

    optimize_power_start = opti.optimize_power_start and opti.date_start is not None

    borefield = create_borefield(geo)
    t1 = tt.perf_counter()
    print(f"[Runtime] section 1: setup borefield: {t1 - t_start:.2f} s")

    _, g_of_eskilson = precompute_g_function(borefield, geo.r_b)
    t2 = tt.perf_counter()
    print(f"[Runtime] section 2: calculate gfunction: {t2 - t1:.2f} s")

    pos = [(-geo.D_s, 0.0), (0.0, -geo.D_s), (geo.D_s, 0.0), (0.0, geo.D_s)]

    m_flow_borehole_ts = df["m_flow"].to_numpy()
    m_min = max(0.01, float(np.nanmin(m_flow_borehole_ts[m_flow_borehole_ts > 0])))
    m_max = float(np.nanmax(m_flow_borehole_ts) * 1.2)
    m_grid = np.linspace(m_min, m_max, 40)
    m_grid_index = np.abs(m_grid[None, :] - m_flow_borehole_ts[:, None]).argmin(axis=1)

    starts = {
        "T_s": 12.0,
        "k_s": 2.0,
        "cv_s": 2.0e6,
        "k_g": 1.0,
        "power_start": 0.0,
    }
    bounds = {
        "T_s": (10.0, 15.0),
        "k_s": (1.5, 3.0),
        "cv_s": (1.0e6, 3.0e6),
        "k_g": (0.8, 2.0),
        "power_start": (-20.0, 20.0),
    }
    steps = {
        "T_s": 0.2,
        "k_s": 0.1,
        "cv_s": 2.0e5,
        "k_g": 0.1,
        "power_start": 2.0,
    }
    penalty_weight = 0.0 if opti.no_penalty else 1.0

    if opti.sensitivity_sweep:
        run_sensitivity_sweep(
            df=df,
            dt=dt,
            time=time,
            q_b_per_m=q_b_per_m,
            borefield=borefield,
            g_of_eskilson=g_of_eskilson,
            geo=geo,
            cp_f=cp_f,
            rho_f=rho_f,
            mu_f=mu_f,
            k_f=k_f,
            pos=pos,
            m_grid=m_grid,
            m_grid_index=m_grid_index,
            pre_steps=pre_steps,
            starts=starts,
            bounds=bounds,
            optimize_power_start=optimize_power_start,
            sweep_points=opti.sweep_points,
            out_dir=opti.output_dir,
        )
        print(f"[Total Runtime] {tt.perf_counter() - t_start:.2f} s")
        return

    if opti.sweep_2d:
        run_sensitivity_sweep_2d(
            df=df,
            dt=dt,
            time=time,
            q_b_per_m=q_b_per_m,
            borefield=borefield,
            g_of_eskilson=g_of_eskilson,
            geo=geo,
            cp_f=cp_f,
            rho_f=rho_f,
            mu_f=mu_f,
            k_f=k_f,
            pos=pos,
            m_grid=m_grid,
            m_grid_index=m_grid_index,
            pre_steps=pre_steps,
            starts=starts,
            bounds=bounds,
            optimize_power_start=optimize_power_start,
            sweep_points=opti.sweep_2d_points,
            out_dir=opti.output_dir,
        )
        print(f"[Total Runtime] {tt.perf_counter() - t_start:.2f} s")
        return

    objective, last_eval = objective_factory(
        df=df,
        q_b_per_m=q_b_per_m,
        time=time,
        dt=dt,
        borefield=borefield,
        g_of_eskilson=g_of_eskilson,
        geo=geo,
        cp_f=cp_f,
        rho_f=rho_f,
        mu_f=mu_f,
        k_f=k_f,
        k_p=0.4,
        pos=pos,
        m_grid=m_grid,
        pre_steps=pre_steps,
        optimize_power_start=optimize_power_start,
        steps=steps,
        starts=starts,
        penalty_weight=penalty_weight,
    )

    if optimize_power_start:
        x0 = np.array([starts["T_s"], starts["k_s"], starts["cv_s"], starts["k_g"], starts["power_start"]])
        bounds_list = [bounds["T_s"], bounds["k_s"], bounds["cv_s"], bounds["k_g"], bounds["power_start"]]
        names = ["T_s", "k_s", "cv_s", "k_g", "power_start"]
    else:
        x0 = np.array([starts["T_s"], starts["k_s"], starts["cv_s"], starts["k_g"]])
        bounds_list = [bounds["T_s"], bounds["k_s"], bounds["cv_s"], bounds["k_g"]]
        names = ["T_s", "k_s", "cv_s", "k_g"]
    t3 = tt.perf_counter()
    print(f"[Runtime] section 3: setup opti: {t3 - t2:.2f} s")

    iteration = {"count": 0, "nfev_prev": 0}

    def log_iteration(xk: np.ndarray):
        iteration["count"] += 1
        loss = last_eval["loss"]
        nfev_total = int(last_eval.get("nfev", 0))
        nfev_delta = nfev_total - iteration["nfev_prev"]
        iteration["nfev_prev"] = nfev_total
        current = last_eval["params"] or dict(zip(names, xk))
        if not optimize_power_start and "power_start" not in current:
            current["power_start"] = starts["power_start"]
        timings = last_eval["timings"] or {}
        params_str = ", ".join(f"{k}={v:.6g}" for k, v in current.items())
        timing_str = " | ".join(
            f"{k}={timings.get(k, float('nan')):.3f}s"
            for k in ["total_iter", "calc_Tb", "compute_fluid_temps"]
        )
        print(f"Iter {iteration['count']:03d} | loss={loss:.6g} | {params_str}")
        print(f"  timings: {timing_str}")
        print(f"  nfev: {nfev_total} (Δ{nfev_delta}) | njev: 0")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds_list,
        callback=log_iteration,
        options={"maxiter": opti.max_iter},
    )
    t4 = tt.perf_counter()
    print(f"[Runtime] section 4: total optimize: {t4 - t3:.2f} s")

    fitted = dict(zip(names, result.x))
    summary_path = opti.output_dir / "example_opti_fit.csv"

    if optimize_power_start:
        power_start = fitted["power_start"]
    else:
        power_start = starts["power_start"]

    alpha = fitted["k_s"] / fitted["cv_s"]
    load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
    time_req = load_agg.get_times_for_simulation()
    t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
    g_needed = g_of_eskilson(t_eskilson_req)
    load_agg.initialize(g_needed / (2 * np.pi * fitted["k_s"]))

    T_b = simulate_Tb(
        time=time,
        q_b_per_m=q_b_per_m,
        LoadAgg=load_agg,
        T_s=fitted["T_s"],
        dt=dt,
        pre_steps=pre_steps,
        power_start=power_start,
    )

    network_grid = build_network_grid(
        m_grid,
        borefield,
        pos,
        geo.r_in,
        geo.r_out,
        0.4,
        fitted["k_s"],
        fitted["k_g"],
        geo.epsilon,
        mu_f,
        rho_f,
        k_f,
        cp_f,
        n_pipes=geo.n_pipes,
        config="parallel",
    )
    n_boreholes = len(borefield)
    m_flow_total = df["m_flow"].to_numpy() * n_boreholes
    Q_tot = n_boreholes * geo.H * q_b_per_m
    _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
        Q_tot,
        T_b,
        m_flow_total,
        df["m_flow"].to_numpy(),
        m_grid,
        network_grid,
        cp_f,
        m_grid_index=m_grid_index,
    )
    t5 = tt.perf_counter()
    print(f"[Runtime] section 5: post-fit: {t5 - t4:.2f} s")

    df_out = df.copy()
    df_out["Tf_out_sim"] = T_f_out_sim
    df_out["residual"] = df_out["Tf_out_sim"] - df_out["Tf_out"]
    df_out.to_csv(summary_path, index=False)
    t6 = tt.perf_counter()
    print(f"[Runtime] section 6: write outputs: {t6 - t5:.2f} s")

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(df_out["timestamp"], df_out["Tf_out"], label="measured", alpha=0.8)
    ax.plot(df_out["timestamp"], df_out["Tf_out_sim"], label="simulated", alpha=0.8)
    ax.set_ylabel("Tf_out [degC]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(opti.output_dir / "example_opti_fit.png", dpi=150)
    plt.close(fig)

    params_path = opti.output_dir / "example_opti_params.txt"
    with params_path.open("w", encoding="utf-8") as handle:
        handle.write("Optimization result\n")
        handle.write(f"Success: {result.success}\n")
        handle.write(f"Message: {result.message}\n")
        handle.write("\nParameters:\n")
        for name in names:
            handle.write(f"- {name}: {fitted[name]:.6g}\n")
        if not optimize_power_start:
            handle.write(f"- power_start (fixed): {power_start:.6g}\n")

    print(f"Saved fitted outputs to {summary_path}")
    print(f"Saved plot to {opti.output_dir / 'example_opti_fit.png'}")
    print(f"Saved params to {params_path}")
    t_end = tt.perf_counter()
    print(f"[Total Runtime] {t_end - t_start:.2f} s")


if __name__ == "__main__":
    main()
