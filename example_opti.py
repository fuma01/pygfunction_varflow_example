# -*- coding: utf-8 -*-
"""Example: parameter estimation from synthetic borefield measurements.

This script loads measurements (Tf_in, Tf_out, m_flow) and estimates
T_s, k_s, cv_s, k_g and optionally power_start to match Tf_out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pygfunction as gt
from scipy.optimize import minimize
import time as tt
from shared_utils import (
    build_network_grid,
    compute_fluid_temperatures_with_network_grid,
    create_borefield,
    load_config,
    merge_config,
    plot_fit_timeseries,
    plot_input,
    plot_sensitivity_2d,
    plot_sensitivity_combined,
    plot_sensitivity_param,
    precompute_g_function,
    rmse,
    simulate_Tb,
)


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


def run_sensitivity_sweep(
    *,
    df: pd.DataFrame,
    dt: float,
    time: np.ndarray,
    q_b_per_m: np.ndarray,
    m_flow_borehole_ts: np.ndarray,
    mask_fit: np.ndarray,
    mask_power: np.ndarray | None,
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
    m_grid_index: np.ndarray,
    pre_steps: int,
    starts: dict,
    bounds: dict,
    optimize_power_start: bool,
    sweep_points: int,
    out_dir: Path,
):
    n_boreholes = len(borefield)
    tf_out_meas = df["Tf_out"].to_numpy()

    cache = {}
    eval_cache = {}
    eval_cache = {}

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

    sweep_params = ["T_g", "k_s", "cv_s", "k_g"]
    if optimize_power_start:
        sweep_params.append("power_start")

    summary_rows = []
    combined = {}
    eval_counter = 0
    for param in sweep_params:
        values = np.linspace(bounds[param][0], bounds[param][1], sweep_points)
        rows = []
    m_flow_start_idx = np.abs(m_grid - starts["m_flow_start"]).argmin()
    m_grid_index_eff = m_grid_index
    if mask_power is not None:
        m_grid_index_eff = m_grid_index.copy()
        m_grid_index_eff[mask_power] = m_flow_start_idx
        for val in values:
            t_eval_start = tt.perf_counter()
            params = dict(starts)
            params[param] = float(val)
            if not optimize_power_start:
                params["power_start"] = starts["power_start"]

            q_b_eff = q_b_per_m.copy()
            m_flow_eff = m_flow_borehole_ts.copy()
            if mask_power is not None:
                q_b_eff[mask_power] = params["power_start"]
                m_flow_eff[mask_power] = starts["m_flow_start"]
            Q_tot = n_boreholes * geo.H * q_b_eff
            m_flow_total = m_flow_eff * n_boreholes

            alpha = params["k_s"] / params["cv_s"]
            load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
            n_boreholes = len(borefield)
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

            sweep_params = ["T_g", "k_s", "cv_s", "k_g"]
            if optimize_power_start:
                sweep_params.append("power_start")

            m_flow_start_idx = np.abs(m_grid - starts["m_flow_start"]).argmin()
            m_grid_index_eff = m_grid_index
            if mask_power is not None:
                m_grid_index_eff = m_grid_index.copy()
                m_grid_index_eff[mask_power] = m_flow_start_idx

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

                    q_b_eff = q_b_per_m.copy()
                    m_flow_eff = m_flow_borehole_ts.copy()
                    if mask_power is not None:
                        q_b_eff[mask_power] = params["power_start"]
                        m_flow_eff[mask_power] = starts["m_flow_start"]
                    Q_tot = n_boreholes * geo.H * q_b_eff
                    m_flow_total = m_flow_eff * n_boreholes

                    alpha = params["k_s"] / params["cv_s"]
                    load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
                    time_req = load_agg.get_times_for_simulation()
                    t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
                    g_needed = g_of_eskilson(t_eskilson_req)
                    load_agg.initialize(g_needed / (2 * np.pi * params["k_s"]))

                    T_b = simulate_Tb(
                        time=time,
                        q_b_per_m=q_b_eff,
                        LoadAgg=load_agg,
                        T_s=params["T_g"],
                        dt=dt,
                        pre_steps=pre_steps,
                        power_start=params["power_start"],
                    )

                    network_grid = get_network_grid(params["k_s"], params["k_g"])
                    _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
                        Q_tot,
                        T_b,
                        m_flow_total,
                        m_flow_eff,
                        m_grid,
                        network_grid,
                        cp_f,
                        m_grid_index=m_grid_index_eff,
                    )

                    rmse_out = rmse(T_f_out_sim[mask_fit], tf_out_meas[mask_fit])
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

                plot_sensitivity_param(df_param, param, out_dir / f"sensitivity_{param}.png")

            pd.DataFrame(summary_rows).to_csv(out_dir / "sensitivity_summary.csv", index=False)

            plot_sensitivity_combined(combined, out_dir / "sensitivity_combined.png")
    z_rmse = np.zeros((len(cv_s_vals), len(k_s_vals)), dtype=float)
    m_flow_start_idx = np.abs(m_grid - starts["m_flow_start"]).argmin()
    m_grid_index_eff = m_grid_index
    if mask_power is not None:
        m_grid_index_eff = m_grid_index.copy()
        m_grid_index_eff[mask_power] = m_flow_start_idx
    eval_counter = 0
    for i, cv_s in enumerate(cv_s_vals):
        for j, k_s in enumerate(k_s_vals):
            t_eval_start = tt.perf_counter()
            params = dict(starts)
            params["cv_s"] = float(cv_s)
            params["k_s"] = float(k_s)
            if not optimize_power_start:
                params["power_start"] = starts["power_start"]

            q_b_eff = q_b_per_m.copy()
            m_flow_eff = m_flow_borehole_ts.copy()
            if mask_power is not None:
                q_b_eff[mask_power] = params["power_start"]
                m_flow_eff[mask_power] = starts["m_flow_start"]
            Q_tot = n_boreholes * geo.H * q_b_eff
            m_flow_total = m_flow_eff * n_boreholes

            alpha = params["k_s"] / params["cv_s"]
            load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
            time_req = load_agg.get_times_for_simulation()
            t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
            g_needed = g_of_eskilson(t_eskilson_req)
            load_agg.initialize(g_needed / (2 * np.pi * params["k_s"]))

            T_b = simulate_Tb(
                time=time,
                q_b_per_m=q_b_eff,
                LoadAgg=load_agg,
                T_s=params["T_g"],
                dt=dt,
                pre_steps=pre_steps,
                power_start=params["power_start"],
            )

            network_grid = get_network_grid(params["k_s"], params["k_g"])
            _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
                Q_tot,
                T_b,
                m_flow_total,
                m_flow_eff,
                m_grid,
                network_grid,
                cp_f,
                m_grid_index=m_grid_index_eff,
            )

            rmse_out = rmse(T_f_out_sim[mask_fit], tf_out_meas[mask_fit])
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

    plot_sensitivity_2d(ks_grid, cvs_grid, z_rmse, out_dir / "sensitivity_2d_k_s_cv_s.png")


def objective_factory(
    *,
    df: pd.DataFrame,
    q_b_per_m: np.ndarray,
    m_flow_borehole_ts: np.ndarray,
    mask_fit: np.ndarray,
    mask_power: np.ndarray | None,
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
    bounds: dict,
    penalty_weight: float,
):
    cache = {}
    eval_cache = {}
    last_eval = {
        "loss": None,
        "params": None,
        "timings": None,
        "nfev": 0,
    }
    eval_counts = {"calls": 0, "unique": 0}
    n_boreholes = len(borefield)
    m_flow_total = m_flow_borehole_ts * n_boreholes
    m_grid_index = np.abs(m_grid[None, :] - m_flow_borehole_ts[:, None]).argmin(axis=1)
    m_flow_start_idx = np.abs(m_grid - starts["m_flow_start"]).argmin()

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
        eval_counts["calls"] += 1
        if optimize_power_start:
            T_g, k_s, cv_s, k_g, power_start = x
        else:
            T_g, k_s, cv_s, k_g = x
            power_start = starts["power_start"]

        def snap_value(value: float, name: str) -> float:
            if name not in bounds or name not in steps:
                return value
            low, high = bounds[name]
            step = steps[name]
            value = float(np.clip(value, low, high))
            return round(value / step) * step

        T_g = snap_value(T_g, "T_g")
        k_s = snap_value(k_s, "k_s")
        cv_s = snap_value(cv_s, "cv_s")
        k_g = snap_value(k_g, "k_g")
        if optimize_power_start:
            power_start = snap_value(power_start, "power_start")

        cache_key = (T_g, k_s, cv_s, k_g, power_start)
        if cache_key in eval_cache:
            cached = eval_cache[cache_key]
            last_eval["loss"] = cached["loss"]
            last_eval["params"] = cached["params"]
            last_eval["timings"] = {"total_iter": 0.0, "calc_Tb": 0.0, "compute_fluid_temps": 0.0}
            last_eval["nfev"] = eval_counts["unique"]
            return cached["loss"]

        t0 = tt.perf_counter()
        q_b_eff = q_b_per_m.copy()
        m_flow_eff = m_flow_borehole_ts.copy()
        m_grid_index_eff = m_grid_index
        if mask_power is not None:
            q_b_eff[mask_power] = power_start
            m_flow_eff[mask_power] = starts["m_flow_start"]
            m_grid_index_eff = m_grid_index.copy()
            m_grid_index_eff[mask_power] = m_flow_start_idx
        Q_tot = n_boreholes * geo.H * q_b_eff
        m_flow_total = m_flow_eff * n_boreholes

        alpha = k_s / cv_s

        load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
        time_req = load_agg.get_times_for_simulation()
        t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
        g_needed = g_of_eskilson(t_eskilson_req)
        load_agg.initialize(g_needed / (2 * np.pi * k_s))
        t1 = tt.perf_counter()

        T_b = simulate_Tb(
            time=time,
            q_b_per_m=q_b_eff,
            LoadAgg=load_agg,
            T_s=T_g,
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
            m_flow_eff,
            m_grid,
            network_grid,
            cp_f,
            m_grid_index=m_grid_index_eff,
        )
        t4 = tt.perf_counter()

        resid = T_f_out_sim[mask_fit] - df["Tf_out"].to_numpy()[mask_fit]
        mse = np.nanmean(resid**2)

        penalty = 0.0
        for name, step in steps.items():
            if name == "power_start":
                continue
            penalty += ((locals()[name] - starts[name]) / step) ** 2
        loss = mse + penalty_weight * penalty
        last_eval["loss"] = loss
        last_eval["params"] = {
            "T_g": T_g,
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
        eval_counts["unique"] += 1
        last_eval["nfev"] = eval_counts["unique"]
        params_str = ", ".join(f"{k}={v:.6g}" for k, v in last_eval["params"].items())
        print(f"Eval {eval_counts['unique']:04d} | loss={loss:.6g} | run_t={t4 - t0:.3f}s | {params_str}")
        eval_cache[cache_key] = {"loss": loss, "params": dict(last_eval["params"])}
        return loss

    return objective, last_eval


def main():
    t_start = tt.perf_counter()
    base_dir = Path(__file__).resolve().parent
    common_cfg = load_config(base_dir / "inputs" / "common.json")
    config = merge_config(common_cfg, load_config(base_dir / "inputs" / "opti.json"))

    run_cfg = config.get("run", {})
    measurement_path = base_dir / run_cfg.get("measurements", "outputs/varflow/varflow_measurements.csv")
    date_start = run_cfg.get("date_start")
    max_iter = int(run_cfg.get("max_iter", 30))
    optimize_power_start = bool(run_cfg.get("optimize_power_start", False))
    penalty = bool(run_cfg.get("penalty", True))
    sensitivity_sweep = bool(run_cfg.get("sensitivity_sweep", False))
    sweep_points = int(run_cfg.get("sweep_points", 9))
    sweep_2d = bool(run_cfg.get("sweep_2d", False))
    sweep_2d_points = int(run_cfg.get("sweep_2d_points", 9))

    outputs_cfg = config.get("outputs", {})
    base_dir = Path(__file__).resolve().parent
    default_opti_dir = base_dir / outputs_cfg.get("opti", "outputs/opti")
    default_sensitivity_dir = base_dir / outputs_cfg.get("sensitivity", "outputs/sensitivity")

    output_override = run_cfg.get("output_dir")
    if output_override:
        output_dir = base_dir / output_override
    else:
        output_dir = default_sensitivity_dir if (sensitivity_sweep or sweep_2d) else default_opti_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    geo_cfg = config.get("geometry", {})
    geo = GeometryConfig(
        D=float(geo_cfg.get("D", GeometryConfig.D)),
        H=float(geo_cfg.get("H", GeometryConfig.H)),
        r_b=float(geo_cfg.get("r_b", GeometryConfig.r_b)),
        N_1=int(geo_cfg.get("N_1", GeometryConfig.N_1)),
        N_2=int(geo_cfg.get("N_2", GeometryConfig.N_2)),
        B=float(geo_cfg.get("B", GeometryConfig.B)),
        r_out=float(geo_cfg.get("r_out", GeometryConfig.r_out)),
        r_in=float(geo_cfg.get("r_in", GeometryConfig.r_in)),
        D_s=float(geo_cfg.get("D_s", GeometryConfig.D_s)),
        epsilon=GeometryConfig.epsilon,
        n_pipes=int(geo_cfg.get("n_pipes", GeometryConfig.n_pipes)),
    )

    fluid_cfg_data = config.get("fluid", {})
    fluid_cfg = FluidConfig(
        fluid_name=fluid_cfg_data.get("fluid_name", FluidConfig.fluid_name),
        fluid_temp=float(fluid_cfg_data.get("fluid_temp", FluidConfig.fluid_temp)),
    )
    t1 = tt.perf_counter()

    df = load_measurements(measurement_path)
    measurement_begin = run_cfg.get("measurement_begin")
    if measurement_begin:
        measurement_begin_ts = pd.Timestamp(measurement_begin)
        mask_fit = df["timestamp"] >= measurement_begin_ts
        mask_power = df["timestamp"] < measurement_begin_ts
    else:
        mask_fit = np.ones(len(df), dtype=bool)
        mask_power = None
    dt = infer_dt_seconds(df)
    t2 = tt.perf_counter()

    fluid = gt.media.Fluid(fluid_cfg.fluid_name, fluid_cfg.fluid_temp)
    cp_f = fluid.cp
    rho_f = fluid.rho
    mu_f = fluid.mu
    k_f = fluid.k
    common_mat_cfg = config.get("common_material", {})
    k_p = float(common_mat_cfg.get("k_p", 0.4))
    epsilon = float(common_mat_cfg.get("epsilon", geo.epsilon))
    geo.epsilon = epsilon
    t3 = tt.perf_counter()

    q_b_per_m = compute_q_b_per_m(df, cp_f=cp_f, H=geo.H)
    time = np.arange(1, len(df) + 1) * dt
    t4 = tt.perf_counter()

    measurement_start = df["timestamp"].iloc[0]
    if date_start is not None:
        commissioning = pd.Timestamp(date_start)
        if commissioning >= measurement_start:
            pre_steps = 0
        else:
            pre_steps = int(np.round((measurement_start - commissioning).total_seconds() / dt))
    else:
        pre_steps = 0

    optimize_power_start = optimize_power_start

    borefield = create_borefield(
        N_1=geo.N_1,
        N_2=geo.N_2,
        B=geo.B,
        H=geo.H,
        D=geo.D,
        r_b=geo.r_b,
    )
    t1 = tt.perf_counter()
    print(f"[Runtime] section 1: setup borefield: {t1 - t_start:.2f} s")

    _, g_of_eskilson = precompute_g_function(borefield=borefield, r_b=geo.r_b)
    t2 = tt.perf_counter()
    print(f"[Runtime] section 2: calculate gfunction: {t2 - t1:.2f} s")

    pos = [(-geo.D_s, 0.0), (0.0, -geo.D_s), (geo.D_s, 0.0), (0.0, geo.D_s)]

    m_flow_borehole_ts = df["m_flow"].to_numpy()

    opti_cfg = config.get("opti", {})
    starts = opti_cfg.get(
        "starts",
        {"T_g": 12.0, "k_s": 2.0, "cv_s": 2.0, "k_g": 1.0, "power_start": 0.0, "m_flow_start": 0.35},
    )
    bounds_raw = opti_cfg.get(
        "bounds",
        {
            "T_g": [10.0, 15.0],
            "k_s": [1.5, 3.0],
            "cv_s": [1.0, 3.0],
            "k_g": [0.8, 2.0],
            "power_start": [-20.0, 20.0],
        },
    )
    bounds = {k: tuple(v) for k, v in bounds_raw.items()}
    steps = opti_cfg.get(
        "steps",
        {"T_g": 0.2, "k_s": 0.1, "cv_s": 0.2, "k_g": 0.1, "power_start": 2.0},
    )
    starts = dict(starts)
    bounds = dict(bounds)
    steps = dict(steps)
    starts["cv_s"] = float(starts["cv_s"]) * 1.0e6
    bounds["cv_s"] = (float(bounds["cv_s"][0]) * 1.0e6, float(bounds["cv_s"][1]) * 1.0e6)
    steps["cv_s"] = float(steps["cv_s"]) * 1.0e6
    penalty_weight = float(opti_cfg.get("penalty_weight", 1.0)) if penalty else 0.0
    print(f"[Penalty] enabled={bool(penalty)} | weight={penalty_weight}")

    def snap_value(value: float, name: str) -> float:
        if name not in bounds or name not in steps:
            return float(value)
        low, high = bounds[name]
        step = steps[name]
        value = float(np.clip(value, low, high))
        return round(value / step) * step

    m_min_meas = float(np.nanmin(m_flow_borehole_ts[m_flow_borehole_ts > 0]))
    m_max_meas = float(np.nanmax(m_flow_borehole_ts))
    m_min = max(0.01, min(m_min_meas, float(starts["m_flow_start"])))
    m_max = max(m_max_meas, float(starts["m_flow_start"])) * 1.2
    m_grid = np.linspace(m_min, m_max, 40)
    m_grid_index = np.abs(m_grid[None, :] - m_flow_borehole_ts[:, None]).argmin(axis=1)

    if sensitivity_sweep:
        run_sensitivity_sweep(
            df=df,
            dt=dt,
            time=time,
            q_b_per_m=q_b_per_m,
            m_flow_borehole_ts=m_flow_borehole_ts,
            mask_fit=mask_fit,
            mask_power=mask_power,
            borefield=borefield,
            g_of_eskilson=g_of_eskilson,
            geo=geo,
            cp_f=cp_f,
            rho_f=rho_f,
            mu_f=mu_f,
            k_f=k_f,
            k_p=k_p,
            pos=pos,
            m_grid=m_grid,
            m_grid_index=m_grid_index,
            pre_steps=pre_steps,
            starts=starts,
            bounds=bounds,
            optimize_power_start=optimize_power_start,
            sweep_points=sweep_points,
            out_dir=output_dir,
        )
        print(f"[Total Runtime] {tt.perf_counter() - t_start:.2f} s")
        return

    if sweep_2d:
        run_sensitivity_sweep_2d(
            df=df,
            dt=dt,
            time=time,
            q_b_per_m=q_b_per_m,
            m_flow_borehole_ts=m_flow_borehole_ts,
            mask_fit=mask_fit,
            mask_power=mask_power,
            borefield=borefield,
            g_of_eskilson=g_of_eskilson,
            geo=geo,
            cp_f=cp_f,
            rho_f=rho_f,
            mu_f=mu_f,
            k_f=k_f,
            k_p=k_p,
            pos=pos,
            m_grid=m_grid,
            m_grid_index=m_grid_index,
            pre_steps=pre_steps,
            starts=starts,
            bounds=bounds,
            optimize_power_start=optimize_power_start,
            sweep_points=sweep_2d_points,
            out_dir=output_dir,
        )
        print(f"[Total Runtime] {tt.perf_counter() - t_start:.2f} s")
        return

    if optimize_power_start:
        objective_stage1, _ = objective_factory(
            df=df,
            q_b_per_m=q_b_per_m,
            m_flow_borehole_ts=m_flow_borehole_ts,
            mask_fit=mask_fit,
            mask_power=mask_power,
            time=time,
            dt=dt,
            borefield=borefield,
            g_of_eskilson=g_of_eskilson,
            geo=geo,
            cp_f=cp_f,
            rho_f=rho_f,
            mu_f=mu_f,
            k_f=k_f,
            k_p=k_p,
            pos=pos,
            m_grid=m_grid,
            pre_steps=pre_steps,
            optimize_power_start=True,
            steps=steps,
            starts=starts,
            bounds=bounds,
            penalty_weight=0.0,
        )

        def objective_power_only(xp: np.ndarray) -> float:
            x_full = np.array(
                [
                    starts["T_g"],
                    starts["k_s"],
                    starts["cv_s"],
                    starts["k_g"],
                    float(xp[0]),
                ]
            )
            return objective_stage1(x_full)

        t3 = tt.perf_counter()
        print(f"[Runtime] section 3: setup opti: {t3 - t2:.2f} s")
        result_stage1 = minimize(
            objective_power_only,
            np.array([starts["power_start"]]),
            method="Powell",
            bounds=[bounds["power_start"]],
            options={"maxiter": max_iter, "disp": True},
        )
        starts["power_start"] = snap_value(float(result_stage1.x[0]), "power_start")
        optimize_power_start = False
        print(f"[Stage 1] power_start optimized (Powell): {starts['power_start']:.6g} W/m")

    objective, last_eval = objective_factory(
        df=df,
        q_b_per_m=q_b_per_m,
        m_flow_borehole_ts=m_flow_borehole_ts,
        mask_fit=mask_fit,
        mask_power=mask_power,
        time=time,
        dt=dt,
        borefield=borefield,
        g_of_eskilson=g_of_eskilson,
        geo=geo,
        cp_f=cp_f,
        rho_f=rho_f,
        mu_f=mu_f,
        k_f=k_f,
        k_p=k_p,
        pos=pos,
        m_grid=m_grid,
        pre_steps=pre_steps,
        optimize_power_start=optimize_power_start,
        steps=steps,
        starts=starts,
        bounds=bounds,
        penalty_weight=penalty_weight,
    )

    x0 = np.array([starts["T_g"], starts["k_s"], starts["cv_s"], starts["k_g"]])
    bounds_list = [bounds["T_g"], bounds["k_s"], bounds["cv_s"], bounds["k_g"]]
    names = ["T_g", "k_s", "cv_s", "k_g"]
    if not optimize_power_start:
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
        method="Powell",
        bounds=bounds_list,
        callback=log_iteration,
        options={"maxiter": max_iter},
    )
    t4 = tt.perf_counter()
    print(f"[Runtime] section 4: total optimize: {t4 - t3:.2f} s")

    fitted = dict(zip(names, result.x))
    for name in list(fitted.keys()):
        fitted[name] = snap_value(fitted[name], name)
    summary_path = output_dir / "example_opti_fit.csv"

    if optimize_power_start:
        power_start = fitted["power_start"]
    else:
        power_start = starts["power_start"]

    q_b_eff = q_b_per_m.copy()
    m_flow_eff = m_flow_borehole_ts.copy()
    if mask_power is not None:
        q_b_eff[mask_power] = power_start
        m_flow_eff[mask_power] = starts["m_flow_start"]
    m_flow_start_idx = np.abs(m_grid - starts["m_flow_start"]).argmin()
    m_grid_index_eff = m_grid_index
    if mask_power is not None:
        m_grid_index_eff = m_grid_index.copy()
        m_grid_index_eff[mask_power] = m_flow_start_idx

    alpha = fitted["k_s"] / fitted["cv_s"]
    load_agg = gt.load_aggregation.ClaessonJaved(dt, dt * (pre_steps + len(time)))
    time_req = load_agg.get_times_for_simulation()
    t_eskilson_req = alpha * np.asarray(time_req) / (geo.r_b**2)
    g_needed = g_of_eskilson(t_eskilson_req)
    load_agg.initialize(g_needed / (2 * np.pi * fitted["k_s"]))

    T_b = simulate_Tb(
        time=time,
        q_b_per_m=q_b_eff,
        LoadAgg=load_agg,
        T_s=fitted["T_g"],
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
        k_p,
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
    m_flow_total = m_flow_eff * n_boreholes
    Q_tot = n_boreholes * geo.H * q_b_eff
    _, T_f_out_sim = compute_fluid_temperatures_with_network_grid(
        Q_tot,
        T_b,
        m_flow_total,
        m_flow_eff,
        m_grid,
        network_grid,
        cp_f,
        m_grid_index=m_grid_index_eff,
    )
    t5 = tt.perf_counter()
    print(f"[Runtime] section 5: post-fit: {t5 - t4:.2f} s")

    df_out = df.copy()
    df_out["Tf_out_sim"] = T_f_out_sim
    df_out["residual"] = df_out["Tf_out_sim"] - df_out["Tf_out"]
    df_out.to_csv(summary_path, index=False)
    t6 = tt.perf_counter()
    print(f"[Runtime] section 6: write outputs: {t6 - t5:.2f} s")

    df_plot = df_out.copy()
    df_plot.loc[~mask_fit, "Tf_out"] = np.nan
    plot_fit_timeseries(df_plot, output_dir / "example_opti_fit.png")

    hours = np.arange(1, len(time) + 1) * dt / 3600.0
    plot_input(
        hours=hours,
        q_b_per_m=q_b_eff,
        m_flow_borehole_ts=m_flow_eff,
        out_path=output_dir / "example_opti_input.png",
    )

    params_path = output_dir / "example_opti_params.txt"
    with params_path.open("w", encoding="utf-8") as handle:
        handle.write("Optimization result\n")
        handle.write(f"Success: {result.success}\n")
        handle.write(f"Message: {result.message}\n")
        handle.write(f"Iterations (nit): {getattr(result, 'nit', 'n/a')}\n")
        handle.write(f"Evaluations (nfev): {last_eval.get('nfev', getattr(result, 'nfev', 'n/a'))}\n")
        handle.write(f"Final loss: {getattr(result, 'fun', 'n/a')}\n")
        handle.write(f"Penalty weight: {penalty_weight}\n")
        handle.write(f"Optimize power_start: {optimize_power_start}\n")
        units = {
            "T_g": "degC",
            "k_s": "W/mK",
            "cv_s": "MJ/m³/K",
            "k_g": "W/mK",
            "power_start": "W/m",
            "m_flow_start": "kg/s",
        }
        def format_value(name, value):
            if name == "cv_s":
                value = value / 1.0e6
            unit = units.get(name, "")
            return f"{value:.6g} {unit}".rstrip()
        handle.write("\nParameters:\n")
        for name in names:
            handle.write(f"- {name}: {format_value(name, fitted[name])}\n")
        if not optimize_power_start:
            handle.write(f"- power_start (fixed): {format_value('power_start', power_start)}\n")
        if "m_flow_start" in starts:
            handle.write(f"- m_flow_start (fixed): {format_value('m_flow_start', starts['m_flow_start'])}\n")
        handle.write("\nStart values:\n")
        for key, value in starts.items():
            handle.write(f"- {key}: {format_value(key, value)}\n")
        handle.write("\nBounds:\n")
        for key, value in bounds.items():
            low = format_value(key, value[0])
            high = format_value(key, value[1])
            handle.write(f"- {key}: [{low}, {high}]\n")
        handle.write("\nSteps:\n")
        for key, value in steps.items():
            handle.write(f"- {key}: {format_value(key, value)}\n")

    print(f"Saved fitted outputs to {summary_path}")
    print(f"Saved plot to {output_dir / 'example_opti_fit.png'}")
    print(f"Saved params to {params_path}")
    t_end = tt.perf_counter()
    print(f"[Total Runtime] {t_end - t_start:.2f} s")


if __name__ == "__main__":
    main()
