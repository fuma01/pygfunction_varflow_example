# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system with pygfunction
    Editor: martin.fuchsluger@geosphere.at
    Date: 2026-01-17

    This script simulates a geothermal borefield with time-varying load and mass flow rate using pygfunction.
    It computes a geometry-only g-function (Eskilson time) and uses the Claesson–Javed load aggregation
    method to obtain the borehole wall temperature from the load per meter.

    The borehole thermal resistance R_b is precomputed on a mass flow grid using the pygfunction network model
    and then interpolated during the time simulation. Fluid temperatures are calculated using a grid of network
    objects, fully accounting for the m_flow dependency.

    Outputs include temperature histories, load/flow inputs, R_b vs. mass flow, the Eskilson g-function,
    and borefield layout/cross-section plots.

    Key points:
    - The g-function is geometry-based (Eskilson time), independent of R_b and m_flow.
    - T_b is computed using load aggregation and the g-function.
    - R_b(m_flow) is precomputed on a grid and interpolated in time.
    - Fluid temperatures are calculated using a network grid for full m_flow dependency.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pygfunction as gt
import time as tt
from shared_utils import (
    build_network_grid,
    compute_Rb_grid_for_plot,
    compute_fluid_temperatures_with_network_grid,
    create_borefield,
    load_config,
    merge_config,
    plot_borefield_cross_section,
    plot_g_function,
    plot_input,
    plot_rb_vs_mflow,
    plot_results,
    precompute_g_function,
    simulate_Tb,
    synthetic_load_and_flow,
)

def main():
    t_start = tt.perf_counter()
    base_dir = Path(__file__).resolve().parent
    common_cfg = load_config(base_dir / "inputs" / "common.json")
    config = merge_config(common_cfg, load_config(base_dir / "inputs" / "varflow.json"))
    # -------------------------------------------------------------------------
    # Simulation parameters
    # Editor: martin.fuchsluger@geosphere.at | Date: 2026-01-17
    # -------------------------------------------------------------------------
    # Output directory
    out_dir = Path(__file__).resolve().parent / config.get("outputs", {}).get("varflow", "outputs/varflow")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Borehole dimensions
    geo_cfg = config.get("geometry", {})
    D = float(geo_cfg.get("D", 4.0))
    H = float(geo_cfg.get("H", 150.0))
    r_b = float(geo_cfg.get("r_b", 0.075))

    # Bore field geometry
    N_1 = int(geo_cfg.get("N_1", 2))
    N_2 = int(geo_cfg.get("N_2", 4))
    B = float(geo_cfg.get("B", 7.5))

    # Pipe dimensions
    r_out = float(geo_cfg.get("r_out", 0.016))
    r_in = float(geo_cfg.get("r_in", 0.014))
    D_s = float(geo_cfg.get("D_s", 0.04))
    common_mat_cfg = config.get("common_material", {})
    epsilon = float(common_mat_cfg.get("epsilon", 1.0e-6))

    # Pipe positions (Double U-tube)
    n_pipes = int(geo_cfg.get("n_pipes", 2))
    pos = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Ground properties
    ground_cfg = config.get("ground", {})
    alpha = float(ground_cfg.get("alpha", 1.0e-6))
    k_s = float(ground_cfg.get("k_s", 2.0))
    T_g = float(ground_cfg.get("T_g", 12.0))

    # Grout properties
    k_g = float(ground_cfg.get("k_g", 1.0))

    # Pipe properties
    k_p = float(common_mat_cfg.get("k_p", 0.4))

    # Fluid properties
    fluid_cfg = config.get("fluid", {})
    fluid = gt.media.Fluid(fluid_cfg.get("fluid_name", "MEA"), float(fluid_cfg.get("fluid_temp", 12.0)))
    cp_f = fluid.cp
    rho_f = fluid.rho
    mu_f = fluid.mu
    k_f = fluid.k

    # Simulation parameters
    varflow_cfg = config.get("varflow", {})
    dt_hours = float(varflow_cfg.get("dt_hours", 1.0))
    dt = dt_hours * 3600.0
    tmax_years = float(varflow_cfg.get("tmax_years", 2.0))
    tmax = tmax_years * 8760.0 * 3600.0
    Nt = int(np.ceil(tmax / dt))
    time = dt * np.arange(1, Nt + 1)

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)

    # -------------------------------------------------------------------------
    # Initialize bore field (geometry only)
    # -------------------------------------------------------------------------

    borefield = create_borefield(N_1=N_1, N_2=N_2, B=B, H=H, D=D, r_b=r_b)
    nBoreholes = len(borefield)
    
    # -------------------------------------------------------------------------
    # Calculate g-function on a fixed t_eskilson grid (independent of alpha)
    # -------------------------------------------------------------------------

    t_eskilson_grid, g_of_eskilson = precompute_g_function(borefield=borefield, r_b=r_b)

    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    t_eskilson_req = alpha * np.asarray(time_req) / (r_b**2)
    g_needed = g_of_eskilson(t_eskilson_req)
    LoadAgg.initialize(g_needed / (2 * np.pi * k_s))

    t1 = tt.perf_counter()
    print(f"[Runtime] section 1: setup gfunction: {t1 - t_start:.2f} s")

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    # 1) Calculate T_b (borehole wall temperature)
    q_b_per_m, m_flow_borehole_ts = synthetic_load_and_flow(
        time / 3600.0,
        max_w_per_m=float(varflow_cfg.get("max_w_per_m", 30.0)),
        cp_f=cp_f,
        H=H,
    )
    Q_tot = nBoreholes * H * q_b_per_m

    T_b = simulate_Tb(time=time, q_b_per_m=q_b_per_m, LoadAgg=LoadAgg, T_g=T_g)

    t2 = tt.perf_counter()
    print(f"[Runtime] section 2: calc T_b: {t2 - t1:.2f} s")

    # 2) Create mass flow grid and network grid for R_b(m_flow) and T_f calculation
    m_flow_total = m_flow_borehole_ts * nBoreholes

    m_min = float(0.01)
    m_max = float(np.nanmax(m_flow_borehole_ts)*1.2)
    m_grid = np.linspace(m_min, m_max, 50)  # 50 grid points

    network_grid = build_network_grid(
        m_grid, borefield, pos, r_in, r_out, k_p, k_s, k_g, epsilon, mu_f, rho_f, k_f, cp_f, n_pipes=n_pipes, config="parallel"
    )
    t3 = tt.perf_counter()
    print(f"[Runtime] section 3: build network grid: {t3 - t2:.2f} s")
    
    # 3) Calculate fluid temperatures using the network grid
    T_f_in, T_f_out = compute_fluid_temperatures_with_network_grid(
        Q_tot, T_b, m_flow_total, m_flow_borehole_ts, m_grid, network_grid, cp_f
    )
    date_cfg = config.get("dates", {})
    simulation_start = date_cfg.get("simulation_start", "2020-01-01")
    timestamps = pd.date_range(
        start=pd.Timestamp(simulation_start),
        periods=Nt,
        freq=pd.to_timedelta(dt, unit="s"),
    )
    df_measurements = pd.DataFrame(
        {
            "timestamp": timestamps,
            "Tf_in": T_f_in,
            "Tf_out": T_f_out,
            "m_flow": m_flow_borehole_ts,
        }
    )
    df_measurements.to_csv(out_dir / "varflow_measurements.csv", index=False)
    t4 = tt.perf_counter()
    print(f"[Runtime] section 4: compute fluid temps: {t4 - t3:.2f} s")

    # -------------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------------
    # Calculate R_b for R_b vs. m_flow plot
    Rb_grid = compute_Rb_grid_for_plot(m_grid, network_grid, nBoreholes, cp_f)
    R_b_ts = np.interp(m_flow_borehole_ts, m_grid, Rb_grid)
    # Setze R_b_ts auf NaN, wenn m_flow_borehole_ts <= 0
    R_b_ts[m_flow_borehole_ts <= 0] = np.nan
    # Optionally plot R_b at unique simulation values if less than 25 unique values
    m_sim_uni = np.unique(m_flow_borehole_ts[(~np.isnan(m_flow_borehole_ts)) & (m_flow_borehole_ts != 0)])
    if len(m_sim_uni) <= 25:
        Rb_sim = np.interp(m_sim_uni, m_grid, Rb_grid)
    else:
        Rb_sim = np.array([])

    # time for plots
    hours = np.arange(1, Nt+1) * dt / 3600.

    # Plot g-function vs. Eskilson-Zeit
    plot_g_function(
        t_eskilson_grid=t_eskilson_grid,
        g_func=g_of_eskilson(t_eskilson_grid),
        t_eskilson_req=t_eskilson_req,
        out_path=out_dir / "g_function_eskilson.png",
    )
    
    # Cross-section plot (MultipleUTube)
    plot_borefield_cross_section(
        borefield=borefield,
        pos=pos,
        r_in=r_in,
        r_out=r_out,
        n_pipes=n_pipes,
        out_path=out_dir / "BHE_cross_section.png",
    )

    # Input plot (q_b and m_flow)
    plot_input(
        hours=hours,
        q_b_per_m=q_b_per_m,
        m_flow_borehole_ts=m_flow_borehole_ts,
        out_path=out_dir / "example_varflow_input.png",
    )

    # Results plot (T_b, R_b, T_f, delta_T)
    plot_results(
        hours=hours,
        T_b=T_b,
        R_b_ts=R_b_ts,
        T_f_in=T_f_in,
        T_f_out=T_f_out,
        out_path=out_dir / "example_varflow_results.png",
    )
    
    # Plot R_b vs m_flow (per borehole) and indicate laminar/turbulent transition
    plot_rb_vs_mflow(
        m_grid=m_grid,
        Rb_grid=Rb_grid,
        m_sim=m_sim_uni,
        Rb_sim=Rb_sim,
        r_in=r_in,
        mu_f=mu_f,
        n_pipes=n_pipes,
        out_path=out_dir / "Rb_vs_mflow.png",
    )

    t5 = tt.perf_counter()
    print(f"[Runtime] section 5: plotting: {t5 - t4:.2f} s")
    print(f"[Total Runtime] {t5 - t_start:.2f} s")
    return

if __name__ == '__main__':
    main()