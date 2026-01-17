# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system with pygfunction
    Editor: martin.fuchsluger@geosphere.at
    Date: 2026-01-17

    This script simulates a geothermal borefield with a time‑varying load and a
    variable mass flow rate. It computes a geometry‑only
    g‑function on Eskilson time and uses the Claesson–Javed load aggregation
    method to obtain the borehole wall temperature from the load per meter.

    Ground thermal properties only shift the mapping between real time and
    Eskilson time via alpha; the g‑function curve itself is unchanged. The
    borehole thermal resistance R_b is precomputed on a mass‑flow grid using the
    pygfunction network model and then interpolated during the time simulation.

    Outputs include temperature histories, load/flow inputs, R_b vs. mass flow,
    the Eskilson g‑function, and borefield layout/cross‑section plots.

    Key idea: g‑function is geometry‑based (Eskilson time), T_b uses only load
    aggregation + g‑function (no R_b, no m_flow), and R_b(m_flow) is computed
    once on a grid and interpolated in time.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

from pathlib import Path

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # Editor: martin.fuchsluger@geosphere.at | Date: 2026-01-17
    # -------------------------------------------------------------------------
    # Output directory
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)

    # Bore field geometry (rectangular array 4 x 8)
    N_1 = 2             # Number of boreholes in x-direction (columns)
    N_2 = 4             # Number of boreholes in y-direction (rows)
    B = 7.5             # Borehole spacing (m)

    # Pipe dimensions
    r_out = 0.016      # Pipe outer radius (m)
    r_in = 0.014       # Pipe inner radius (m)
    D_s = 0.04         # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions (Double U-tube)
    pos = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)
    T_g = 10.0          # Undisturbed ground temperature (degC)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow_borehole = 0.25      # kg/s per borehole (nominal for pipe resistance)
    fluid = gt.media.Fluid('MEA', 12.)
    cp_f = fluid.cp
    rho_f = fluid.rho
    mu_f = fluid.mu
    k_f = fluid.k

    # g-Function calculation options
    options = {'nSegments': 8,
               'disp': True}

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 2. * 8760. * 3600.    # Maximum time (s)
    Nt = int(np.ceil(tmax / dt))
    time = dt * np.arange(1, Nt + 1)

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)

    # -------------------------------------------------------------------------
    # Initialize bore field (geometry only)
    # -------------------------------------------------------------------------

    borefield = gt.borefield.Borefield.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    nBoreholes = len(borefield)
    
    # Cross-section plot (MultipleUTube)
    plot_borefield_cross_section(
        borefield=borefield,
        pos=pos,
        r_in=r_in,
        r_out=r_out,
        k_s=k_s,
        k_g=k_g,
        k_p=k_p,
        m_flow_borehole=m_flow_borehole,
        mu_f=mu_f,
        rho_f=rho_f,
        k_f=k_f,
        cp_f=cp_f,
        epsilon=epsilon,
        out_path=out_dir / "borefield_cross_section.png",
    )

    # -------------------------------------------------------------------------
    # Calculate g-function on a fixed t_eskilson grid (independent of alpha)
    # -------------------------------------------------------------------------

    # Bounds for alpha from ks_min/max and cv_min/max
    ks_min, ks_max = 1.0, 4.0          # W/mK
    cv_min, cv_max = 1.0, 4.0          # MJ/m3/K

    alpha_min = ks_min / (cv_max * 1.0e6)
    alpha_max = ks_max / (cv_min * 1.0e6)

    # t range: 1h to 100 years
    t_min = 3600.0
    t_max = 100.0 * 8760.0 * 3600.0

    t_eskilson_min = alpha_min * t_min / (r_b**2)
    t_eskilson_max = alpha_max * t_max / (r_b**2)

    t_eskilson_grid = np.logspace(np.log10(t_eskilson_min), np.log10(t_eskilson_max), 150)

    # Compute g-function once with reference alpha (UBWT, no flow dependency)
    alpha_ref = 1.0e-6
    t_grid = t_eskilson_grid * (r_b**2) / alpha_ref
    gFunc = gt.gfunction.gFunction(
        borefield, alpha_ref, time=t_grid, boundary_condition='UBWT', options=options
    )
    g_of_eskilson = interp1d(t_eskilson_grid, gFunc.gFunc, kind="linear", fill_value="extrapolate")

    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    t_eskilson_req = alpha * np.asarray(time_req) / (r_b**2)
    g_needed = g_of_eskilson(t_eskilson_req)
    LoadAgg.initialize(g_needed / (2 * np.pi * k_s))

    # Plot g-function vs. Eskilson-Zeit
    plot_g_function(
        t_eskilson_grid=t_eskilson_grid,
        g_func=gFunc.gFunc,
        t_eskilson_req=t_eskilson_req,
        out_path=out_dir / "g_function_eskilson.png",
    )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    # 1) Variablen für T_b
    q_b_per_m, m_flow_borehole_ts = synthetic_load_and_flow(
        time / 3600., max_w_per_m=30.0
    )
    Q_tot = nBoreholes * H * q_b_per_m

    T_b = simulate_Tb(
        time=time,
        q_b_per_m=q_b_per_m,
        LoadAgg=LoadAgg,
        T_g=T_g,
    )

    # 2) Variablen für m_flow / R_b (Network nur auf Raster)
    m_flow_total = m_flow_borehole_ts * nBoreholes

    m_min = float(np.nanmin(m_flow_borehole_ts))
    m_max = float(np.nanmax(m_flow_borehole_ts))
    m_grid = np.linspace(m_min, m_max, 50)  # 50 Punkte

    Rb_grid = compute_Rb_grid_with_network(
        m_grid=m_grid,
        borefield=borefield,
        pos=pos,
        r_in=r_in,
        r_out=r_out,
        k_p=k_p,
        k_s=k_s,
        k_g=k_g,
        epsilon=epsilon,
        mu_f=mu_f,
        rho_f=rho_f,
        k_f=k_f,
        cp_f=cp_f,
        n_pipes=2,
        config="parallel",
    )

    R_b_ts = np.interp(m_flow_borehole_ts, m_grid, Rb_grid)

    # 3) T_f berechnen
    T_f_in, T_f_out = compute_fluid_temperatures(
        Q_tot=Q_tot,
        T_b=T_b,
        m_flow=m_flow_total,
        cp_f=cp_f,
        divider=nBoreholes * H,
        Rb_field=R_b_ts,
    )

    # -------------------------------------------------------------------------
    # Calculate exact solution from convolution in the Fourier domain
    # -------------------------------------------------------------------------

    Q_b = H * q_b_per_m
    dQ = np.zeros(Nt)
    dQ[0] = Q_b[0]
    dQ[1:] = Q_b[1:] - Q_b[:-1]

    t_eskilson_time = alpha * np.asarray(time) / (r_b**2)
    g = g_of_eskilson(t_eskilson_time)

    T_b_exact = T_g - fftconvolve(
        dQ, g / (2.0 * np.pi * k_s * H), mode='full')[0:Nt]

    # -------------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------------

    hours = np.arange(1, Nt+1) * dt / 3600.

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
    
    # Plot R_b vs m_flow (per borehole) + Reynolds-Zahl auf 2. y-Achse
    plot_rb_vs_mflow(
        m_grid=m_grid,
        Rb_grid=Rb_grid,
        r_in=r_in,
        mu_f=mu_f,
        out_path=out_dir / "Rb_vs_mflow.png",
    )



def simulate_Tb(*, time, q_b_per_m, LoadAgg, T_g):
    """
    Time loop for borehole wall temperature T_b.
    """
    Nt = len(time)
    T_b = np.zeros(Nt)

    for i, (t, q_b_i) in enumerate(zip(time, q_b_per_m)):
        LoadAgg.next_time_step(t)
        LoadAgg.set_current_load(q_b_i)
        deltaT_b = LoadAgg.temporal_superposition()
        T_b[i] = T_g - deltaT_b

    return T_b


def compute_Rb_grid_with_network(
    *,
    m_grid: np.ndarray,
    borefield,
    pos,
    r_in: float,
    r_out: float,
    k_p: float,
    k_s: float,
    k_g: float,
    epsilon: float,
    mu_f: float,
    rho_f: float,
    k_f: float,
    cp_f: float,
    n_pipes: int = 2,
    config: str = "parallel",
) -> np.ndarray:
    m_grid = np.asarray(m_grid, dtype=float)
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in, r_out, k_p)
    n_bh = len(borefield)

    Rb_grid = np.full_like(m_grid, np.nan, dtype=float)
    for i, m_bh in enumerate(m_grid):
        if not np.isfinite(m_bh) or m_bh <= 0:
            continue

        m_flow_pipe = m_bh / float(n_pipes)
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon
        )
        R_f = 1.0 / (h_f * 2.0 * np.pi * r_in)
        R_fp = R_f + R_p

        UTubes = [
            gt.pipes.MultipleUTube(
                pos, r_in, r_out, borehole, k_s, k_g, R_fp,
                nPipes=n_pipes, config=config
            )
            for borehole in borefield
        ]
        network = gt.networks.Network(borefield, UTubes)

        m_flow_network = m_bh * n_bh
        Rb_grid[i] = gt.networks.network_thermal_resistance(network, m_flow_network, cp_f)

    return Rb_grid


def compute_fluid_temperatures(
    *,
    Q_tot: np.ndarray,
    T_b: np.ndarray,
    m_flow: np.ndarray,
    cp_f: float,
    divider: float,
    Rb_field: np.ndarray,
):
    T_f_in = T_b.copy()
    T_f_out = T_b.copy()

    mask = m_flow > 0
    if np.any(mask):
        Rb_use = np.asarray(Rb_field)
        if Rb_use.ndim == 0:
            Rb_use = float(Rb_use)
        else:
            Rb_use = Rb_use[mask]

        T_f_in[mask] = (-Q_tot[mask] / 2.0 / m_flow[mask] / cp_f) + (
            -Q_tot[mask] / divider * Rb_use
        ) + T_b[mask]
        T_f_out[mask] = 2.0 * (-Q_tot[mask] / divider * Rb_use + T_b[mask]) - T_f_in[mask]

    return T_f_in, T_f_out


def synthetic_load_and_flow(x, *, max_w_per_m: float = 30.0):
    """
    Monthly step load: max extraction in January, max recharge ~6 months later.

    Returns:
      - q_b_per_m: load per meter (W/m)
        (both set to 0 when |q_b_per_m| < 10 W/m)
    """
    hours = np.asarray(x, dtype=float)
    hours_in_year = 24.0 * 365.0
    day_of_year = (hours % hours_in_year) / 24.0

    month_starts = np.array(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
        dtype=float,
    )
    month_index = np.searchsorted(month_starts[1:], day_of_year, side="right")

    max_w = float(max_w_per_m)
    ramp1 = np.linspace(max_w, -max_w, 8)
    ramp2 = np.linspace(-max_w, max_w, 6)
    monthly_values_per_m = np.concatenate([ramp1, ramp2[1:5]])

    q_b_per_m = monthly_values_per_m[month_index].astype(float)

    abs_q = np.abs(q_b_per_m)
    min_flow = 0.1
    max_flow = 0.5
    q_ref = 10.0

    m_flow_borehole = np.zeros_like(abs_q, dtype=float)
    if max_w_per_m > q_ref:
        slope = (max_flow - min_flow) / (max_w_per_m - q_ref)
        m_flow_borehole = np.where(
            abs_q >= q_ref,
            np.clip(min_flow + slope * (abs_q - q_ref), min_flow, max_flow),
            0.0,
        )

    q_b_per_m = np.where(abs_q < q_ref, 0.0, q_b_per_m)

    return q_b_per_m, m_flow_borehole

def plot_borefield_cross_section(
    *,
    borefield,
    pos,
    r_in: float,
    r_out: float,
    k_s: float,
    k_g: float,
    k_p: float,
    m_flow_borehole: float,
    mu_f: float,
    rho_f: float,
    k_f: float,
    cp_f: float,
    epsilon: float,
    out_path: Path,
):
    fig_cs, ax_cs = plt.subplots(1, 1, figsize=(6, 6))

    sample_borehole = borefield[0]
    m_flow_pipe = m_flow_borehole / 2.0
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon
    )
    R_f = 1.0 / (h_f * 2.0 * np.pi * r_in)
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in, r_out, k_p)
    R_fp = R_f + R_p

    u_tube = gt.pipes.MultipleUTube(
        pos, r_in, r_out, sample_borehole, k_s, k_g, R_fp,
        nPipes=2, config="parallel"
    )
    u_tube.visualize_pipes()

    ax_cs.set_aspect("equal", "box")
    fig_cs.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_cs)

    # Field layout plot (top + 3D)
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
    ax_g.plot(t_eskilson_grid, g_func, color="tab:blue")
    ax_g.set_xscale("log")
    ax_g.set_xlabel(r'$t_{\mathrm{eskilson}}$')
    ax_g.set_ylabel('g-Funktion')
    ax_g.grid(True, which="both", alpha=0.3)

    t_req_min = np.nanmin(t_eskilson_req)
    t_req_max = np.nanmax(t_eskilson_req)
    ax_g.axvspan(t_req_min, t_req_max, color="tab:orange", alpha=0.2, label="t_eskilson (Simulation)")
    ax_g.axvline(t_req_min, color="tab:orange", alpha=0.6, linestyle="--")
    ax_g.axvline(t_req_max, color="tab:orange", alpha=0.6, linestyle="--")
    ax_g.legend()

    fig_g.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_g)


def plot_input(*, hours, q_b_per_m, m_flow_borehole_ts, out_path: Path):
    fig_in, (ax_q, ax_m) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_q.set_ylabel(r'$q_b$ [W/m]')
    gt.utilities._format_axes(ax_q)
    ax_q.plot(hours, q_b_per_m)
    ax_q.grid(True, which="both", alpha=0.2)

    ax_m.set_xlabel(r'$t$ [hours]')
    ax_m.set_ylabel(r'$\dot{m}$ [kg/s]')
    gt.utilities._format_axes(ax_m)
    ax_m.plot(hours, m_flow_borehole_ts)
    ax_m.grid(True, which="both", alpha=0.2)

    fig_in.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_in)


def plot_results(*, hours, T_b, R_b_ts, T_f_in, T_f_out, out_path: Path):
    fig_res, (ax_r1, ax_r2, ax_r3, ax_r4) = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    ax_r1.set_ylabel(r'$T_b$ [degC]')
    gt.utilities._format_axes(ax_r1)
    ax_r1.plot(hours, T_b)
    ax_r1.grid(True, which="both", alpha=0.2)

    ax_r2.set_ylabel(r'$R_b$ [K/W]')
    gt.utilities._format_axes(ax_r2)
    ax_r2.plot(hours, R_b_ts)
    ax_r2.grid(True, which="both", alpha=0.2)

    ax_r3.set_ylabel(r'$T_f$ [degC]')
    gt.utilities._format_axes(ax_r3)
    ax_r3.plot(hours, T_f_in, label=r'$T_{f,in}$')
    ax_r3.plot(hours, T_f_out, label=r'$T_{f,out}$')
    ax_r3.grid(True, which="both", alpha=0.2)
    ax_r3.legend()

    delta_T = np.abs(T_f_in - T_f_out)
    ax_r4.set_xlabel(r'$t$ [hours]')
    ax_r4.set_ylabel(r'$\Delta T$ [degC]')
    gt.utilities._format_axes(ax_r4)
    ax_r4.plot(hours, delta_T)
    ax_r4.grid(True, which="both", alpha=0.2)

    fig_res.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_res)


def plot_rb_vs_mflow(*, m_grid, Rb_grid, r_in: float, mu_f: float, out_path: Path):
    fig_rb, ax_rb = plt.subplots(1, 1, figsize=(9, 4))

    ax_rb.plot(m_grid, Rb_grid, marker="o", linestyle="-", label=r"$R_b$")

    D_h = 2.0 * r_in
    m_flow_pipe_transition = 2300.0 * np.pi * D_h * mu_f / 4.0
    m_flow_bh_transition = m_flow_pipe_transition * 2.0

    ax_rb.axvspan(m_grid.min(), min(m_flow_bh_transition, m_grid.max()),
                  color="tab:blue", alpha=0.08, label="Laminar (Re < 2300)")
    if m_flow_bh_transition < m_grid.max():
        ax_rb.axvspan(m_flow_bh_transition, m_grid.max(),
                      color="tab:orange", alpha=0.08, label="Turbulent (Re > 2300)")

    ax_rb.set_xlabel(r'$\dot{m}$ [kg/s] (pro Bohrloch)')
    ax_rb.set_ylabel(r'$R_b$ [K/W]')
    ax_rb.set_ylim(0.0, 0.5)
    ax_rb.grid(True, which="both", alpha=0.2)
    ax_rb.legend(loc="best")

    fig_rb.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig_rb)

if __name__ == '__main__':
    main()