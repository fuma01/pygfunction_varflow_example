# Example: Variable Mass Flow Borefield Simulation

This example (`example_varflow.py`) simulates a geothermal borefield with a time‑varying load and a mass flow rate that depends on the load. The borehole wall temperature is computed using the Claesson–Javed load aggregation method and an Eskilson g‑function.

## Key Idea: g‑Function Independent of Material Parameters and Mass Flow

The g‑function is expressed using the Eskilson time:

t_esk = alpha * t / r_b^2

When time is scaled to t_esk, the g‑function depends only on the borefield geometry and the boundary condition (here: UBWT).  
It is independent of:
- ground thermal properties (k_s, c_v)
- mass flow rate

Changing ground thermal properties does not change the g‑function curve in Eskilson time. 
It only shifts the mapping between real time and Eskilson time: higher alpha compresses the real‑time axis, lower alpha stretches it.

### How this is done in the script

1. A single g‑function is computed using a reference diffusivity `alpha_ref`.
2. The g‑function is stored on an Eskilson time grid and reused by interpolation for any actual `alpha`.

### Eskilson Time Grid Range

The grid is built to cover 1 hour to 100 years for all plausible ground properties:

- k_s: 1–4 W/mK  
- c_v: 1–4 MJ/m³K  

This yields:

t_esk_min = alpha_min * 1h / r_b^2  
t_esk_max = alpha_max * 100y / r_b^2

This ensures the g‑function is valid for any material parameter choice within the range.

## Borehole Wall Temperature (T_b) Does NOT Use R_b or m_flow

The borehole wall temperature is computed only from:
- the load per meter q_b  
- the g‑function  
- the load aggregation method  

R_b and mass flow are not required to compute T_b.

## Borehole Resistance (R_b) vs. Mass Flow

The dependence of R_b on mass flow is handled outside the time loop:

1. A grid of mass flow values is created.
2. For each grid point, a network‑based R_b is computed.
3. During time simulation, R_b(t) is obtained by interpolation.

This keeps the transient simulation fast and avoids recomputing the network model each time step.

## Inputs

Key inputs are defined at the top of `example_varflow.py`:

- Borefield geometry (N_1, N_2, B, H, D, r_b)
- Ground properties (k_s, alpha, T_g) and grout/pipe properties (k_g, k_p, r_in, r_out)
- Fluid properties (cp_f, rho_f, mu_f, k_f) and nominal mass flow per borehole
- Load profile and mass‑flow rule from `synthetic_load_and_flow`
- Simulation settings (time step `dt`, total duration `tmax`, aggregation options)

**Synthetic load and flow (`synthetic_load_and_flow`)**  
The load is a monthly step profile that ramps from maximum extraction to maximum recharge across the year.
The maximum load is set by the user in watts per borehole meter, and the default is 30 W/m.
The mass flow per borehole scales with the absolute load: it is zero below 10 W/m, then increases linearly between a minimum and maximum flow up to the specified peak load.

These inputs control the geometry, thermal response, and the time‑varying load/flow behavior.

## Outputs

Plots are written to the `outputs/` directory (relative to the script):

- `borefield_cross_section.png`
- `borefield_layout.png`
- `g_function_eskilson.png`
- `example_varflow_input.png`
- `example_varflow_results.png`
- `Rb_vs_mflow.png`

## Author

Martin Fuchsluger (martin.fuchsluger@geosphere.at)

## Acknowledgments

Special thanks to the authors of `pygfunction`.