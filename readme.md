# Examples: Variable Mass Flow Simulation + Parameter Optimization

This repository contains **two examples**:

1. **`example_varflow.py`** — a compact demonstration of g‑function independence (geometry‑only) and the use of **variable mass flow**.
	Its output is a set of **synthetic measurements** used as input for the optimization example.
2. **`example_opti.py`** — a parameter estimation workflow that fits simulated outlet temperatures to the synthetic measurements.

Below you will find a **short overview**, then a **detailed description of `example_varflow.py`**, followed by a **detailed description of `example_opti.py`** (including the optimization strategy).

Because the synthetic load profile is fully known, the two examples provide full control over the optimization target. This makes them especially suitable for testing optimization cadence and behavior, including changes to start values, bounds, and step sizes.

## Example 1 (Details): `example_varflow.py`

`example_varflow.py` simulates a geothermal borefield with a time‑varying load and a variable mass flow rate. The borehole wall temperature is computed using the Claesson–Javed load aggregation method and an Eskilson g‑function. The script writes **synthetic measurements** that are used by `example_opti.py`.

### Key Idea: g‑Function Independent of Material Parameters and Mass Flow

The g‑function is expressed using the Eskilson time:

$$
t_{\mathrm{esk}} = \alpha \cdot t / r_b^2
$$

When time is scaled to $t_{\mathrm{esk}}$, the g‑function depends only on the borefield geometry and the boundary condition (here: UBWT).  
It is independent of:
- ground thermal properties ($k_s$, $c_v$)
- mass flow rate

Changing ground thermal properties does not change the g‑function curve in Eskilson time. 
It only shifts the mapping between real time and Eskilson time: higher $\alpha$ compresses the real‑time axis, lower $\alpha$ stretches it.

### How this is done in the script

1. A single g‑function is computed using a reference diffusivity `alpha_ref`.
2. The g‑function is stored on an Eskilson time grid and reused by interpolation for any actual `alpha`.

### Eskilson Time Grid Range

The grid is built to cover 1 hour to 100 years for the following ground properties range:

- $k_s$: 1–4 W/mK  
- $c_v$: 1–4 MJ/m³K  

This yields:

$$
t_{\mathrm{esk,min}} = \alpha_{\min} \cdot 1\,\mathrm{h} / r_b^2 \\
t_{\mathrm{esk,max}} = \alpha_{\max} \cdot 100\,\mathrm{y} / r_b^2
$$

This ensures the g‑function is valid for any material parameter choice within the range.

### Borehole Wall Temperature ($T_b$) Does NOT Use $R_b$ or $m_\mathrm{flow}$

The borehole wall temperature is computed only from:
- the load per meter $q_b$  
- the g‑function  
- the load aggregation method  

$R_b$ and mass flow are not required to compute $T_b$.

### Borehole Resistance ($R_b$) vs. Mass Flow

The dependence of $R_b$ on mass flow is handled outside the time loop:

1. A grid of mass flow values is created.
2. For each grid point, a network‑based $R_b$ is computed using a pre-built network grid (see below).
3. During time simulation, $R_b(t)$ is obtained by interpolation.

This keeps the transient simulation fast and avoids recomputing the network model each time step.

### Network Grid and Fluid Temperatures

- A grid of network objects is built for all relevant mass flow rates using the current geometry and fluid properties.
- Fluid temperatures ($T_{f,\mathrm{in}}$, $T_{f,\mathrm{out}}$) are computed for each time step by interpolating the network grid according to the current mass flow.

## Inputs

Inputs are defined in JSON files under `inputs/`:

- `common.json`: shared `geometry` and `fluid`
- `varflow.json`: `ground`, `varflow`, `outputs.varflow`
- `opti.json`: `opti`, `run`, `outputs.opti`, `outputs.sensitivity`

Both scripts use `inputs/common.json` plus their specific config.

Units: `cv_s` is specified in MJ/m³/K in `inputs/opti.json` and converted internally to J/m³/K.


**Synthetic load and flow (`synthetic_load_and_flow`)**  
The load profile was created as an example/test case for demonstration. It is defined by monthly steps:

| Month | Load Factor | ΔT (K) |
|-------|-------------|--------|
|   1   |   max_load  |   2    |
|   2   |   2/3       |   3    |
|   3   |   1/3       |   4    |
|   4   |   0         |   0    |
|   5   |  -1/3       |   4    |
|   6   |  -2/3       |   3    |
|   7   |  -max_load  |   2    |
|   8   |  -2/3       |   3    |
|   9   |  -1/3       |   4    |
|  10   |   0         |   0    |
|  11   |   1/3       |   4    |
|  12   |   2/3       |   3    |

For each month, the mass flow per borehole is calculated analytically so that the temperature difference ΔT matches the table above for the given load. If the load is zero, the mass flow is set to zero and the fluid temperatures (inlet and outlet) are, by definition, set to $T_b$.

## Example 2 (Details): `example_opti.py`

`example_opti.py` loads the synthetic measurements from `outputs/varflow/varflow_measurements.csv` and estimates
the parameters $T_g$, $k_s$, $c_{v,s}$, and $k_g$ to match the measured outlet temperature. It also optionally
optimizes a **pre‑measurement** `power_start` that is applied before the measurement window.

**Simulation vs. measurement start:** `power_start` is used only when the simulation start date
(`dates.simulation_start`) is earlier than the measurement start date (`run.measurement_begin`).
This decision is made automatically.

### Optimization Strategy

The optimizer is **Powell**, and the search is **snapped to the discrete grid** defined by `bounds` and `steps`.
The workflow is staged:

1. **Stage 1 (power_start only):** Powell optimizes `power_start` while the other parameters stay at their
	start values. The result is snapped to the grid.
2. **Stage 2 (material parameters):** Powell optimizes $T_g$, $k_s$, $c_{v,s}$, and $k_g$ with `power_start` fixed.
	All parameters are snapped to the grid for evaluation and reporting.

**Penalty (optional):** Controlled by `run.penalty` and `opti.penalty_weight` in `inputs/opti.json`. When enabled, the penalty is applied only to $T_g$, $k_s$, $c_{v,s}$, and $k_g$ (never to `power_start`).

Outputs (in `outputs/opti`):
- `example_opti_fit.csv` (measured vs. simulated $T_{f,out}$ and residuals)
- `example_opti_fit.png` (comparison plot)
- `example_opti_input.png` (final input load and mass flow)
- `example_opti_params.txt` (best‑fit parameters)

### `run` parameters in `inputs/opti.json`

- `measurement_begin`: Measurement window start date (plotting starts here; loss can exclude an initial buffer).
- `max_iter`: Maximum Powell iterations for each optimization stage.
- `optimize_power_start`: If true, use staged optimization with a dedicated `power_start` stage when applicable.
- `penalty`: Enable/disable the regularization penalty on $T_g$, $k_s$, $c_{v,s}$, $k_g$.
- `sensitivity_sweep`: Run 1D sensitivity sweep instead of optimization.
- `sweep_points`: Number of points per parameter in the 1D sweep.
- `sweep_2d_cv_s_k_s`: Enable 2D sweep for $c_{v,s}$ vs. $k_s$.
- `sweep_2d_T_g_power_start`: Enable 2D sweep for $T_g$ vs. `power_start`.
- `sweep_2d_points`: Grid size (per axis) for 2D sweeps.

## Outputs

Outputs are organized into subfolders (relative to the repo root):

- `outputs/varflow`: simulation plots and measurements
- `outputs/opti`: optimization results
- `outputs/sensitivity`: sensitivity sweep outputs (plots and tables)

Plots from `example_varflow.py` are written to `outputs/varflow`:

- `borefield_cross_section.png` (geometry only, no flow/material dependency)
- `borefield_layout.png`
- `g_function_eskilson.png`
- `example_varflow_input.png`
- `example_varflow_results.png`
- `Rb_vs_mflow.png`

## Runtime Measurement

The script measures and prints the runtime for the following sections:
1. Setup and g-function calculation
2. Calculation of $T_b$
3. Building the network grid
4. Fluid temperature calculation
5. Plotting and output

## Author

Martin Fuchsluger (martin.fuchsluger@geosphere.at)

## Acknowledgments

Special thanks to the authors of `pygfunction`.