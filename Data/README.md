## DATA

* my datasets

# Synthetic Materials Engineering Dataset (NIO Ready)

This synthetic dataset simulates a materials engineering process (e.g., alloy design) and is intended for training and evaluating Neural Input Optimization (NIO) models.

## 📦 Dataset Overview

* **Samples:** 1000
* **Inputs:** 7 continuous variables (`i_` prefix)
* **Outputs:** 3 continuous outcomes (`o_` prefix)
* **Purpose:** Use NIO to generate feasible input vectors that satisfy target constraints on outputs.

## 📅 Input Features (7)

| Column Name             | Description                               |
| ----------------------- | ----------------------------------------- |
| `i_iron_pct`            | Iron percentage in alloy                  |
| `i_carbon_pct`          | Carbon percentage                         |
| `i_chromium_pct`        | Chromium percentage                       |
| `i_temp_celsius`        | Heating temperature in Celsius            |
| `i_cool_rate_c_per_min` | Cooling rate (degrees Celsius per minute) |
| `i_pressure_atm`        | Pressure during processing (atm)          |
| `i_additiveA_pct`       | Additive A concentration (%)              |

## 📉 Output Features (3)

| Column Name              | Description                              |
| ------------------------ | ---------------------------------------- |
| `o_tensile_strength_mpa` | Tensile strength of final material (MPa) |
| `o_corrosion_resistance` | Resistance to corrosion (scale: 0 to 1)  |
| `o_brittleness`          | Brittleness score (scale: 0 to 10)       |

## 🧐 Physics-Inspired Relationships

* **Tensile strength** rises with carbon, pressure, iron, and additive A; peaks with moderate carbon.
* **Corrosion resistance** increases with chromium and additive A; reduced by carbon.
* **Brittleness** increases with carbon and fast cooling; reduced by pressure.

## 🔁 Suggested NIO Use

This dataset is structured for experiments like:

> "Given a desired strength ≥ 300 MPa, corrosion resistance ≥ 0.7, and brittleness ≤ 5, what input settings produce this alloy?"

Use your forward model to approximate output mappings, then apply constrained NIO for input generation.
