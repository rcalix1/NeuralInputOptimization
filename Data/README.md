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


# Synthetic Portfolio Construction Dataset (NIO Ready)

This synthetic dataset simulates a portfolio allocation scenario and is designed for training and evaluating Neural Input Optimization (NIO) models.

## 📦 Dataset Overview

* **Samples:** 1000
* **Inputs:** 7 continuous variables (`i_` prefix)
* **Outputs:** 4 continuous outcomes (`o_` prefix)
* **Purpose:** Use NIO to generate investment allocations that meet specific performance and risk constraints.

## 📅 Input Features (7)

| Column Name            | Description                        |
| ---------------------- | ---------------------------------- |
| `i_stock_pct`          | Allocation to stocks (%)           |
| `i_bond_pct`           | Allocation to bonds (%)            |
| `i_crypto_pct`         | Allocation to cryptocurrencies (%) |
| `i_real_estate_pct`    | Allocation to real estate (%)      |
| `i_cash_pct`           | Allocation to cash (%)             |
| `i_risk_appetite`      | User’s risk preference (0–1 scale) |
| `i_time_horizon_years` | Investment horizon (years)         |

## 📉 Output Features (4)

| Column Name             | Description                                |
| ----------------------- | ------------------------------------------ |
| `o_expected_return_pct` | Expected return (%)                        |
| `o_risk_score`          | Composite portfolio risk score (0–1 scale) |
| `o_max_drawdown_pct`    | Historical max drawdown (%)                |
| `o_esg_score`           | ESG performance score (0–100)              |

## 🧐 Financial Behavior (Simulated)

* **Expected return** increases with crypto, stocks, and risk appetite.
* **Risk score** increases with crypto % and risk appetite, decreases with cash.
* **Drawdown** is driven by volatile assets like crypto and stocks.
* **ESG score** is boosted by bonds, real estate, and cash; reduced by crypto.

## 🔁 Suggested NIO Use

This dataset is designed for inverse modeling experiments such as:

> "Find allocations that give expected return ≥ 8%, risk ≤ 0.4, and ESG score ≥ 60."

Use your forward model to learn `x → y`, and NIO to find `x` that satisfies output constraints.

