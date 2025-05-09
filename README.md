# 🧪 Baryon Mass and Sigma-Term Analysis from Lattice QCD

This repository provides scripts, input files, and plotting utilities necessary for extracting baryon and meson masses, effective masses, and sigma terms from 2-point correlator data generated in lattice QCD simulations.

---

## 📁 Repository Structure

```
.
├── bootstrap_and_plot_*/        # 2-point fits to extract effective masses
├── global_fit/                  # Physical extrapolations
├── final_results/               # Final plots and JSON data
├── params/                      # Ensemble parameters, fit priors, and environment settings
├── temp/                        # Intermediate results (PNG, PDF, NumPy; auto-generated, not for human use)
├── utils/                       # Common plotting and I/O utilities
├── *.sh                         # Launcher scripts for analysis
```

---

## ⚙️ Installation & Data Setup (Docker)

The entire analysis workflow is packaged in a Docker image for convenience:

```bash
docker pull <docker_image_name>

docker run --rm -v /path/to/lqcd_data:/lqcd_data_path <docker_image_name>
```

The `/lqcd_data_path` directory should contain your raw `.h5` 2-point correlator files, propagators, gauge configurations, and previous outputs. This is also prepared for compatibility with another Docker image (`chroma-meas`) for direct baryon 2-point calculations.
set $lqcd_data_path in params/env.sh

---

## 🚀 Quick Start

Run each command from the repository root to reproduce the entire analysis pipeline:

```bash
./*baryon_2pt.sh 1 0          # Light baryon fit and plot of fit to $lqcd_data_path/figures and fit results to  $lqcd_data_path/pydata_eff_mass 1 for ensemble1 (C24P29) and second number 0/1 for unitary point all quark combinations,



./*_quark_2pt.sh               # PCAC quark masses and pseudoscalar meson decay constants
./*_global_fit.sh     #physical extraplltion
./final_results.sh                 # Generate final plots and JSON data to $lqcd_data_path/final_results
```

---

## 🔢 Inputs & Configuration

* **Fit priors** are stored as JSON files: `params/fit_init_values/*.json`
* **Ensemble definitions** (beta, mu, ms, volumes, names): `params/ensemble_info.sh`
* **Plotting styles** are defined in: `utils/physrev.mplstyle`
* **Data paths** are configured in: `params/env.sh`

---

## 🧾 Output Summary

| Location               | File Type | Description                          |
| ---------------------- | --------- | ------------------------------------ |
| `$lqcd_data_path/final_results/*.pdf`  | PDF       | Final comparison and analysis plots  |
| `$lqcd_data_path/final_results/*.json` | JSON      | Machine-readable mass & sigma tables |

Intermediate and final analysis results reside under the mounted `$lqcd_data_path`:

```
/lqcd_data_path
├── baryon_meson_data_h5           # Raw 2-point data (.h5)
├── precomputed_Propagators        # Propagator data for testing
├── precomputed_pydata_eff_mass    # Precomputed effective masses
├── precomputed_pydata_global_fit  # Precomputed global fit data
├── pydata_eff_mass                # Generated effective masses
├── pydata_global_fit              # Generated global fit data
├── final_results                  # Final analysis outputs (PDF, JSON)
├── figures                        # Additional figures and plots
```

---

## 🧭 Running 2-point Fits

The main 2-point fit script syntax:

```bash
bash charmed_baryon_2pt.sh <ensemble_id> <mode>
```

* **`ensemble_id`**: Integer from 0–14, defined in `params/ensemble_info.sh`
* **`mode`**:

  * `0`: Unitary point (valence quark masses equal sea quark masses)
  * `1`: All possible valence quark combinations (light, strange, charm)

---

The complete analysis workflow is conveniently bundled and distributed via Docker.
