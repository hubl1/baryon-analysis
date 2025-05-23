#!/bin/bash

# Load environment paths
source params/env.sh

# --- List of charmed baryons for global fit ---
# Uncomment baryons you wish to include
baryons=(
  "LAMBDA_C"
  # "SIGMA_C"
  # "XI_C"
  # "XI_C_PRIME"
  # "XI_CC"
  # "OMEGA_C"
  # "OMEGA_CC"
  # "SIGMA_STAR_C"
  # "XI_STAR_C"
  # "OMEGA_STAR_C"
  # "XI_STAR_CC"
  # "OMEGA_STAR_CC"
  # "OMEGA_CCC"
  # "D"
  # "D_S"
  # "ETA_C"
  # "D_STAR"
  # "DS_STAR"
  # "JPSI"
  # "CHI_C0"
  # "CHI_C1"
  # "H_C"
)

# --- Systematic variations for error budgeting ---
variations=(
  "None"          # Default fit including all errors
  # "mpi"         # Exclude statistical errors of pion mass (m_pi)
  # "metas"       # Exclude statistical errors of eta_s meson mass
  # "D_S"         # Exclude statistical errors of D_s meson mass
  # "alttc_stat"  # Exclude statistical errors of lattice spacing
  # "alttc_sys"   # Shift lattice spacing (w0) by 1 sigma for systematic uncertainty
)

# --- Choose discretization error form ---
# Uncomment the desired discretization error form
fit_form="--additive_Ca"  # additive discretization error
# fit_form=""             # multiplicative discretization error (comment out above line to use)

# --- Main global fitting loop ---
for baryon in "${baryons[@]}"; do
  for variation in "${variations[@]}"; do
    echo "Performing global fit for baryon=$baryon with variation=$variation and discretization form=$([[ -n $fit_form ]] && echo 'additive' || echo 'multiplicative')"

    ./global_fit/charmed_baryon_global_fit.py "$baryon" "$variation" $fit_form "$lqcd_data_path"

  done
done

# --- Explanation of fit variations ---
# "None"        : Default fit; includes all statistical and systematic errors normally.
# "mpi"         : Exclude statistical error of pion mass (m_pi).
# "metas"       : Exclude statistical error of eta_s meson mass.
# "D_S"         : Exclude statistical error of D_s meson mass.
# "alttc_stat"  : Exclude statistical error of lattice spacing.
# "alttc_sys"   : Shift lattice spacing (w0) by 1 sigma for systematic uncertainty.

# Combine these variations to create a comprehensive error budget.
