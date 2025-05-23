#!/bin/bash

# Load environment paths
source params/env.sh

# --- List of baryons for global fit ---
# Uncomment the baryons you wish to include
baryons=(
  "PROTON"
  # "LAMBDA"
  # "SIGMA"
  # "XI"
  # "DELTA"
  # "SIGMA_STAR"
  # "XI_STAR"
  # "OMEGA"
)

# --- List of systematic variations for error budgeting ---
variations=(
  "None"          # Default fit: includes all errors normally
  #"mpi"          # Ignore statistical errors of pion mass (m_pi)
  #"metas"        # Ignore statistical errors of eta_s meson mass
  #"alttc_stat"   # Ignore statistical errors of lattice spacing
  #"alttc_sys"    # Shift w0 by 1 sigma to estimate systematic lattice spacing error
)

# --- Choose discretization error form ---
# Uncomment the desired discretization error form
fit_form="--additive_Ca"  # additive discretization error
# fit_form=""             # multiplicative discretization error (if line above commented out)

# --- Main global fitting loop ---
for baryon in "${baryons[@]}"; do
  for variation in "${variations[@]}"; do
    echo "Performing global fit for baryon=$baryon with variation=$variation and discretization form=$([[ -n $fit_form ]] && echo 'additive' || echo 'multiplicative')"

    ./global_fit/light_baryon_global_fit.py "$baryon" "$variation" $fit_form "$lqcd_data_path"

  done
done

# --- Explanation of fit variations ---
# "None"        : Default fit; includes all statistical and systematic errors normally.
# "mpi"         : Exclude statistical error of pion mass (m_pi).
# "metas"       : Exclude statistical error of eta_s meson mass.
# "alttc_stat"  : Exclude statistical error of lattice spacing.
# "alttc_sys"   : Shift lattice spacing (w0) by 1 sigma to estimate systematic uncertainty.

# By combining results from these variations, you can build a comprehensive error budget.

