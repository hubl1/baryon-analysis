#!/bin/bash

# --- List of baryons to analyze ---
# Uncomment the baryons you want to include
baryons=(
  #"D"
  #"D_S"
  #"ETA_C"
  #"D_STAR"
  #"DS_STAR"
  #"JPSI"
  #"LAMBDA_C"
  #"SIGMA_C"
  #"XI_C_PRIME"
  "XI_C"
  #"XI_CC"
  #"OMEGA_C"
  # "OMEGA_CC"
  #"SIGMA_STAR_C"
  #"XI_STAR_C"
  #"OMEGA_STAR_C"
  #"XI_STAR_CC"
  #"OMEGA_STAR_CC"
  #"OMEGA_CCC"
  #"CHI_C0"
  #"CHI_C1"
  #"H_C"
)

# Valence-light baryons (special case handling)
baryons_val_light=(
  "D" "D_STAR" "LAMBDA_C" "SIGMA_C" "XI_C"
  "XI_C_PRIME" "XI_CC" "SIGMA_STAR_C"
  "XI_STAR_C" "XI_STAR_CC"
)

ulimit -n 2048

# --- Load ensemble info and environment settings ---
source params/ensemble_info.sh
source params/env.sh

# --- Parse arguments ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ensemble_id: 0-14> <mode: 0 (unitary) or 1 (all combinations)>"
  exit 1
fi

nensemble=$1
mode=$2

get_ensemble_info "$nensemble"

# Helper function to check if baryon is valence-light
is_valence_light() {
  local baryon="$1"
  printf '%s\n' "${baryons_val_light[@]}" | grep -qx "$baryon"
}

# Initialize pdf files array
pdf_files=()

# --- Main analysis loop ---
for baryon in "${baryons[@]}"; do
  # Determine index arrays based on mode
  if [[ $mode -eq 0 ]]; then
    ii_list=("${ii_uni[$nensemble]}")
    jj_list=("${jj_uni[$nensemble]}")
    kk_list=(6)
  elif [[ $mode -eq 1 ]]; then
    if is_valence_light "$baryon"; then
      ii_list=(0 1 2)
    else
      ii_list=("${ii_uni[$nensemble]}")
    fi
    jj_list=(3 4 5)
    kk_list=(6 7 8)
  else
    echo "Invalid mode specified: $mode. Use 0 or 1."
    exit 1
  fi

  # Nested loop for combinations
  for ii in "${ii_list[@]}"; do
    for jj in "${jj_list[@]}"; do
      for kk in "${kk_list[@]}"; do
        tag1="mu${mm[ii]}_ms${mm[jj]}_mc${mm[kk]}_rs1"
        echo "Processing: ensemble=$nensemble ii=$ii jj=$jj kk=$kk baryon=$baryon"

        ./bootstrap_and_plot_charm_h5/bootstrap_cor_joblib.py \
          "$tag1" "$ii" "$nensemble" "${mm[ii]}" "$tag" "$baryon" 1 \
          "$jj" "${ii_uni[$nensemble]}" "${jj_uni[$nensemble]}" "$kk" \
          "$lqcd_data_path"

        pdf_files+=("temp/combined_pngs/${names[nensemble]}_${tag1}_${baryon}.pdf")
      done
    done
  done
done

# --- PDF consolidation ---
output_pdf="$lqcd_data_path/figures/charmed_baryon_2pt.pdf"
existing_pdfs=()

# Check PDFs existence
for pdf in "${pdf_files[@]}"; do
  if [[ -f $pdf ]]; then
    existing_pdfs+=("$pdf")
  else
    echo "⚠️ Missing PDF: $pdf"
  fi
done

if [[ ${#existing_pdfs[@]} -gt 0 ]]; then
  pdfunite "${existing_pdfs[@]}" "$output_pdf"
  echo "✅ PDFs combined successfully into $output_pdf"
else
  echo "❌ No PDFs were generated. Skipping PDF combination."
fi
