#!/bin/bash

# --- List of charm mesons to analyze ---
# Uncomment or comment mesons as needed
baryons=(
  'D_S'
)

ulimit -n 2048

# --- Load ensemble info and environment settings ---
source params/ensemble_info.sh
source params/env.sh

# --- Parse arguments ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ensemble_id (0-14)> <mode (0|1)>"
  exit 1
fi

ensemble=$1
mode=$2

# Get ensemble-specific info
get_ensemble_info "$ensemble"

# Determine indices based on mode
case "$mode" in
  0)
    ii_list=(${ii_uni[$ensemble]})
    jj_list=(${jj_uni[$ensemble]})
    kk_list=(${kk_phy[$ensemble]})
    ;;
  1)
    ii_list=(0 1 2)
    jj_list=(3 4 5)
    kk_list=(6 7 8)
    ;;
  *)
    echo "Invalid mode specified: $mode. Use 0 or 1."
    exit 1
    ;;
esac

# Initialize PDF files array
pdf_files=()

# --- Main analysis loop ---
for baryon in "${baryons[@]}"; do
  for ii in "${ii_list[@]}"; do
    for jj in "${jj_list[@]}"; do
      for kk in "${kk_list[@]}"; do
        rs=1
        tag1="mu${mm[ii]}_ms${mm[jj]}_mc${mm[kk]}_rs${rs}"
        echo "Processing: ensemble=$ensemble ii=$ii jj=$jj kk=$kk meson=$baryon"

        ./bootstrap_and_plot_charm_quark_h5/bootstrap_cor_joblib.py \
          "$tag1" "$ii" "$ensemble" "${mm[ii]}" "$tag" "$baryon" 1 \
          "$jj" "${ii_uni[$ensemble]}" "${jj_uni[$ensemble]}" "$kk" "$lqcd_data_path"

        pdf_files+=("temp/combined_pngs/${names[ensemble]}_${tag1}_${baryon}.pdf")
      done
    done
  done
done

# --- PDF consolidation ---
output_pdf="$lqcd_data_path/figures/charm_quark_2pt.pdf"
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
  echo "❌ No PDFs generated, skipping PDF combination."
fi
