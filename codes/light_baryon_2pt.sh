#!/bin/bash

# --- List of baryons to analyze ---
# Uncomment or comment baryons as needed
baryons=(
  'PROTON'
  # 'LAMBDA'
  # 'SIGMA'
  # 'XI'
  # 'DELTA'
  # 'SIGMA_STAR'
  # 'XI_STAR'
  # 'OMEGA'
)

ulimit -n 2048

# --- Load ensemble info and environment settings ---
source params/ensemble_info.sh
source params/env.sh

# --- Parse arguments ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ensemble_id (e.g., 0)> <mode (0|1|2)>"
  exit 1
fi

nensemble=$1
mode=$2

get_ensemble_info "$nensemble"
# Initialize pdf files array
pdf_files=()

# --- Main analysis loop ---
for baryon in "${baryons[@]}"; do

    # Determine indices based on mode
    case "$mode" in
      0)
        ii_list=(${ii_uni[$nensemble]})
        jj_list=(${jj_uni[$nensemble]})
        ;;
      1)
        ii_list=(0 1 2)
        jj_list=(3 4 5)
        ;;
      2)
        ii_list=(${ii_uni[$nensemble]})
        jj_list=(3 4 5)
        ;;
      *)
        echo "Invalid mode specified: $mode. Use 0, 1, or 2."
        exit 1
        ;;
    esac

    for ii in "${ii_list[@]}"; do
      for jj in "${jj_list[@]}"; do

        tag1="mu${mm[ii]}_ms${mm[jj]}_rs1"
        echo "Processing: ensemble=$nensemble ii=$ii jj=$jj baryon=$baryon"

        ./bootstrap_and_plot_light_h5/bootstrap_cor_joblib.py \
          "$tag1" "$ii" "$nensemble" "${mm[ii]}" "$tag" "$baryon" 1 \
          "$jj" "${ii_uni[$nensemble]}" "${jj_uni[$nensemble]}" "$lqcd_data_path"

        pdf_files+=("temp/combined_pngs/${names[nensemble]}_${tag1}_${baryon}.pdf")
      done
    done

  done

# --- PDF consolidation ---
output_pdf="$lqcd_data_path/figures/light_baryon_2pt.pdf"
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
