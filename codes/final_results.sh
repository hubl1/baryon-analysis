#!/bin/bash
source params/env.sh

# python3 ./final_results/error_budget_json_gen.py $lqcd_data_path
python3 ./final_results/baryon_mass_plot_using_json.py $lqcd_data_path
# python3 ./final_results/sigma_Ha_plot_using_json.py $lqcd_data_path
# python3 ./final_results/mass_alttc_dep.py $lqcd_data_path
# python3 ./final_results/all_mass_alttc_dep.py $lqcd_data_path
# python3 ./final_results/all_sigmaH_alttc_dep.py $lqcd_data_path
# python3 ./final_results/ratio_3in1_using_json.py $lqcd_data_path
# python3 ./final_results/meson_sigma_Ha_plot_using_json.py $lqcd_data_path
