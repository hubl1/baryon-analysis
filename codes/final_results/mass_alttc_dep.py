# Adjust the previous script according to the new request:
# 1. Move 'PROTON' to the bottom plot.
# 2. Only show the x-axis label for 'PROTON', omit it for the upper two plots.

# Here is the updated version of the script:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from params.plt_labels import plot_configurations
import json
import sys


lqcd_data_path=sys.argv[1]
pydata_path=lqcd_data_path+'/precomputed_pydata_global_fit'
plt.style.use("utils/science.mplstyle")
json_file = f"{lqcd_data_path}/final_results/data_error_budget.json"
with open(json_file, "r") as f:
    data_json = json.load(f)
baryons = [
    "SIGMA_C", "XI_C","XI_C_PRIME", "OMEGA_C", "SIGMA_STAR_C", "XI_STAR_C",
    "OMEGA_STAR_C", "XI_CC", "OMEGA_CC", "XI_STAR_CC",
    "OMEGA_STAR_CC", "OMEGA_CCC"
]
light_baryons = [
    "PROTON", "LAMBDA", "SIGMA", "XI", "DELTA", "SIGMA_STAR",
    "XI_STAR", "OMEGA"
]
plot_baryons = [ 'OMEGA_C','OMEGA','PROTON']
plot_tag=['C','B','A']

# Set error bar parameters
errorbar_kwargs = {
    "linewidth": 0.8, "elinewidth": 0.8, "capthick": 1,
    "capsize": 2, "mew": 0.8, "linestyle": "none",
    "fillstyle": "none", "markersize": 5
}
errorbar_kwargs_modified = errorbar_kwargs.copy()
errorbar_kwargs_modified.pop('fillstyle', None)

# Create a 3-row, 1-column plot layout
fig, axs = plt.subplots(3, 1, figsize=(4, 4), dpi=140)

for j, (baryon, ax) in enumerate(zip(plot_baryons, axs)):
    # Load data
    mass_cntrl=float(data_json[baryon]['mass']['cntrl'])
    mass_err=(float(data_json[baryon]['mass']['total stat'])**2+float(data_json[baryon]['mass']['alttc_sys'])**2+float(data_json[baryon]['mass']['fit ansatz'])**2)**0.5
    if baryon not in light_baryons:
        data = np.load(f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy", allow_pickle=True).item()
        chi2=max(data['fit_chi2'],1)
    else:
        data = np.load(f"{pydata_path}/{baryon}_mass_extra_plot_data_addCa_True_None.npy", allow_pickle=True).item()
        chi2=max(data['fit_chi2'],1)

    # Calculate displacements
    displacements = np.zeros(len(data["a_2"]))
    unique_a2 = np.unique(data["a_2"])

    for value in unique_a2:
        indices = np.where(data["a_2"] == value)[0]
        n = len(indices)
        offsets = np.linspace(-0.00010 * (n - 1), 0.00010 * (n - 1), n)
        displacements[indices] = offsets

    # Plot each baryon
    for i in range(len(data["combined_data_err"])):
        ax.errorbar(
            data["a_2"][i] + displacements[i],
            data["combined_data_Ds_phy"][i],
            yerr=data["combined_data_err"][i],
            color=data["colors"][i],
            label=data["labels"][i] if j == 0 else "",
            marker=data["markers"][i],
            **errorbar_kwargs
        )
    ax.scatter(
        -0.00015,float(plot_configurations[baryon]["mass"]) / 1000 , color="red", marker="*", zorder=50, s=20
    )  # s设置星星的大小，zorder确保星星在其他元素上面
    total_stat_err=data["fit_fcn_err"]*chi2**(1/2)
    ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="-")
    ax.fill_between(
        data["fit_a_2"],
        # data["fit_fcn"] - data["fit_fcn_err"],
        # data["fit_fcn"] + data["fit_fcn_err"],
        data["fit_fcn"] - total_stat_err,
        data["fit_fcn"] + total_stat_err,
        alpha=0.5, color="grey", edgecolor="none"
    )

    ax.errorbar(
        # 0+0.0002,  # Apply displacement
        0,  # Apply displacement
        mass_cntrl,
        yerr=mass_err,
        color='black',  # Automatically assign color
        label='',
        marker='',
        **errorbar_kwargs_modified,
    )

    # Set individual limits and labels
    ax.set_xlim([-0.0003, 0.013])
    if j == 0:
        # ax.set_ylim([2.59, 2.73])
        ax.set_ylim([2.58, 2.78])
    elif j == 1:
        ax.set_ylim([1.52, 1.72])
    else:
        ax.set_ylim([0.831, 1.031])
        ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")  # Only add x-axis label for the bottom plot

    if j < 2:  # Hide x-axis labels for the upper two plots
        ax.set_xticklabels([])
    # if j > 0:
    #     ax.yaxis.set_major_formatter(mticker.NullFormatter())



    # latex_baryon='$m_'+plot_configurations[baryon]['label_y']+'$'
    # Generate a properly formatted LaTeX string for the y-label
    latex_baryon = r'$m_{' + plot_configurations[baryon]['label_y'].replace('$','') + '}$'
    print(latex_baryon)
    ax.text(-0.16, 1, plot_tag[j], transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.set_ylabel(latex_baryon+'(GeV)')

# Save the modified plot
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.savefig(f"{lqcd_data_path}/final_results/mass_alttc.pdf")
