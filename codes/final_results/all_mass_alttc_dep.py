import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from params.plt_labels import plot_configurations
from matplotlib.ticker import MultipleLocator
import json
import sys 

lqcd_data_path=sys.argv[1]
pydata_path=lqcd_data_path+'/precomputed_pydata_global_fit'

plt.style.use("utils/science.mplstyle")
json_file = f"{lqcd_data_path}/final_results/data_error_budget.json"
with open(json_file, "r") as f:
    data_json = json.load(f)
# print(data)
dark_cyan = (0, 0.7, 0.7)

colors = [
    "red",         # 红色
    "blue",        # 蓝色
    "green",       # 绿色
    "orange",      # 橙色
    "purple",      # 紫色
    # "cyan",        # 青色
    "magenta",     # 洋红色
    "brown",        # 棕色 (代替黄色)
    "teal",     # 蓝绿色
    # dark_cyan,
]
baryons = [
    "LAMBDA_C","SIGMA_C", "XI_C", "XI_C_PRIME","OMEGA_C", "SIGMA_STAR_C", "XI_STAR_C",
    "OMEGA_STAR_C", "XI_CC", "OMEGA_CC", "XI_STAR_CC",
    "OMEGA_STAR_CC", "OMEGA_CCC"
]
light_baryons = [
    "PROTON", "LAMBDA", "SIGMA", "XI", "DELTA", "SIGMA_STAR",
    "XI_STAR", "OMEGA"
]
charm_valance_light = [                                                                                                                                                                                                         
    "D",
    "D_STAR",
    "LAMBDA_C",
    "SIGMA_C",
    "XI_C",
    "XI_C_PRIME",
    "XI_CC",
    "SIGMA_STAR_C",
    "XI_STAR_C",
    "XI_STAR_CC",
]

# Set error bar parameters
errorbar_kwargs = {
    "linewidth": 0.5, "elinewidth": 0.8, "capthick": 0.8,
    "capsize": 2, "mew": 0.5, "linestyle": "none",
    "fillstyle": "none", "markersize": 4
}

errorbar_kwargs_modified = errorbar_kwargs.copy()
errorbar_kwargs_modified.pop('fillstyle', None)

# Define markers
markers = ["^", "<", "v", ">", "x", "s", "d", "o", "8", "p", "*", "h", "H", ".", "+"]

# Create a 3-row, 1-column plot layout for three groups
# fig, axs = plt.subplots(1, 3, figsize=(10.5, 4), dpi=140)
fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # Set width_ratios to adjust the width of the third plot

# Grouping baryons based on number of 'C' in their names
group_0 = [b for b in baryons+light_baryons if b.count("C") == 0]  # One 'C'
group_1 = [b for b in baryons+light_baryons if b.count("C") == 1]  # One 'C'
group_2 = [b for b in baryons+light_baryons if b.count("C") == 2]  # Two 'C's
group_3 = [b for b in baryons+light_baryons if b.count("C") == 3]  # Three 'C's

ylim_strt=[0.8,2.0,4.6,3.45]
ylim_end=[1.8,3.0,5.1,3.95]
# groups = [group_3, group_2, group_1,group_0]
groups1 = [group_0, group_1]
groups2 = [group_3,group_2]

# Loop over groups and subplots
for n_plt,(group) in enumerate( groups1):
    ax = fig.add_subplot(gs[0, n_plt])

    for i, baryon in enumerate(group):
        # Load data
        mass_cntrl=float(data_json[baryon]['mass']['cntrl'])
        mass_err=(float(data_json[baryon]['mass']['total stat'])**2+float(data_json[baryon]['mass']['alttc_sys'])**2+float(data_json[baryon]['mass']['fit ansatz'])**2)**0.5
        if baryon not in light_baryons:
            if baryon in charm_valance_light:
                data = np.load(f"{pydata_path}/{baryon}_mass_extra_plot_data_addCa_True_None.npy", allow_pickle=True).item()
                chi2=max(data['fit_chi2'],1)
            else:
                # data = np.load(f"./npy_error_budget/{baryon}_mass_data_addCa_True_None.npy", allow_pickle=True).item()
                data = np.load(f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy", allow_pickle=True).item()
                chi2=max(data['fit_chi2'],1)
        else:
            data = np.load(f"{pydata_path}/{baryon}_mass_extra_plot_data_addCa_True_None.npy", allow_pickle=True).item()
            chi2=max(data['fit_chi2'],1)
            # chi2=1

        # Calculate displacements
        displacements = np.zeros(len(data["a_2"]))
        unique_a2 = np.unique(data["a_2"])

        for value in unique_a2:
            indices = np.where(data["a_2"] == value)[0]
            n = len(indices)
            offsets = np.linspace(-0.00010 * (n - 1), 0.00010 * (n - 1), n)
            displacements[indices] = offsets

        # Plot each baryon with slight displacement in x-axis to avoid overlap
        for j in range(len(data["combined_data_err"])):
            ax.errorbar(
                data["a_2"][j] + displacements[j],  # Apply displacement
                data["combined_data_Ds_phy"][j],
                yerr=data["combined_data_err"][j],
                color=colors[i % 8],  # Automatically assign color
                label=plot_configurations[baryon]["label_y"] if j == 0 else "",  # Use label_y as the label, but only for the first point
                marker=markers[i % len(markers)],  # Cycle through the marker list
                **errorbar_kwargs
            )

        # Plot fit line and shaded error region
        # ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="-")
        ax.plot(data["fit_a_2"], data["fit_fcn"],color=colors[i % 8] , linestyle="-")
        total_stat_err=data["fit_fcn_err"]*chi2**(1/2)
        ax.fill_between(
            data["fit_a_2"],
            # data["fit_fcn"] - data["fit_fcn_err"],
            # data["fit_fcn"] + data["fit_fcn_err"],
            data["fit_fcn"] - total_stat_err,
            data["fit_fcn"] + total_stat_err,
            alpha=0.5, color="grey", edgecolor="none"
            # alpha=0.3,color=f"{colors[i % 5]}"  , edgecolor="none"
        )
        ax.scatter( -0.0003,float(plot_configurations[baryon]["mass"]) / 1000 , color=colors[i % 8], marker="*", zorder=50, s=20)
        ax.errorbar(
            0,  # Apply displacement
            mass_cntrl,
            yerr=mass_err,
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=markers[i % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs_modified,
        )

    ax.set_ylim([ylim_strt[n_plt],ylim_end[n_plt] ])  # Dynamically set y-limits
    ax.set_xlim([-0.0005, 0.013])
    ax.legend(loc="best", ncol=4, fontsize='small')
    ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    ax.text(0.05, 0.97, chr(ord('A')+n_plt), transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
    if n_plt==0:
        ax.set_ylabel(r"$m_H$ (GeV)")

# Loop over groups and subplots
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2])  # Create 2x1 grid within the 3rd subplot
for n_plt,group in enumerate( groups2):
    ax = fig.add_subplot(gs_right[n_plt,0])
    n_plt+=2
    for i, baryon in enumerate(group):
        mass_cntrl=float(data_json[baryon]['mass']['cntrl'])
        mass_err=(float(data_json[baryon]['mass']['total stat'])**2+float(data_json[baryon]['mass']['alttc_sys'])**2+float(data_json[baryon]['mass']['fit ansatz'])**2)**0.5
        # Load data
        # if baryon not in light_baryons:
        #     data = np.load(f"./npy/{baryon}_mass_data_addCa_True.npy", allow_pickle=True).item()
        #     data2 = np.load(f"./npy/{baryon}_mass_data_addCa_False.npy", allow_pickle=True).item()
        #     chi2=max(data['fit_chi2'],1)
        # else:
        #     data = np.load(f"../light_hadron_fit_like_proton/npy/{baryon}_mass_extra_plot_data_addCa_True.npy", allow_pickle=True).item()
        #     data2 = np.load(f"../light_hadron_fit_like_proton/npy/{baryon}_mass_extra_plot_data_addCa_False.npy", allow_pickle=True).item()
        #     # data = np.load(f"../light_hadron_fit_like_proton/npy/{baryon}_mass_extra_plot_data.npy", allow_pickle=True).item()
        #     chi2=1
        if baryon not in light_baryons:
            if baryon in charm_valance_light:
                data = np.load(f"{pydata_path}/{baryon}_mass_extra_plot_data_addCa_True_None.npy", allow_pickle=True).item()
            else:
                data = np.load(f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy", allow_pickle=True).item()
                chi2=max(data['fit_chi2'],1)
        else:
            data = np.load(f"{pydata_path}/{baryon}_mass_extra_plot_data_addCa_True_None.npy", allow_pickle=True).item()
            # data = np.load(f"../light_hadron_fit_like_proton/npy/{baryon}_mass_extra_plot_data.npy", allow_pickle=True).item()
            chi2=1

        # Calculate displacements
        displacements = np.zeros(len(data["a_2"]))
        unique_a2 = np.unique(data["a_2"])

        for value in unique_a2:
            indices = np.where(data["a_2"] == value)[0]
            n = len(indices)
            offsets = np.linspace(-0.00010 * (n - 1), 0.00010 * (n - 1), n)
            displacements[indices] = offsets

        # Plot each baryon with slight displacement in x-axis to avoid overlap
        for j in range(len(data["combined_data_err"])):
            ax.errorbar(
                data["a_2"][j] + displacements[j],  # Apply displacement
                data["combined_data_Ds_phy"][j],
                yerr=data["combined_data_err"][j],
                color=f"{colors[i % 8]}",  # Automatically assign color
                label=plot_configurations[baryon]["label_y"] if j == 0 else "",  # Use label_y as the label, but only for the first point
                marker=markers[i % len(markers)],  # Cycle through the marker list
                **errorbar_kwargs
            )

        # Plot fit line and shaded error region
        # ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="-")
        ax.plot(data["fit_a_2"], data["fit_fcn"],color=f"{colors[i % 8]}" , linestyle="-")
        total_stat_err=data["fit_fcn_err"]*chi2**(1/2)
        ax.fill_between(
            data["fit_a_2"],
            # data["fit_fcn"] - data["fit_fcn_err"],
            # data["fit_fcn"] + data["fit_fcn_err"],
            data["fit_fcn"] - total_stat_err,
            data["fit_fcn"] + total_stat_err,
            alpha=0.5, color="grey", edgecolor="none"
            # alpha=0.3,color=f"{colors[i % 5]}"  , edgecolor="none"
        )
        ax.scatter( -0.0003,float(plot_configurations[baryon]["mass"]) / 1000 , color=f"{colors[i % 8]}", marker="*", zorder=50, s=20)
        ax.errorbar(
            0,  # Apply displacement
            mass_cntrl,
            yerr=mass_err,
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=markers[i % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs_modified,
        )

    ax.set_ylim([ylim_strt[n_plt],ylim_end[n_plt] ])  # Dynamically set y-limits
    ax.set_xlim([-0.0005, 0.013])
    ax.legend(loc="best", ncol=4, fontsize='small')
    ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    ax.text(0.05, 0.95, chr(ord('F')-n_plt), transform=ax.transAxes, fontsize=14, fontweight=1000, ha='left', va='top')
    print(n_plt)
    if n_plt==2:
        ax.set_xticklabels([])  # Remove x-axis labels for upper plots
    ax.yaxis.set_major_locator(MultipleLocator(0.2))


# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(wspace=0.13, hspace=0.02)
plt.savefig(f"{lqcd_data_path}/final_results/all_mass_alttc.pdf")
