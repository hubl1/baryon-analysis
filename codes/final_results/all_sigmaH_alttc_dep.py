import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from params.plt_labels import plot_configurations
import json
import sys

lqcd_data_path=sys.argv[1]
pydata_path=lqcd_data_path+'/precomputed_pydata_global_fit'
plt.style.use("utils/science.mplstyle")

dark_cyan = (0, 0.7, 0.7)
colors = [
    "red",  # 红色
    "blue",  # 蓝色
    "green",  # 绿色
    "orange",  # 橙色
    "purple",  # 紫色
    "magenta",  # 洋红色
    "brown",  # 棕色 (代替黄色)
    # dark_cyan,  # 青色
    'teal',  # 青色
]
baryons = [
    "LAMBDA_C",
    "SIGMA_C",
    "XI_C",
    "XI_C_PRIME",
    "OMEGA_C",
    "SIGMA_STAR_C",
    "XI_STAR_C",
    "OMEGA_STAR_C",
    "XI_CC",
    "OMEGA_CC",
    "XI_STAR_CC",
    "OMEGA_STAR_CC",
    "OMEGA_CCC",
]
light_baryons = [
    "PROTON",
    "LAMBDA",
    "SIGMA",
    "XI",
    "DELTA",
    "SIGMA_STAR",
    "XI_STAR",
    "OMEGA",
]

# Set error bar parameters
errorbar_kwargs = {
    "linewidth": 0.1,
    "elinewidth": 0.8,
    "capthick": 1,
    "capsize": 2,
    "mew": 0.6,
    "linestyle": "none",
    "fillstyle": "none",
    "markersize": 5,
}

errorbar_kwargs_modified = errorbar_kwargs.copy()
errorbar_kwargs_modified.pop('fillstyle', None)
# Define markers
markers = ["^", "<", "v", ">", "x", "s", "d", "o", "8", "p", "*", "h", "H", ".", "+"]

# Load JSON data
json_file = f"{lqcd_data_path}/final_results/data_error_budget.json"
with open(json_file, "r") as f:
    data_json = json.load(f)

# Create a 3-row, 1-column plot layout for three groups
fig = plt.figure(figsize=(10.5, 4))
gs = gridspec.GridSpec(
    1, 3, width_ratios=[1, 1, 1]
)  # Set width_ratios to adjust the width of the third plot

# Grouping baryons based on number of 'C' in their names
group_0 = [b for b in baryons + light_baryons if b.count("C") == 0]  # One 'C'
group_1 = [b for b in baryons + light_baryons if b.count("C") == 1]  # One 'C'
group_2 = [b for b in baryons + light_baryons if b.count("C") == 2]  # Two 'C's
group_3 = [b for b in baryons + light_baryons if b.count("C") == 3]  # Three 'C's

ylim_strt = [0.00, 0.95, 2.85, 1.95]
ylim_end = [0.65, 1.60, 2.85 + 0.325, 2.25 + 0.325]
# groups = [group_0, group_1, group_2,group_3]
groups1 = [group_0, group_1]
groups2 = [group_3, group_2]

light_baryon_disp=[0,0,0,0,1,2,0,0]
charm_baryon_disp=[0,1,0,1,1,2,0,0]
charm_baryon_disp2=[0,1,1,2]
# Loop over groups and subplots
for n_plt, (group) in enumerate(groups1):
    ax = fig.add_subplot(gs[0, n_plt])
    if n_plt==0:
        baryon_0_disp=light_baryon_disp
    else:
        baryon_0_disp=charm_baryon_disp
    for i, baryon in enumerate(group):
        # Load data
        if baryon not in light_baryons:
            data = np.load(
                f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy", allow_pickle=True
            ).item()
            # data2 = np.load(
            #     f"./npy/{baryon}_Hm_data_addCa_False.npy", allow_pickle=True
            # ).item()
            # chi2 = max(data["fit_ch2"], 1)
        else:
            data = np.load(
                f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                allow_pickle=True,
            ).item()
            # data2 = np.load(
            #     f"../light_hadron_fit_like_proton/npy/{baryon}_Hm_data_addCa_False.npy",
            #     allow_pickle=True,
            # ).item()
            # chi2 = 1

        if baryon in light_baryons:
            Hm_cntrl=float(data_json[baryon]['Hm']['cntrl'])
            Hm_err=(float(data_json[baryon]['Hm']['total stat'])**2+float(data_json[baryon]['Hm']['alttc_sys'])**2+float(data_json[baryon]['Hm']['fit ansatz'])**2+float(data_json[baryon]['Hm']['correction'])**2)**0.5
        else:
            Hm_cntrl=float(data_json[baryon]['Hm']['cntrl'])
            Hm_err=(float(data_json[baryon]['Hm']['total stat'])**2+float(data_json[baryon]['Hm']['alttc_sys'])**2+float(data_json[baryon]['Hm']['fit ansatz'])**2+float(data_json[baryon]['Hm']['correction'])**2+float(data_json[baryon]['Hm']['D_S'])**2)**0.5
        Hm_corr=float(data_json[baryon]['Hm_corr']['cntrl'])
        # print(Hm_corr)

        # Collect all y-data points for setting dynamic y-limits
        displacements = np.zeros(len(data["a_2"]))
        unique_a2 = np.unique(data["a_2"])

        for value in unique_a2:
            indices = np.where(data["a_2"] == value)[0]
            n = len(indices)
            offsets = np.linspace(-0.00010 * (n - 1), 0.00010 * (n - 1), n)
            displacements[indices] = offsets

        # Plot each baryon with slight displacement in x-axis to avoid overlap
        for j in range(len(data["combined_data_err"])):
            # print(baryon,Hm_corr)
            ax.errorbar(
                # data["a_2"][j] + displacements[j]+0.00005*i,  # Apply displacement
                data["a_2"][j] +0.0002*baryon_0_disp[i] + displacements[j],  # Apply displacement
                data["combined_data_Ds_phy"][j]+Hm_corr,
                yerr=data["combined_data_err"][j],
                color=colors[i % 8],  # Automatically assign color
                label=plot_configurations[baryon]["label_y"]
                if j == 0
                else "",  # Use label_y as the label, but only for the first point
                marker=markers[i % len(markers)],  # Cycle through the marker list
                **errorbar_kwargs,
            )

        ax.errorbar(
            0+0.0002*baryon_0_disp[i],  # Apply displacement
            Hm_cntrl,
            yerr=Hm_err,
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=markers[i % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs_modified,
        )

        # Plot fit line and shaded error region
        # ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="-")
        ax.plot(
            data["fit_a_2"], data["fit_fcn"]+Hm_corr, color=colors[i % 8], linestyle="-"
        )
        total_err = data["fit_fcn_err"]
        ax.errorbar(
            0+0.0002*baryon_0_disp[i],  # Apply displacement
            # 0+0.00008*i,  # Apply displacement
            data["fit_fcn"][0] +Hm_corr,
            yerr=total_err[0],
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=None,  # Cycle through the marker list
            # marker=markers[i % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs_modified,
        )
        ax.fill_between(
            data["fit_a_2"],
            # data["fit_fcn"] - data["fit_fcn_err"],
            # data["fit_fcn"] + data["fit_fcn_err"],
            data["fit_fcn"] - total_err+Hm_corr,
            data["fit_fcn"] + total_err+Hm_corr,
            alpha=0.2,
            color="grey",
            edgecolor="none"
            # alpha=0.3,color=f"{colors[i % 5]}"  , edgecolor="none"
        )

    ax.text(0.05, 0.97, chr(ord('A')+n_plt), transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.set_ylim([ylim_strt[n_plt], ylim_end[n_plt]])  # Dynamically set y-limits
    ax.set_xlim([-0.0003, 0.013])
    ax.legend(loc="best", ncol=4, fontsize="small")
    ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    if n_plt == 0:
        # ax.set_ylabel(r"$\langle H_m \rangle_H$ (GeV)")
        ax.set_ylabel(r"$\sigma_H$ (GeV)")

# Loop over groups and subplots
gs_right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 2]
)  # Create 2x1 grid within the 3rd subplot
for n_plt, group in enumerate(groups2):
    ax = fig.add_subplot(gs_right[n_plt, 0])
    baryon_0_disp=charm_baryon_disp2
    n_plt += 2
    for i, baryon in enumerate(group):
        # Load data
        if baryon not in light_baryons:
            data = np.load(
                f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy", allow_pickle=True
            ).item()
            # data2 = np.load(
            #     f"./npy/{baryon}_Hm_data_addCa_False.npy", allow_pickle=True
            # ).item()
            # chi2 = max(data["fit_ch2"], 1)
        else:
            data = np.load(
                f"{pydata_path}/npy/{baryon}_Hm_addCa_True.npy",
                allow_pickle=True,
            ).item()
        if baryon in light_baryons:
            Hm_cntrl=float(data_json[baryon]['Hm']['cntrl'])
            Hm_err=(float(data_json[baryon]['Hm']['total stat'])**2+float(data_json[baryon]['Hm']['alttc_sys'])**2+float(data_json[baryon]['Hm']['fit ansatz'])**2+float(data_json[baryon]['Hm']['correction'])**2)**0.5
        else:
            Hm_cntrl=float(data_json[baryon]['Hm']['cntrl'])
            Hm_err=(float(data_json[baryon]['Hm']['total stat'])**2+float(data_json[baryon]['Hm']['alttc_sys'])**2+float(data_json[baryon]['Hm']['fit ansatz'])**2+float(data_json[baryon]['Hm']['correction'])**2+float(data_json[baryon]['Hm']['D_S'])**2)**0.5
        Hm_corr=float(data_json[baryon]['Hm_corr']['cntrl'])
            # data2 = np.load(
            #     f"../light_hadron_fit_like_proton/npy/{baryon}_Hm_addCa_False.npy",
            #     allow_pickle=True,
            # ).item()
            # data = np.load(f"../light_hadron_fit_like_proton/npy/{baryon}_mass_extra_plot_data.npy", allow_pickle=True).item()
            # chi2 = 1

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
                # data["a_2"][j] + displacements[j]+0.0001*i,  # Apply displacement
                data["a_2"][j] +0.0002*baryon_0_disp[i] + displacements[j],  # Apply displacement
                data["combined_data_Ds_phy"][j]+Hm_corr,
                yerr=data["combined_data_err"][j],
                color=colors[i % 8],  # Automatically assign color
                label=plot_configurations[baryon]["label_y"]
                if j == 0
                else "",  # Use label_y as the label, but only for the first point
                marker=markers[i % len(markers)],  # Cycle through the marker list
                **errorbar_kwargs,
            )
        ax.errorbar(
            # 0+0.0001*i,  # Apply displacement
            0+0.0002*baryon_0_disp[i],  # Apply displacement
            Hm_cntrl,
            yerr=Hm_err,
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=markers[i % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs_modified,
        )
        ax.errorbar(
            # 0+0.0001*i,  # Apply displacement
            0+0.0002*baryon_0_disp[i],  # Apply displacement
            data["fit_fcn"][0]+Hm_corr,
            yerr=data["fit_fcn_err"][0],
            color=colors[i % 8],  # Automatically assign color
            label='',
            marker=None,  # Cycle through the marker list
            **errorbar_kwargs,
        )

        # Plot fit line and shaded error region
        # ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="-")
        ax.plot(
            data["fit_a_2"], data["fit_fcn"]+Hm_corr, color=colors[i % 8], linestyle="-"
        )
        total_err =data["fit_fcn_err"]
        # total_err = (
        #     data["fit_fcn_err"] ** 2 + (data["fit_fcn"] - data2["fit_fcn"]) ** 2
        # ) ** (1 / 2) * chi2 ** (1 / 2)
        ax.fill_between(
            data["fit_a_2"],
            # data["fit_fcn"] - data["fit_fcn_err"],
            # data["fit_fcn"] + data["fit_fcn_err"],
            data["fit_fcn"] - total_err+Hm_corr,
            data["fit_fcn"] + total_err+Hm_corr,
            alpha=0.2,
            color="grey",
            edgecolor="none"
            # alpha=0.3,color=f"{colors[i % 5]}"  , edgecolor="none"
        )

    # Set individual limits and labels based on data
    # y_min = min(all_data_points) - 0.1  # Add some padding
    # y_max = max(all_data_points) + 0.15  # Add some padding
    # ax.set_ylim([y_min, y_max])  # Dynamically set y-limits
    ax.text(0.05, 0.95, chr(ord('F')-n_plt), transform=ax.transAxes, fontsize=18, fontweight=1000, ha='left', va='top')
    ax.set_ylim([ylim_strt[n_plt], ylim_end[n_plt]])  # Dynamically set y-limits
    ax.set_xlim([-0.0003, 0.013])
    # ax.set_ylabel(r"$H_m^{\mathrm{tot}}$ (GeV)")
    # if n_plt<3:
    #     ax.set_xticklabels([])  # Remove x-axis labels for upper plots
    # ax.legend(loc="best", ncol=len(group), fontsize='small',prop={'size': 7})
    ax.legend(loc="best", ncol=4, fontsize="small")
    # ax.legend(loc="best", ncol=4, fontsize='small',prop={'size': 6})
    ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    if n_plt == 2:
        ax.set_xticklabels([])  # Remove x-axis labels for upper plots

# Add x-axis label only for the bottom plot
# axs[3].set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
# axs[0].set_ylabel(r"$H_m^{\mathrm{tot}}$ (GeV)")

# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.02)
plt.savefig(f"{lqcd_data_path}/final_results/Hm_tot_alttc.pdf")
