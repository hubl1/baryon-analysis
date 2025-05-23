import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from params.plt_labels import plot_configurations
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from utils.gv3 import gv2
import json
from matplotlib.ticker import MultipleLocator
import sys


lqcd_data_path=sys.argv[1]
json_file = f"{lqcd_data_path}/final_results/data_error_budget.json"
plt.style.use("utils/science.mplstyle")
# colors = [
#     "black",  # 红色
#     "red",  # 红色
#     "blue",  # 蓝色
#     "green",  # 绿色
#     "orange",  # 橙色
#     "purple",  # 紫色
#     "cyan",  # 青色
#     "magenta",  # 洋红色
#     "brown",  # 棕色 (代替黄色)
# ]
with open(json_file, "r") as f:
    data = json.load(f)
# colors = [
#     "red",      # 红色
#     "black",    # 黑色
#     "purple",   # 紫色
#     "blue",     # 蓝色
#     # "orange",   # 橙色
#     # "cyan",     # 青色
#     "green",    # 绿色
#     "magenta",  # 洋红色
#     "brown",    # 棕色 (代替黄色)
#     # "pink",     # 粉色
#     "teal",     # 蓝绿色
#     "navy",     # 深蓝色
#     "gold",     # 金色
#     "gray",     # 灰色
#     "violet",   # 紫罗兰色
#     "turquoise",# 绿松石色
#     "beige",    # 米色
#     "olive",    # 橄榄色
#     "salmon",   # 鲜肉色
#     "lime",     # 亮绿色
# ]
colors = [
    # "red",      # 红色
    "#D62728",      # 红色
    # "#bc2e3b",      # 红色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    # "steelblue",    # 黑色
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
    "#4b89ca", #蓝
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
    # "linewidth": 0.2,
    "elinewidth": 0.65,
    "mew": 0.65,
    "capthick": 1,
    "capsize": 2,
    "linestyle": "none",
    "fillstyle": "none",
    "markersize": 3,
}

# Define markers
markers = [ "<",">", "^","v",  "x", "s", "d", "o", "*", "p",  "h", "H", ".", "+","*"]

# Create a 3-row, 1-column plot layout for three groups
# fig, axs = plt.subplots(1, 3, figsize=(10.5, 4), dpi=140)
fig = plt.figure(figsize=(8*0.85, 3*0.85))
# fig = plt.figure()
gs = gridspec.GridSpec(
    1, 3, width_ratios=[1, 1, 0.50]
)  # Set width_ratios to adjust the width of the third plot

# Grouping baryons based on number of 'C' in their names
group_0 = [b for b in baryons + light_baryons if b.count("C") == 0]  # One 'C'
group_1 = [b for b in baryons + light_baryons if b.count("C") == 1]  # One 'C'
group_2 = [b for b in baryons + light_baryons if b.count("C") == 2]  # Two 'C's
group_3 = [b for b in baryons + light_baryons if b.count("C") == 3]  # Three 'C's

# ylim_strt = [0.85, 2.0, 4.55, 3.45]
# ylim_end = [1.85, 3.0, 5.0, 3.95]
ylim_strt = [0.75, 2.0, 4.55, 3.45]
ylim_end = [1.75, 3.0, 5.0, 3.95]
# groups = [group_3, group_2, group_1,group_0]
groups1 = [group_0, group_1]
groups2 = [group_3, group_2]
liteature = [
    "BMW_08",
    "Alexandrou_14",
    "RQCD_22",
    "Alexandrou_23",
    # "Alexandrou_17",
    # "Aoki_09",
    # "QCDSF_UKQCD_11",
    "Briceno_12",
    "Brown_14",
    # "Liu",
    # "Namekawa_13",
    # "Padm_14",
    # "Perez-Rubio_15",
    # "Durr_12",
    # "Li_22",
    "Dhindsa_24",
    "Mathur_19",
]
# labels_dict = {
#     "Alexandrou_14": "Alexandrou 14 ($a\\to 0$)",
#     "Alexandrou_17": "Alexandrou 17 ($a \\approx 0.09$ fm)",
#     "Alexandrou_23": "Alexandrou 23  ($a\\to 0$)",
#     "Briceno_12": "Brice$\\~\mathrm{n}$o 12 ($a\\to 0$)",
#     "Brown_14": "Brown 14 ($a\\to 0$)",
#     "Liu": "Liu 10 ($a \\approx 0.12$ fm)",
#     "Namekawa_13": "Namekawa 13 ($a \\approx 0.09$ fm)",
#     "Padm_14": "Padmanath 13 ($a \\approx 0.12$ fm)",
#     "Perez-Rubio_15": "P$\\'\mathrm{e}$rez-Rubio 15 ($a \\approx 0.075$ fm)",
#     "BMW_08": "BMWc 08 ($a\\to 0$)",
#     "Aoki_09": "PACS-CS 08 ($a \\approx 0.09$ fm)",
#     "QCDSF_UKQCD_11": "QCDSF UKQCD 11 ($a \\approx 0.075$ fm)",
#     "Durr_12": 'D$\\"\mathrm{u}$rr 12 ($a \\approx 0.07$ fm)',
#     "Li_22": 'Li 22 ($a \\approx 0.07$ fm)',
#     "Dhindsa_24": 'Dhindsa 24 ($a \\to 0$)',
# }
labels_dict = {
    "Alexandrou_14": "Alexandrou 14 ",
    "Alexandrou_17": "Alexandrou 17 ",
    "Alexandrou_23": "Alexandrou 23  ",
    # "Briceno_12": "Brice$\\~\mathrm{n}$o 12 ",
    # "Briceno_12": "Brice\~no 12",
    "Briceno_12": "Briceño 12",
    "Brown_14": "Brown 14 ",
    "Liu": "Liu 10 ($a \\approx 0.12$ fm)",
    "Namekawa_13": "Namekawa 13 ($a \\approx 0.09$ fm)",
    "Padm_14": "Padmanath 13 ($a \\approx 0.12$ fm)",
    "Perez-Rubio_15": "P$\\'\mathrm{e}$rez-Rubio 15 ($a \\approx 0.075$ fm)",
    "BMW_08": "BMWc 08 ",
    "Aoki_09": "PACS-CS 08 ($a \\approx 0.09$ fm)",
    "QCDSF_UKQCD_11": "QCDSF UKQCD 11 ($a \\approx 0.075$ fm)",
    "Durr_12": 'D$\\"\mathrm{u}$rr 12 ($a \\approx 0.07$ fm)',
    "Li_22": 'Li 22 ($a \\approx 0.07$ fm)',
    "Dhindsa_24": 'Dhindsa 24 ',
    "Mathur_19": 'Mathur 19',
    "RQCD_22": 'RQCD 22',
}
appeared=[ 0 for i in range(len(liteature)) ]

# Loop over groups and subplots
for n_plt, (group) in enumerate(groups1):
    # print(n_plt)
    ax = fig.add_subplot(gs[0, n_plt])

    if n_plt <1:
        # disps = np.array([ -1,0,1,2]) * 0.25
        disps = np.array([ -1,0,1,2,3]) * 0.22
        disp_hubl =2*0.22
    else:
        disps = (np.array([-2,-1,-1,-0,1,2])) * 0.17
        disp_hubl =2*0.17
    for i, baryon in enumerate(group):
        # Load data
        mass_cntrl=float(data[baryon]['mass']['cntrl'])
        mass_err=(float(data[baryon]['mass']['total stat'])**2+float(data[baryon]['mass']['alttc_sys'])**2+float(data[baryon]['mass']['fit ansatz'])**2)**0.5

        x_positions = range(len(group))  # Numerical positions for each baryon
        x_base = x_positions[i]  # Base x position for this baryon


        ax.errorbar(
            plot_configurations[baryon]["label_y"],
            mass_cntrl,  # Apply displacement
            yerr=mass_err,
            color=f"{colors[0 % len(colors)]}",  # Automatically assign color
            alpha=0,
            label= None,
            marker=markers[0 % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs,
        )
        # Use fill_between to create the horizontal bar
        # Define bounds for the horizontal bar
        x_low = x_base - disp_hubl - mass_err
        x_high = x_base - disp_hubl + mass_err
        y = mass_cntrl
        ax.fill_betweenx(
            [y - mass_err, y + mass_err],  # Vertical range (y-axis)
            x_base-0.5,  # Lower bound for x
            x_base+0.5,  # Upper bound for x
            color=f"{colors[0 % len(colors)]}",  # Use the same color as the errorbar
            alpha=0.2,  # Transparency
            label=None,  # No additional label
            edgecolor=None
        )
        ax.errorbar(
            x_base-disp_hubl,
            mass_cntrl,  # Apply displacement
            yerr=mass_err,
            color=f"{colors[0 % len(colors)]}",  # Automatically assign color
            label="This work" if baryon == "PROTON" else None,
            marker=markers[0 % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs,
        )

        # liteature=['Alexandrou_14']
        for j, lit in enumerate(liteature):
            with open(f"final_results/literature/{lit}.json", "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if baryon in json_data:
                if baryon == 'LAMBDA' or (baryon in baryons and appeared[j]==0):
                    appeared[j]=1
                # Get the x-position based on the label

                # Add displacement
                disp = disps[j]  # Center displacements
                x_position = x_base + disp
                baryon_gv = gv2(json_data[baryon]["mass"])

                ax.errorbar(
                    x_position,
                    float(baryon_gv.mean),  # Apply displacement
                    yerr=float(baryon_gv.sdev),
                    color=f"{colors[(j+1) % len(colors)]}",  # Automatically assign color
                    label=labels_dict[lit] if appeared[j]==1 else None,
                    marker=markers[
                        (j + 1) % len(markers)
                    ],  # Cycle through the marker list
                    **errorbar_kwargs,
                )
                appeared[j]=2

        # 绘制实验方框，使用矩形
        if "width" in plot_configurations[baryon].keys():
            width = plot_configurations[baryon]["width"] / 1000
            mass = float(plot_configurations[baryon]["mass"]) / 1000
            if width > -1:
                # ax.scatter(j, mass )
                # plt.plot(j, mass, 'k',  marker='_', markersize=6.5,markeredgewidth=0.15,linewidth=0.2,label='Experiment' if baryon == 'PROTON' else None)
                plt.plot(
                    i,
                    mass,
                    "k",
                    marker="_",
                    markersize=0,
                    # linewidth=0.4,
                    linewidth=0.5,
                    label="Experiment" if baryon == "PROTON" else None,
                )
                rect = Rectangle(
                    (i - 0.35, mass - width / 2),
                    0.7,
                    width,
                    edgecolor="black",
                    facecolor=to_rgba("black", 0.3),
                    linewidth=0.5,
                    label="Width" if baryon == "PROTON" else None,
                )
                rect2 = Rectangle(
                    (i - 0.35, mass),
                    0.7,
                    0,
                    edgecolor="black",
                    facecolor="none",
                    linewidth=0.5,
                    label=None,
                )
                ax.add_patch(rect)
                ax.add_patch(rect2)

    ax.set_ylim([ylim_strt[n_plt], ylim_end[n_plt]])  # Dynamically set y-limits
    # Set x-ticks and labels
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels([plot_configurations[baryon]["label_y"] for baryon in group])
    # ax.set_xticks([])
    # ax.set_xlim([-0.0003, 0.013])
    # legend=ax.legend(loc="upper left", ncol=1, fontsize="x-small")
    legend=ax.legend(loc="lower right",handletextpad=0.3, ncol=2, fontsize="x-small")
    # legend=ax.legend(loc="upper left", ncol=1, fontsize=6)
    legend.set_zorder(0)
    ax.text(0.05, 0.95, chr(ord('A')+n_plt), transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
    # ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    if n_plt == 0:
        ax.set_ylabel(r"$m_H$ (GeV)")
    # ax.xminorticks_off()
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))



# Loop over groups and subplots
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2])  # Create 2x1 grid within the 3rd subplot

for n_plt,group in enumerate( groups2):
    print(group, n_plt)
    ax = fig.add_subplot(gs_right[n_plt,0])
    n_plt+=2
    # disps = np.array([-2,-1,0,2,4,5]) * 0.1
    print(n_plt)
    # if n_plt==3:
    #     disps = np.array([-2,-1,0,0,1,2,3]) * 0.165
    #     disp_hubl =0.165*3
    # else:
    if n_plt == 2:
        disps = (np.array([-2,-1,0,0,1,2,3,4])-0.5) * 0.19
        disp_hubl =0.19*2.5
    if n_plt == 3:
        disps = np.array([-3,-2,-2,-1,0,1,1,2,4,5]) * 0.16
        disp_hubl =0.16*3
    
    for i, baryon in enumerate(group):
        # Load data
        mass_cntrl=float(data[baryon]['mass']['cntrl'])
        mass_err=(float(data[baryon]['mass']['total stat'])**2+float(data[baryon]['mass']['alttc_sys'])**2+float(data[baryon]['mass']['fit ansatz'])**2)**0.5

        x_positions = range(len(group))  # Numerical positions for each baryon
        x_base = x_positions[i]  # Base x position for this baryon

        ax.errorbar(
            plot_configurations[baryon]["label_y"],
            mass_cntrl,  # Apply displacement
            yerr=mass_err,
            color=f"{colors[0 % len(colors)]}",  # Automatically assign color
            alpha=0,
            label= None,
            marker=markers[0 % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs,
        )
        # Use fill_between to create the horizontal bar
        # Define bounds for the horizontal bar
        x_low = x_base - disp_hubl - mass_err
        x_high = x_base - disp_hubl + mass_err
        y = mass_cntrl
        ax.fill_betweenx(
            [y - mass_err, y + mass_err],  # Vertical range (y-axis)
            x_base-0.5,  # Lower bound for x
            x_base+0.5,  # Upper bound for x
            color=f"{colors[0 % len(colors)]}",  # Use the same color as the errorbar
            alpha=0.2,  # Transparency
            label=None,  # No additional label
            edgecolor=None
        )
        ax.errorbar(
            x_base-disp_hubl,
            mass_cntrl,  # Apply displacement
            yerr=mass_err,
            color=f"{colors[0 % len(colors)]}",  # Automatically assign color
            label="This work" if baryon == "PROTON" else None,
            marker=markers[0 % len(markers)],  # Cycle through the marker list
            **errorbar_kwargs,
        )
        # ax.errorbar(
        #     plot_configurations[baryon]["label_y"],
        #     data["fit_fcn"][0],  # Apply displacement
        #     yerr=total_err[0],
        #     color=f"{colors[0 % len(colors)]}",  # Automatically assign color
        #     label="CLQCD" if baryon == "PROTON" else None,
        #     marker=markers[0 % len(markers)],  # Cycle through the marker list
        #     **errorbar_kwargs,
        # )

        # liteature=['Alexandrou_14']

        for j, lit in enumerate(liteature):
            with open(f"final_results/literature/{lit}.json", "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if baryon in json_data:
                # Get the x-position based on the label

                # Add displacement
                disp = disps[j]  # Center displacements
                x_position = x_base + disp
                baryon_gv = gv2(json_data[baryon]["mass"])

                ax.errorbar(
                    x_position,
                    float(baryon_gv.mean),  # Apply displacement
                    yerr=float(baryon_gv.sdev),
                    color=f"{colors[(j+1) % len(colors)]}",  # Automatically assign color
                    label=labels_dict[lit] if appeared[j]==0 else None,
                    marker=markers[
                        (j + 1) % len(markers)
                    ],  # Cycle through the marker list
                    **errorbar_kwargs,
                )
                appeared[j]=1

        # 绘制实验方框，使用矩形
        if "width" in plot_configurations[baryon].keys():
            width = plot_configurations[baryon]["width"] / 1000
            mass = float(plot_configurations[baryon]["mass"]) / 1000
            if width > -1:
                # ax.scatter(j, mass )
                # plt.plot(j, mass, 'k',  marker='_', markersize=6.5,markeredgewidth=0.15,linewidth=0.2,label='Experiment' if baryon == 'PROTON' else None)
                plt.plot(
                    i,
                    mass,
                    "k",
                    marker="_",
                    markersize=0,
                    linewidth=0.5,
                    label="Experiment" if baryon == "PROTON" else None,
                )
                rect = Rectangle(
                    (i - 0.35, mass - width / 2),
                    0.7,
                    width,
                    edgecolor="black",
                    facecolor=to_rgba("grey", 0.5),
                    linewidth=0.5,
                    label="Width" if baryon == "PROTON" else None,
                )
                rect2 = Rectangle(
                    (i - 0.35, mass),
                    0.7,
                    0,
                    edgecolor="black",
                    facecolor="none",
                    linewidth=0.4,
                    label=None,
                )
                ax.add_patch(rect)
                ax.add_patch(rect2)



    ax.set_ylim([ylim_strt[n_plt],ylim_end[n_plt] ])  # Dynamically set y-limits
    legend=ax.legend(loc="lower right",handletextpad=0.3, ncol=1, fontsize='x-small')
    legend.set_zorder(0)
    ax.text(0.05, 0.95, chr(ord('F')-n_plt), transform=ax.transAxes, fontsize=14, fontweight=1000, ha='left', va='top')
    # ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$")
    # if n_plt==2:
    #     ax.set_xticklabels([])  # Remove x-axis labels for upper plots
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))


# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(wspace=0.16, hspace=0.195)
plt.savefig(f"{lqcd_data_path}/final_results/continuum_mass_literature.pdf")
