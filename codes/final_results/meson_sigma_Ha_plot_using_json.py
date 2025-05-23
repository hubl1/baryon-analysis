import numpy as np
import matplotlib.pyplot as plt
from params.plt_labels import plot_configurations
import json
import sys

lqcd_data_path=sys.argv[1]

plt.style.use("utils/science.mplstyle")
colors = ['grey', 'r', 'b', 'g']
markers = ['^', 'v', '<', '>']
# spin = ['1/2', '3/2']
spin = ['0', '1']
orbital_momentum= ['S', 'P']
plt.rcParams['figure.dpi'] = "300"
plt.rcParams['legend.fontsize'] = 7.5

def plot_combined_baryon_data():
    # Load JSON data
    json_file = f"{lqcd_data_path}/final_results/data_error_budget.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    light_baryons = ['PROTON', 'LAMBDA', 'SIGMA', 'XI', 'DELTA', 'SIGMA_STAR', 'XI_STAR', 'OMEGA']
    baryons = [
        "LAMBDA_C", "SIGMA_C", "XI_C", "XI_C_PRIME", "OMEGA_C", "SIGMA_STAR_C", "XI_STAR_C", "OMEGA_STAR_C",
        # "LAMBDA_C", "SIGMA_C", "XI_C",  "OMEGA_C", "SIGMA_STAR_C", "XI_STAR_C", "OMEGA_STAR_C",
        "XI_CC", "OMEGA_CC", "XI_STAR_CC", "OMEGA_STAR_CC", "OMEGA_CCC"
    ]
    charmed_mesons = [
        "D",
        "D_S",
        "D_STAR",
        "DS_STAR",
        "ETA_C",
        "JPSI",
        "CHI_C0",
        "CHI_C1",
        "H_C",
    ]
    errorbar_kwargs = {
        "linewidth": 0.8,
        "elinewidth": 0.8,
        "capthick": 1,
        "capsize": 2,
        "mew": 0.8,
        "linestyle": "none",
        "fillstyle": "none",
        "markersize": 4.5,
    }
    errorbar_kwargs_modified = errorbar_kwargs.copy()
    errorbar_kwargs_modified.pop('fillstyle', None)

    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(8*0.85, 4.8*0.85))  # 2x2 grid of plots

    # Plot 1: Hml_data
    ax = axs[0, 0]
    # plot_baryons = light_baryons + baryons
    plot_baryons = charmed_mesons
    added_labels = set()

    for j, baryon in enumerate(plot_baryons):
        if baryon in light_baryons:
            Hml_cntrl=float(data[baryon]['Hml']['cntrl'])
            Hml_err=(float(data[baryon]['Hml']['total stat'])**2+float(data[baryon]['Hml']['alttc_sys'])**2+float(data[baryon]['Hml']['fit ansatz'])**2+float(data[baryon]['Hml']['correction'])**2)**0.5
        else:
            Hml_cntrl=float(data[baryon]['Hml']['cntrl'])
            Hml_err=(float(data[baryon]['Hml']['total stat'])**2+float(data[baryon]['Hml']['alttc_sys'])**2+float(data[baryon]['Hml']['fit ansatz'])**2+float(data[baryon]['Hml']['correction'])**2+float(data[baryon]['Hml']['D_S'])**2)**0.5

        # Define label text
        num_light_quarks = plot_configurations[baryon]["quark_content"].count('u') + plot_configurations[baryon]["quark_content"].count('d')
        # label_text = f'Baryon with {num_light_quarks} light quark(s)'
        label_text = f'{num_light_quarks} valance light quark cases'

        # Set label only if it hasn't been added yet
        label = label_text if label_text not in added_labels else None

        ax.errorbar(
            j,
            Hml_cntrl,
            yerr=Hml_err,
            **errorbar_kwargs,
            color=colors[num_light_quarks],
            marker=markers[num_light_quarks],
            label=label
        )

        if label:
            added_labels.add(label_text)

        # Add annotation for each baryon
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            # (j, data_Hm["fit_fcn"][0] - np.sqrt(data_Hm["fit_fcn_err"][0]**2 + (data_Hm["fit_fcn"][0] - data_Hm2["fit_fcn"][0])**2) * np.sqrt(Hm_chi2)),
            (j, Hml_cntrl+Hml_err+0.0025),
            textcoords="offset points",
            fontsize=8,
            xytext=(0, -9),
            ha="center",
        )

    ax.legend()
    ax.set_xticks([])
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(-0.002, 0.025)
    ax.set_ylabel(r'$\sigma_{\pi H}$ (GeV)')
    ax.text(-0.22, 1.00, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Plot 2: Hms_data
    ax = axs[0, 1]
    added_labels = set()
    for j, baryon in enumerate(plot_baryons):
        if baryon in light_baryons:
            Hms_cntrl=float(data[baryon]['Hms']['cntrl'])
            Hms_err=(float(data[baryon]['Hms']['total stat'])**2+float(data[baryon]['Hms']['alttc_sys'])**2+float(data[baryon]['Hms']['fit ansatz'])**2+float(data[baryon]['Hms']['correction'])**2)**0.5
        else:
            Hms_cntrl=float(data[baryon]['Hms']['cntrl'])
            Hms_err=(float(data[baryon]['Hms']['total stat'])**2+float(data[baryon]['Hms']['alttc_sys'])**2+float(data[baryon]['Hms']['fit ansatz'])**2+float(data[baryon]['Hms']['correction'])**2+float(data[baryon]['Hms']['D_S'])**2)**0.5

        num_strange_quarks = plot_configurations[baryon]["quark_content"].count('s')
        label_text = f'{num_strange_quarks} valance strange quark cases'

        label = label_text if label_text not in added_labels else None


        ax.errorbar(
            j,
            Hms_cntrl,
            # data_Hm["fit_fcn"][0]-d2s*Hmsv*metas_phys**2/ms_phys-d2c*Hmcv*metas_phys**2/mc_phys,
            yerr=Hms_err,
            **errorbar_kwargs,
            color=colors[num_strange_quarks],
            marker=markers[num_strange_quarks],
            label=label
        )
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            # (j, data_Hm["fit_fcn"][0] - np.sqrt(data_Hm["fit_fcn_err"][0]**2 + (data_Hm["fit_fcn"][0] - data_Hm2["fit_fcn"][0])**2) * np.sqrt(Hm_chi2)),
            (j, Hms_cntrl+Hms_err+0.025),
            textcoords="offset points",
            fontsize=8,
            xytext=(0, -9),
            ha="center",
        )

        if label:
            added_labels.add(label_text)


    # 获取 legend 句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 反转句柄和标签顺序
    handles = handles[::-1]
    labels = labels[::-1]

    ax.legend(handles, labels)
    ax.set_xticks([])
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(-0.02, 0.25)
    ax.set_ylabel(r'$\sigma_{sH}$ (GeV)')
    ax.text(-0.22, 1.00, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Plot 3: Hmc_data
    ax = axs[1, 0]
    added_labels = set()

    for j, baryon in enumerate(charmed_mesons):
        Hmc_cntrl=float(data[baryon]['Hmc']['cntrl'])
        Hmc_err=(float(data[baryon]['Hmc']['total stat'])**2+float(data[baryon]['Hmc']['alttc_sys'])**2+float(data[baryon]['Hmc']['fit ansatz'])**2+float(data[baryon]['Hmc']['D_S'])**2)**0.5

        num_charm_quarks = plot_configurations[baryon]["quark_content"].count('c')
        # label_text = f'Baryon with {num_charm_quarks} charm quark(s)'
        label_text = f'{num_charm_quarks} valance charm quark cases'

        label = label_text if label_text not in added_labels else None

        ax.errorbar(
            j,
            Hmc_cntrl,
            yerr=Hmc_err,
            **errorbar_kwargs,
            color=colors[num_charm_quarks],
            marker=markers[num_charm_quarks],
            label=label
        )
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            (j, Hmc_cntrl+Hmc_err+0.3),
            textcoords="offset points",
            fontsize=8,
            xytext=(0, -9),
            ha="center",
        )

        if label:
            added_labels.add(label_text)

    # 获取 legend 句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 反转句柄和标签顺序
    handles = handles[::-1]
    labels = labels[::-1]

    ax.legend(handles, labels)
    ax.set_xticks([])
    ax.set_ylim(0.5, 3.2)
    ax.set_ylabel(r'$\sigma_{cH}$ (GeV)')
    ax.text(-0.22, 1.00, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Plot 4: Ha and Hag data
    ax = axs[1, 1]
    added_labels = set()
    for j, baryon in enumerate(plot_baryons):
        if baryon in light_baryons:
            Ha_cntrl=float(data[baryon]['Ha']['cntrl'])
            Ha_err=(float(data[baryon]['Ha']['total stat'])**2+float(data[baryon]['Ha']['alttc_sys'])**2+float(data[baryon]['Ha']['fit ansatz'])**2+float(data[baryon]['Ha']['correction'])**2)**0.5
            Hag_cntrl=float(data[baryon]['Hag']['cntrl'])
            Hag_err=(float(data[baryon]['Hag']['total stat'])**2+float(data[baryon]['Hag']['alttc_sys'])**2+float(data[baryon]['Hag']['fit ansatz'])**2+float(data[baryon]['Hag']['correction'])**2)**0.5
        else:
            Ha_cntrl=float(data[baryon]['Ha']['cntrl'])
            Ha_err=(float(data[baryon]['Ha']['total stat'])**2+float(data[baryon]['Ha']['alttc_sys'])**2+float(data[baryon]['Ha']['fit ansatz'])**2+float(data[baryon]['Ha']['correction'])**2+float(data[baryon]['Ha']['D_S'])**2)**0.5
            Hag_cntrl=float(data[baryon]['Hag']['cntrl'])
            Hag_err=(float(data[baryon]['Hag']['total stat'])**2+float(data[baryon]['Hag']['alttc_sys'])**2+float(data[baryon]['Hag']['fit ansatz'])**2+float(data[baryon]['Hag']['correction'])**2+float(data[baryon]['Hag']['D_S'])**2)**0.5

        spin_type = spin[plot_configurations[baryon]["spin_parity"].count('1')]
        orbital_type = plot_configurations[baryon]["L"]
        # label_text_Ha = rf'$\langle H_a \rangle$ for spin {spin_type} mesons'
        label_text_Ha = rf'$\langle H_a \rangle$ for {orbital_type}-wave mesons'

        label_Ha = label_text_Ha if label_text_Ha not in added_labels else None

        ax.errorbar(
            j,
            # data_Ha["fit_fcn"][0],
            Ha_cntrl,
            yerr=Ha_err,
            # marker=markers[plot_configurations[baryon]["spin_parity"].count('1') + 1],
            # color=colors[plot_configurations[baryon]["spin_parity"].count('1') + 1],
            marker=markers[plot_configurations[baryon]["L"].count('P') + 1],
            color=colors[plot_configurations[baryon]["L"].count('P') + 1],
            **errorbar_kwargs,
            label=label_Ha,
        )

        if label_Ha:
            added_labels.add(label_text_Ha)

        label_text_Hag = rf'$\langle H_a^g \rangle$ for {orbital_type}-wave mesons'

        label_Hag = label_text_Hag if label_text_Hag not in added_labels else None

        ax.errorbar(
            j,
            Hag_cntrl,
            yerr=Hag_err,
            # marker=markers[plot_configurations[baryon]["spin_parity"].count('1') + 1],
            # color=colors[plot_configurations[baryon]["spin_parity"].count('1') + 1],
            **errorbar_kwargs_modified,
            marker=markers[plot_configurations[baryon]["L"].count('P') + 1],
            color=colors[plot_configurations[baryon]["L"].count('P') + 1],
            alpha=1,
            label=label_Hag,
        )

        if label_Hag:
            added_labels.add(label_text_Hag)

        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            (j, Ha_cntrl+Ha_err+0.03),
            textcoords="offset points",
            fontsize=8,
            xytext=(0, 4),
            ha="center",
        )

    ax.legend()
    ax.set_xticks([])
    ax.set_ylim(0.00, 2.4)
    ax.set_ylabel(r'$\langle H_a^{(g)} \rangle_H$ (GeV)')
    ax.text(-0.22, 1.00, 'D', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.02)
    plt.savefig(f'{lqcd_data_path}/final_results/Hm_4in1_meson.pdf')

# Code to save the combined plot to a file.
plot_combined_baryon_data()
