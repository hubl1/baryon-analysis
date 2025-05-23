import numpy as np
import matplotlib.pyplot as plt
from params.plt_labels import plot_configurations
import gvar as gv 
import json
from matplotlib.ticker import MultipleLocator
import sys

lqcd_data_path=sys.argv[1]

plt.style.use("utils/science.mplstyle")
colors = ['grey', 'r', 'b', 'g']
markers = ['^', 'v', '<', '>']
spin = ['1/2', '3/2']
plt.rcParams['figure.dpi'] = "300"
quark_mass_light=3.39/1000
quark_mass_strange=95.6/1000
quark_mass_charm=1.095
gammam=0.295
plot_tag=['A','B','C']

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
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.3))  # 2x2 grid of plots

    # Plot 1: Hml_data
    ax = axs[0]
    plot_baryons = light_baryons + baryons
    added_labels = set()

    k=0
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

        if num_light_quarks >0:
            Hml_cntrl/=num_light_quarks*quark_mass_light/(1+gammam)
            Hml_err/=num_light_quarks*quark_mass_light/(1+gammam)
            ax.errorbar(
                k,
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
                (k, Hml_cntrl-Hml_err),
                textcoords="offset points",
                fontsize=8,
                xytext=(0, -9),
                ha="center",
            )
            k+=1

    ax.legend()
    ax.set_xticks([])
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(0, 14)
    ax.set_ylabel(r'$(1+\gamma_m)\sigma_{\pi H} \, / \, (n_l\, m_l)$',fontsize=8)
    ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Plot 2: Hms_data
    ax = axs[1]
    added_labels = set()
    k=0
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


        if num_strange_quarks >0:
            Hms_cntrl/=num_strange_quarks*quark_mass_strange/(1+gammam)
            Hms_err/=num_strange_quarks*quark_mass_strange/(1+gammam)
            ax.errorbar(
                k,
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
                (j, Hms_cntrl-Hms_err),
                textcoords="offset points",
                fontsize=8,
                xytext=(0, -9),
                ha="center",
            )

        if label:
            added_labels.add(label_text)
        
        k+=1



    # 获取 legend 句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 反转句柄和标签顺序
    handles = handles[::-1]
    labels = labels[::-1]

    ax.legend(handles, labels)
    ax.set_xticks([])
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(0, 6)
    ax.set_ylabel(r'$(1+\gamma_m)\sigma_{s H} \, / \, (n_s\, m_s)$',fontsize=8)
    ax.text(0.05, 0.95, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Plot 3: Hmc_data
    ax = axs[2]
    added_labels = set()

    for j, baryon in enumerate(baryons):
        Hmc_cntrl=float(data[baryon]['Hmc']['cntrl'])
        Hmc_err=(float(data[baryon]['Hmc']['total stat'])**2+float(data[baryon]['Hmc']['alttc_sys'])**2+float(data[baryon]['Hmc']['fit ansatz'])**2+float(data[baryon]['Hmc']['D_S'])**2)**0.5

        num_charm_quarks = plot_configurations[baryon]["quark_content"].count('c')
        # label_text = f'Baryon with {num_charm_quarks} charm quark(s)'
        label_text = f'{num_charm_quarks} valance charm quark cases'

        label = label_text if label_text not in added_labels else None

        Hmc_cntrl/=num_charm_quarks*quark_mass_charm/(1+gammam)
        Hmc_err/=num_charm_quarks*quark_mass_charm/(1+gammam)
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
            (j, Hmc_cntrl-Hmc_err),
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
    ax.set_ylim(1.1, 1.5)
    ax.set_ylabel(r'$(1+\gamma_m)\sigma_{c H} \, / \, (n_c\, m_c)$', fontsize=8)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.text(0.05, 0.95, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')


    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.20, hspace=0.02)
    plt.savefig(f'{lqcd_data_path}/final_results/ratio_3in1.pdf')

# Code to save the combined plot to a file.
plot_combined_baryon_data()
