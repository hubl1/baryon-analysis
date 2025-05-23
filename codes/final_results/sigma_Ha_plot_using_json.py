import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
from params.plt_labels import plot_configurations
import gvar as gv
import json
import sys
lqcd_data_path=sys.argv[1]

plt.style.use('utils/science.mplstyle')
# mpl.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "text.latex.preamble": r"""\usepackage{helvet}
# \renewcommand{\familydefault}{\sfdefault}
# \usepackage[eulergreek]{sansmath}
# \sansmath
# """,
# })
# mpl.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
# "text.usetex": True,
# "text.latex.preamble": r"""
# \usepackage{helvet}
# \renewcommand{\familydefault}{\sfdefault}
# \usepackage[helvet]{mathastext}
# """
# })
# mpl.rcParams.update({
#     "text.usetex": True,              # 用 LaTeX 排版
#     "font.family": "sans-serif",      # 全局无衬线
#     "font.sans-serif": "Helvetica, Arial, DejaVu Sans",
#     "text.latex.preamble": r"""
# \usepackage{helvet}          % Helvetica 正文字体
# \renewcommand{\familydefault}{\sfdefault}
# \usepackage[helvet]{mathastext}  % 让数学也继承 Helvetica，并自动提供希腊字母
# """,
# })
# colors = ['grey', 'r', 'b', 'g']
# colors = ['grey', 'r', 'orange', 'b']
# colors2 = ['grey', 'r', 'orange', 'b']
colors = ['grey', '#D62728', '#E69F00', '#4b89ca']
# colors2 = ['grey', 'r', 'orange', 'b']
mfc=(0, 0, 0, 0.3)
markers = ['^', 'v', '<', '>']
spin = ['1/2', '3/2']
plt.rcParams['figure.dpi'] = "300"
Hag_black=['#2B2B2B', 'black']
Ha_green=['#2CA02C', '#1B7837']
alphas=[0.7,1]
quark_mass_light=3.39/1000
quark_mass_strange=95.6/1000
quark_mass_charm=1.095

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
    # errorbar_kwargs = {
    #     "linewidth": 0.8,
    #     "elinewidth": 0.8,
    #     "capthick": 1,
    #     "capsize": 2,
    #     "mew": 0.8,
    #     "linestyle": "none",
    #     "fillstyle": "none",
    #     "markersize": 4.5,
    # }
    errorbar_kwargs = {
        # "linewidth": 0.5,
        "elinewidth": 0.7*1.1,
        "mew": 0.7,
        "capthick": 1,
        "capsize": 2.3,
        "linestyle": "none",
        "fillstyle": "none",
        "markersize": 3.0,
    }
    errorbar_kwargs_modified = errorbar_kwargs.copy()
    errorbar_kwargs_modified.pop('fillstyle', None)
    kwargs=[errorbar_kwargs, errorbar_kwargs_modified]

    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of plots
    fig, axs = plt.subplots(4, 1, figsize=(8.4*.925, 5.9*0.90))  # 2x2 grid of plots

    # Plot 1: Hml_data
    ax = axs[ 0]
    plot_baryons = light_baryons + baryons
    added_labels = set()

    for n, c in zip([1, 2, 3], ["red", "orange", "blue"]):
        # 计算 band 边界
        y_low  = 4.55/1.295 * n * quark_mass_light
        y_high = 6.55/1.295 * n * quark_mass_light
        # y_cent = 1.21/1.295 * n * quark_mass_strange

        # 半透明水平带：整条横跨 & 无边框
        ax.axhspan(y_low, y_high, xmin=0, xmax=1,
                facecolor=c, alpha=0.08, edgecolor="none", zorder=0)

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
        label_text = f'{num_light_quarks} valance light quark cases'+ r"$\:\:\,\,\,\,\,$"

        # Set label only if it hasn't been added yet
        label = label_text if label_text not in added_labels else None

        ax.errorbar(
            j,
            Hml_cntrl,
            yerr=Hml_err,
            **errorbar_kwargs_modified,
            color=colors[num_light_quarks],
            marker=markers[num_light_quarks],
            label=label,
        )

        if label:
            added_labels.add(label_text)

        # Add annotation for each baryon
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            # (j, data_Hm["fit_fcn"][0] - np.sqrt(data_Hm["fit_fcn_err"][0]**2 + (data_Hm["fit_fcn"][0] - data_Hm2["fit_fcn"][0])**2) * np.sqrt(Hm_chi2)),
            (j, Hml_cntrl+Hml_err),
            textcoords="offset points",
            fontsize=9,
            xytext=(0, 4),
            ha="center",
        )

    ax.text(-0.08, 1, "A", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    # ax.legend(ncol=4,loc='upper left')
    ax.legend(loc='upper left',
        ncol=4,  # 或适当调整，比如 ncol=2 看分布是否更紧凑
        columnspacing=1.8,  # 减小列间距（默认 2.0）
        handletextpad=0.1,  # 图标和文字间距（默认 0.8）
        borderaxespad=0.1,  # legend 和坐标轴间距（默认 0.5）
        handlelength=1.0,   # 图例图标线段长度
        # bbox_to_anchor=(0.5, 1.05),  # 居中微调 legend 位置
        frameon=False,      # 可选，去掉框线以节省空间
        # fontsize=9          # 控制字体大小
    )
    ax.set_xticks([])
    # ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(-0.00, 0.075)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_ylabel(r'$\sigma_{\pi H}$ (GeV)')

    # Plot 2: Hms_data
    ax = axs[ 1]
    added_labels = set()
    for n, c in zip([1, 2, 3], ["red", "orange", "blue"]):
        # 计算 band 边界
        y_low  = 2.1/1.295 * n * quark_mass_strange
        y_high = 2.8/1.295 * n * quark_mass_strange
        # y_cent = 1.21/1.295 * n * quark_mass_strange

        # 半透明水平带：整条横跨 & 无边框
        ax.axhspan(y_low, y_high, xmin=0, xmax=1,
                facecolor=c, alpha=0.08, edgecolor="none", zorder=0)
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
            **errorbar_kwargs_modified,
            color=colors[num_strange_quarks],
            marker=markers[num_strange_quarks],
            label=label
        )
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            # (j, data_Hm["fit_fcn"][0] - np.sqrt(data_Hm["fit_fcn_err"][0]**2 + (data_Hm["fit_fcn"][0] - data_Hm2["fit_fcn"][0])**2) * np.sqrt(Hm_chi2)),
            (j, Hms_cntrl+Hms_err),
            textcoords="offset points",
            fontsize=9,
            xytext=(0, 4),
            ha="center",
        )

        if label:
            added_labels.add(label_text)


    # 获取 legend 句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 反转句柄和标签顺序
    handles = handles[::-1]
    labels = labels[::-1]

    ax.text(-0.08, 1, "B", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    # ax.legend(handles, labels,ncol=4,loc='upper left')
    ax.legend(handles, labels,
        loc='upper left',
        ncol=4,  # 或适当调整，比如 ncol=2 看分布是否更紧凑
        columnspacing=0.3,  # 减小列间距（默认 2.0）
        handletextpad=0.1,  # 图标和文字间距（默认 0.8）
        borderaxespad=0.1,  # legend 和坐标轴间距（默认 0.5）
        handlelength=1.0,   # 图例图标线段长度
        # bbox_to_anchor=(0.5, 1.05),  # 居中微调 legend 位置
        frameon=False,      # 可选，去掉框线以节省空间
        # fontsize=9          # 控制字体大小
    )
    ax.set_xticks([])
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.set_ylim(-0.05, 0.8)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_ylabel(r'$\sigma_{sH}$ (GeV)')

    # Plot 3: Hmc_data
    ax = axs[2]
    added_labels = set()

    # ax.axhline(y=1.21/1.29*quark_mass_charm, color="red", linestyle="--",alpha=0.5)
    # ax.axhline(y=1.21/1.29*2*quark_mass_charm, color="orange", linestyle="--",alpha=0.5)
    # ax.axhline(y=1.21/1.29*3*quark_mass_charm, color="blue", linestyle="--",alpha=0.5)
    # xmin, xmax = ax.get_xlim()                # 当前 x 轴范围
    # x_band = [xmin, xmax]                     # 两端 x 坐标

    for n, c in zip([1, 2, 3], ["red", "orange", "blue"]):
        # 计算 band 边界
        y_low  = 1.16/1.295 * n * quark_mass_charm
        y_high = 1.30/1.295 * n * quark_mass_charm
        y_cent = 1.21/1.295 * n * quark_mass_charm

        # 半透明水平带：整条横跨 & 无边框
        ax.axhspan(y_low, y_high, xmin=0, xmax=1,
                facecolor=c, alpha=0.08, edgecolor="none", zorder=0)

        # # 中心虚线（可删）
        # ax.axhline(y=y_cent, color=c, linestyle="--", alpha=0.6, zorder=1)
    for j, baryon in enumerate(baryons):
        Hmc_cntrl=float(data[baryon]['Hmc']['cntrl'])
        Hmc_err=(float(data[baryon]['Hmc']['total stat'])**2+float(data[baryon]['Hmc']['alttc_sys'])**2+float(data[baryon]['Hmc']['fit ansatz'])**2+float(data[baryon]['Hmc']['D_S'])**2)**0.5

        num_charm_quarks = plot_configurations[baryon]["quark_content"].count('c')
        # label_text = f'Baryon with {num_charm_quarks} charm quark(s)'
        label_text = f'{num_charm_quarks} valance charm quark cases'+ r"$\,\,\,$"

        label = label_text if label_text not in added_labels else None

        ax.errorbar(
            j+8,
            Hmc_cntrl,
            yerr=Hmc_err,
            **errorbar_kwargs_modified,
            color=colors[num_charm_quarks],
            marker=markers[num_charm_quarks],
            label=label
        )
        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            (j+8, Hmc_cntrl+Hmc_err),
            textcoords="offset points",
            fontsize=9,
            xytext=(0, 6),
            ha="center",
        )

        if label:
            added_labels.add(label_text)

    # 获取 legend 句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 反转句柄和标签顺序
    handles = handles[::-1]
    labels = labels[::-1]

    # ax.legend(handles, labels,ncol=4,loc='upper left')
    ax.text(-0.08, 1, "C", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.legend(handles, labels,
        loc='upper left',
        ncol=4,  # 或适当调整，比如 ncol=2 看分布是否更紧凑
        columnspacing=0.85,  # 减小列间距（默认 2.0）
        handletextpad=0.1,  # 图标和文字间距（默认 0.8）
        borderaxespad=0.1,  # legend 和坐标轴间距（默认 0.5）
        handlelength=1.0,   # 图例图标线段长度
        # bbox_to_anchor=(0.5, 1.05),  # 居中微调 legend 位置
        frameon=False,      # 可选，去掉框线以节省空间
        # fontsize=9          # 控制字体大小
    )
    ax.set_xticks([])
    # ax.set_ylim(0.5, 3.2)
    ax.set_xlim(-1, 21)
    ax.set_ylim(0.85, 4.0)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_ylabel(r'$\sigma^{\mathrm{val}}_{cH}$ (GeV)')

    # Plot 4: Ha and Hag data
    ax = axs[ 3]
    added_labels = set()
    ax.axhspan(0.8, 1.2, xmin=0, xmax=1,
            facecolor='grey', alpha=0.2, edgecolor="none", zorder=0)
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

        spin_type = spin[plot_configurations[baryon]["spin_parity"].count('3')]
        label_text_Ha = rf'$\langle H_a \rangle$ for spin {spin_type} baryon'

        label_Ha = label_text_Ha if label_text_Ha not in added_labels else None

        ax.errorbar(
            j,
            # data_Ha["fit_fcn"][0],
            Ha_cntrl,
            yerr=Ha_err,
            marker=markers[plot_configurations[baryon]["spin_parity"].count('3') + 1],
            **kwargs[plot_configurations[baryon]["spin_parity"].count('3') + 1-1],
            # color=colors[plot_configurations[baryon]["spin_parity"].count('3') + 1],
            color=Ha_green[plot_configurations[baryon]["spin_parity"].count('3') + 1-1],
            # **errorbar_kwargs,
            label=label_Ha,
            # alpha=0.7,
        )

        if label_Ha:
            added_labels.add(label_text_Ha)

        label_text_Hag = rf'$\langle H_a^g \rangle$ for spin {spin_type} baryon'

        label_Hag = label_text_Hag if label_text_Hag not in added_labels else None

        ax.errorbar(
            j,
            Hag_cntrl,
            yerr=Hag_err,
            marker=markers[plot_configurations[baryon]["spin_parity"].count('3') + 1],
            **kwargs[plot_configurations[baryon]["spin_parity"].count('3') + 1-1],
            # color=colors[plot_configurations[baryon]["spin_parity"].count('3') + 1],
            # color='black',
            color=Hag_black[plot_configurations[baryon]["spin_parity"].count('3') + 1-1],
            # alpha=alphas[plot_configurations[baryon]["spin_parity"].count('3') + 1-1],
            label=label_Hag,
        )

        if label_Hag:
            added_labels.add(label_text_Hag)

        ax.annotate(
            plot_configurations[baryon]["label_y"].replace("(GeV)", ""),
            (j, Ha_cntrl+Ha_err),
            textcoords="offset points",
            fontsize=9,
            xytext=(0, 4),
            ha="center",
        )

    # ax.legend(ncol=4,loc='upper left')
    ax.text(-0.08, 1, "D", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.legend(loc='upper left',
        ncol=4,  # 或适当调整，比如 ncol=2 看分布是否更紧凑
        columnspacing=1.5,  # 减小列间距（默认 2.0）
        handletextpad=0.1,  # 图标和文字间距（默认 0.8）
        borderaxespad=0.1,  # legend 和坐标轴间距（默认 0.5）
        handlelength=1.0,   # 图例图标线段长度
        # bbox_to_anchor=(0.5, 1.05),  # 居中微调 legend 位置
        frameon=False,      # 可选，去掉框线以节省空间
        # fontsize=9          # 控制字体大小
    )
    ax.set_xticks([])
    ax.set_ylim(0.68, 2.2)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_ylabel(r'$\langle H_a^{(g)} \rangle_H$ (GeV)')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18, hspace=0.04)
    plt.savefig(f"{lqcd_data_path}/final_results/Hm_4in1_v2.pdf")

# Code to save the combined plot to a file.
plot_combined_baryon_data()
