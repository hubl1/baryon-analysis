import matplotlib.pyplot as plt
import numpy as np
def create_plot(t0, t_ary, x, y, y_err,y_lim_strt,y_lim_end, xlabel, ylabel, label1, label2, prefix, T_hlf, fit_fcn, fit_fcn_err, yscale=None, color1="red", color2="black", mark="x", alpha=1):
    if y_lim_strt==0 and y_lim_end==0:
        y_lim_strt=np.max(fit_fcn)-30*np.max(fit_fcn_err)
        y_lim_end=np.max(fit_fcn)+30*np.max(fit_fcn_err)
    # 创建一个字典保存所有数据
    plot_data = {
        't_ary': t_ary,
        'x': x,
        'y': y,
        'y_err': y_err,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'label1': label1,
        'label2': label2,
        'prefix': prefix,
        'T_hlf': T_hlf,
        'fit_fcn': fit_fcn,
        'fit_fcn_err': fit_fcn_err,
        'yscale': yscale,
        'color1': color1,
        'color2': color2,
        'mark': mark,
        'alpha': alpha,
    }

    # 设置参数
    size = [0.2,0.15,0.78,0.8]
    errorbar_kwargs = {
        "linewidth": 0,
        "elinewidth": 3,
        "capthick": 5,
        "capsize": 5,
        "mew": 1,
        "linestyle": "none",
        "fillstyle": 'none'
    }

    # 创建图像
    plt.figure(figsize=(15, 10), dpi=140)
    ax = plt.axes(size)

    # 画散点图
    ax.errorbar(x, y, yerr=y_err, color=color1, marker=mark, markersize=15, label=label1, alpha=alpha, **errorbar_kwargs)

    # 画线
    ax.plot(t_ary, fit_fcn, color=color2, label=label2, linewidth=3)

    # t0 = 0.8/alttc  # 您需要设定 t0 的值
    ax.axvline(x=t0, linestyle='--', color='grey',linewidth=3)
    # 画阴影区域
    ax.fill_between(t_ary, fit_fcn - fit_fcn_err, fit_fcn + fit_fcn_err, alpha=0.6, color="grey")

    # 如果是logscale则自动获取y轴范围
    if yscale == 'log':
        ax.set_yscale('log')
    else:
        # ax.set_ylim([fit_fcn.mean()-15*np.max(fit_fcn_err), fit_fcn.mean()+15*np.max(fit_fcn_err)])
        ax.set_ylim([y_lim_strt, y_lim_end])

    # 其他设置
    ax.minorticks_on()
    plt.xlabel(xlabel, fontsize=35, labelpad=3)
    plt.ylabel(ylabel, fontsize=35, labelpad=16)

    ax.tick_params(axis='both', colors='black')
    # ax.tick_params(which='major', direction='in', width=5, length=12)
    ax.tick_params(which='major', direction='in', width=3, length=12)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    ax.set_xlim([-1, T_hlf+1])


    # 设置边框颜色
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(3)

    # 设置图例
    # ax.legend(loc='upper left', frameon=False, fontsize=25, ncol=2)
    ax.legend(loc='best', frameon=False, fontsize=25, ncol=2)

    # 保存图片
    plt.savefig(f"./temp/{prefix}.png", dpi=200)
    # plt.savefig(f"./4in1/{prefix}.pdf")
    plt.close()

    # 将数据保存为.npy文件
    np.save(f"./temp/dict/{prefix}.npy", plot_data)

def create_plot_ex(t0, t_ary, all_t_ary, x, y, y_err,y_lim_strt,y_lim_end, xlabel, ylabel, label1, label2, prefix, T_hlf, fit_fcn, fit_fcn_err, all_fit_fcn, all_fit_fcn_err, yscale=None, color1="red", color2="black", mark="x", alpha=1):
    if y_lim_strt==0 and y_lim_end==0:
        y_lim_strt=np.max(fit_fcn)-30*np.max(fit_fcn_err)
        y_lim_end=np.max(fit_fcn)+30*np.max(fit_fcn_err)
        # y_lim_strt=np.max(fit_fcn)-0.2
        # y_lim_end=np.max(fit_fcn)+0.2
    # print(y_lim_strt)
    # print(y_lim_end)
    # 创建一个字典保存所有数据
    plot_data = {
        't_ary': t_ary,
        'all_t_ary': all_t_ary,
        'x': x,
        'y': y,
        'y_err': y_err,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'label1': label1,
        'label2': label2,
        'prefix': prefix,
        'T_hlf': T_hlf,
        'fit_fcn': fit_fcn,
        'fit_fcn_err': fit_fcn_err,
        'all_fit_fcn': all_fit_fcn,
        'all_fit_fcn_err': all_fit_fcn_err,
        'yscale': yscale,
        'color1': color1,
        'color2': color2,
        'mark': mark,
        'alpha': alpha,
        't0': t0,
    }

    # 设置参数
    size = [0.2,0.15,0.78,0.8]
    errorbar_kwargs = {
        "linewidth": 0,
        "elinewidth": 3,
        "capthick": 5,
        "capsize": 5,
        "mew": 1,
        "linestyle": "none",
        "fillstyle": 'none'
    }

    # 创建图像
    plt.figure(figsize=(15, 10), dpi=140)
    ax = plt.axes(size)

    # 画散点图
    ax.errorbar(x, y, yerr=y_err, color=color1, marker=mark, markersize=15, label=label1, alpha=alpha, **errorbar_kwargs)

    # 画线
    ax.plot(all_t_ary, all_fit_fcn, color='green',linestyle='--', linewidth=3)
    ax.plot(t_ary, fit_fcn, color=color2, label=label2, linewidth=3)

    # t0 = 0.8/alttc  # 您需要设定 t0 的值
    ax.axvline(x=t0, linestyle='--', color='grey', linewidth=3)
    # 画阴影区域
    ax.fill_between(all_t_ary, all_fit_fcn - all_fit_fcn_err, all_fit_fcn + all_fit_fcn_err, alpha=0.3, color="grey")

    # 如果是logscale则自动获取y轴范围
    if yscale == 'log':
        ax.set_yscale('log')
    else:
        # ax.set_ylim([fit_fcn.mean()-15*np.max(fit_fcn_err), fit_fcn.mean()+15*np.max(fit_fcn_err)])
        ax.set_ylim([y_lim_strt, y_lim_end])

    # 其他设置
    ax.minorticks_on()
    plt.xlabel(xlabel, fontsize=35, labelpad=3)
    plt.ylabel(ylabel, fontsize=35, labelpad=16)

    ax.tick_params(axis='both', colors='black')
    ax.tick_params(which='major', direction='in', width=3, length=12)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    ax.set_xlim([-1, T_hlf+1])


    # 设置边框颜色
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(3)

    # 设置图例
    # ax.legend(loc='upper left', frameon=False, fontsize=25, ncol=2)
    ax.legend(loc='best', frameon=False, fontsize=25, ncol=2)

    # 保存图片
    plt.savefig(f"./temp/{prefix}.png", dpi=200)
    # plt.savefig(f"./4in1/{prefix}.pdf")
    plt.close()

    # 将数据保存为.npy文件
    np.save(f"./temp/dict/{prefix}.npy", plot_data)

def create_plot_from_dict(plot_data_name):
    plot_data=np.load(plot_data_name,allow_pickle=True)[()]
    # Extracting all the necessary data from the dictionary
    t_ary = plot_data['t_ary']
    x = plot_data['x']
    y = plot_data['y']
    y_err = plot_data['y_err']
    xlabel = plot_data['xlabel']
    ylabel = plot_data['ylabel']
    label1 = plot_data['label1']
    label2 = plot_data['label2']
    prefix = plot_data['prefix']
    T_hlf = plot_data['T_hlf']
    fit_fcn = plot_data['fit_fcn']
    fit_fcn_err = plot_data['fit_fcn_err']
    yscale = plot_data.get('yscale', None) # Using get in case 'yscale' is not provided
    color1 = plot_data.get('color1', "red")
    color2 = plot_data.get('color2', "black")
    mark = plot_data.get('mark', "x")
    alpha = plot_data.get('alpha', 1)
    all_t_ary = plot_data.get('all_t_ary', t_ary) # Assuming 'all_t_ary' is same as 't_ary' if not provided
    all_fit_fcn = plot_data.get('all_fit_fcn', fit_fcn)
    all_fit_fcn_err = plot_data.get('all_fit_fcn_err', fit_fcn_err)
    t0 = plot_data.get('t0', 0) # Assuming t0=0 if not provided

    # Determine y-axis limits if not provided
    y_lim_strt, y_lim_end = plot_data.get('y_lim_strt', 0), plot_data.get('y_lim_end', 0)
    if y_lim_strt == 0 and y_lim_end == 0:
        y_lim_strt = np.max(fit_fcn) - 30 * np.max(fit_fcn_err)
        y_lim_end = np.max(fit_fcn) + 30 * np.max(fit_fcn_err)

    # Settings for the error bars
    errorbar_kwargs = {
        "linewidth": 0,
        "elinewidth": 3,
        "capthick": 5,
        "capsize": 5,
        "mew": 1,
        "linestyle": "none",
        "fillstyle": 'none'
    }

    # Creating the figure and axis
    plt.figure(figsize=(15, 10), dpi=140)
    ax = plt.axes([0.2, 0.15, 0.78, 0.8])

    # t0 = 0.8/alttc  # 您需要设定 t0 的值
    ax.axvline(x=t0, linestyle='--', color='grey', linewidth=3)
    # Plotting the data
    ax.errorbar(x, y, yerr=y_err, color=color1, marker=mark, markersize=15, label=label1, alpha=alpha, **errorbar_kwargs)
    ax.plot(all_t_ary, all_fit_fcn, color='green', linestyle='--', linewidth=3)
    ax.plot(t_ary, fit_fcn, color=color2, label=label2, linewidth=3)
    ax.axvline(x=t0, linestyle='--', color='grey', linewidth=3)
    ax.fill_between(all_t_ary, all_fit_fcn - all_fit_fcn_err, all_fit_fcn + all_fit_fcn_err, alpha=0.3, color="grey")

    # Setting the y-axis scale
    if yscale == 'log':
        ax.set_yscale('log')
    else:
        ax.set_ylim([y_lim_strt, y_lim_end])

    # Additional settings
    ax.set_xlim([-1, T_hlf + 1])
    ax.set_xlabel(xlabel, fontsize=35, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=35, labelpad=16)
    ax.legend(loc='best', frameon=False, fontsize=25, ncol=2)
    ax.tick_params(axis='both', which='major', direction='in', width=3, length=12)
    ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(3)

    # Saving the figure and data
    plt.savefig(f"./ttt.pdf")
    # np.save(f"./dict/{prefix}.npy", plot_data)

def create_plot_compare_from_dict(plot_data_name,plot_data_name2):
    plot_data=np.load(plot_data_name,allow_pickle=True)[()]
    plot_data2=np.load(plot_data_name2,allow_pickle=True)[()]
    # Extracting all the necessary data from the dictionary
    t_ary = plot_data['t_ary']
    x = plot_data['x']
    y = plot_data['y']
    y_err = plot_data['y_err']
    xlabel = plot_data['xlabel']
    ylabel = plot_data['ylabel']
    label1 = plot_data['label1']
    label2 = plot_data['label2']
    prefix = plot_data['prefix']
    T_hlf = plot_data['T_hlf']
    fit_fcn = plot_data['fit_fcn']
    fit_fcn_err = plot_data['fit_fcn_err']
    yscale = plot_data.get('yscale', None) # Using get in case 'yscale' is not provided
    color1 = plot_data.get('color1', "red")
    color2 = plot_data.get('color2', "black")
    mark = plot_data.get('mark', "x")
    alpha = plot_data.get('alpha', 1)
    all_t_ary = plot_data.get('all_t_ary', t_ary) # Assuming 'all_t_ary' is same as 't_ary' if not provided
    all_fit_fcn = plot_data.get('all_fit_fcn', fit_fcn)
    all_fit_fcn_err = plot_data.get('all_fit_fcn_err', fit_fcn_err)
    t0 = plot_data.get('t0', 0) # Assuming t0=0 if not provided


    y2 = plot_data2['y']
    y_err2 = plot_data2['y_err']
    fit_fcn2 = plot_data2['fit_fcn']
    fit_fcn_err2 = plot_data2['fit_fcn_err']
    all_fit_fcn2 = plot_data2.get('all_fit_fcn', fit_fcn)
    all_fit_fcn_err2 = plot_data2.get('all_fit_fcn_err', fit_fcn_err)

    # Determine y-axis limits if not provided
    y_lim_strt, y_lim_end = plot_data.get('y_lim_strt', 0), plot_data.get('y_lim_end', 0)
    if y_lim_strt == 0 and y_lim_end == 0:
        y_lim_strt = np.max(fit_fcn) - 30 * np.max(fit_fcn_err)
        y_lim_end = np.max(fit_fcn) + 30 * np.max(fit_fcn_err)

    # Settings for the error bars
    errorbar_kwargs = {
        "linewidth": 0,
        "elinewidth": 3,
        "capthick": 5,
        "capsize": 5,
        "mew": 1,
        "linestyle": "none",
        "fillstyle": 'none'
    }

    # Creating the figure and axis
    plt.figure(figsize=(15, 10), dpi=140)
    ax = plt.axes([0.2, 0.15, 0.78, 0.8])

    # t0 = 0.8/alttc  # 您需要设定 t0 的值
    # ax.axvline(x=t0, linestyle='--', color='grey', linewidth=3)
    # Plotting the data
    ax.errorbar(x, y, yerr=y_err, color=color1, marker=mark, markersize=15, label=label1, alpha=alpha, **errorbar_kwargs)
    ax.errorbar(x+0.1, y2, yerr=y_err2, color='grey', marker='x', markersize=15, label=label1, alpha=0.7, **errorbar_kwargs)

    ax.plot(all_t_ary, all_fit_fcn, color='green', linestyle='--', linewidth=3)
    ax.plot(t_ary, fit_fcn, color=color2, label='correlated fit', linewidth=3)
    ax.plot(all_t_ary, all_fit_fcn2, color='blue', linestyle='--', linewidth=3)
    ax.plot(t_ary, fit_fcn2, color='grey', label='uncorrelated fit', linewidth=3)

    ax.axvline(x=t0, linestyle='--', color='grey', linewidth=3)
    ax.fill_between(all_t_ary, all_fit_fcn - all_fit_fcn_err, all_fit_fcn + all_fit_fcn_err, alpha=0.3, color="green")
    ax.fill_between(all_t_ary, all_fit_fcn2 - all_fit_fcn_err2, all_fit_fcn2 + all_fit_fcn_err2, alpha=0.3, color="blue")

    # Setting the y-axis scale
    if yscale == 'log':
        ax.set_yscale('log')
    else:
        ax.set_ylim([y_lim_strt, y_lim_end])

    # Additional settings
    ax.set_xlim([-1, T_hlf + 1])
    ax.set_xlabel(xlabel, fontsize=35, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=35, labelpad=16)
    ax.legend(loc='best', frameon=False, fontsize=25, ncol=2)
    ax.tick_params(axis='both', which='major', direction='in', width=3, length=12)
    ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(3)

    # Saving the figure and data
    plt.savefig(f"./ttt.pdf")
    # np.save(f"./dict/{prefix}.npy", plot_data)
