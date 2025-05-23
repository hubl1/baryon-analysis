#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import sys
import os
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import platform
import json
import warnings
from filelock import FileLock

warnings.filterwarnings("ignore", category=DeprecationWarning)
# from save_windows import save_window, read_window, read_ini
from save_windows import read_ini
from utils.create_plot import create_plot
from utils.c2pt_io_h5 import h5_get_pp2pt
from utils.image_concat import (
    get_concat_h_cut,
    get_concat_v_cut,
    get_concat_h_resize,
)
from params.params import (
    ns_arr,
    alttc_gv,
    fm2GeV,
    name
)
from c2pt_meff_avg import run_meff_avg_light_quark

tag = sys.argv[1]
nensemble = int(sys.argv[2])
ii = int(sys.argv[3])
jj = int(sys.argv[4])
mud = float(sys.argv[5])
ms = float(sys.argv[6])
sms = sys.argv[6]
ratio = int(sys.argv[7])
param = sys.argv[8]
cov_enabled = bool(int(sys.argv[9]))
baryon = sys.argv[10]
alttc_method =sys.argv[11]
ii_uni =int(sys.argv[12])
baryon_meson_h5_prefix = sys.argv[13]

# alttc = alttc_arr[nensemble]
alttc = alttc_gv[nensemble].mean
# alttc = gv.mean(alttc_gv[nensemble])
prefix="mu"+ sys.argv[5]+ "_ms"+ sys.argv[6]+ "_rs"+ sys.argv[7]



if baryon == 'PION' and nensemble == 5:
    modified_prefix=re.sub(r"ms-0\.\d+", "ms-0.2310", prefix)
    pp2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{modified_prefix}.wp.{baryon}.extra.h5')
    wwpp2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{modified_prefix}.ww.{baryon}.extra.h5')
    a4p2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{modified_prefix}.wp.{baryon}_A4.extra.h5')
else:
    pp2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{prefix}.wp.{baryon}.h5')
    wwpp2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{prefix}.ww.{baryon}.h5')
    a4p2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{prefix}.wp.{baryon}_A4.h5')

n_bootstrap = 4000
pp2pt/= ns_arr[nensemble] ** 3
a4p2pt/= ns_arr[nensemble] ** 3
wwpp2pt/= ns_arr[nensemble] ** 3

with open(os.path.expanduser("./params/ensemble_used_confs_common.json"), 'r', encoding='utf-8') as f:
    common_conf_data = json.load(f)
# print(confs,common_conf_data[f'ensemble{nensemble}'])
# print(len(confs),len(common_conf_data[f'ensemble{nensemble}']))
# exit()

# print(pp2pt.shape)
# print(confs.shape)
index_conf=[ i for i in range(len(confs)) if confs[i] in common_conf_data[f'ensemble{nensemble}']]
# print(index_conf)
# print(set(range(146))-set(index_conf))
# exit()
# if nensemble != 11:
pp2pt=pp2pt[index_conf,:]
# print(pp2pt[0, :])

alttc_samples = np.random.normal(
    # loc=gv.mean(alttc), scale=gv.sdev(alttc), size=n_bootstrap
    loc=gv.mean(alttc), scale=0, size=n_bootstrap
)
print(pp2pt.shape)
print(a4p2pt.shape)
print(wwpp2pt.shape)
ZA=np.ones(20)
ZP=np.ones(20)

def block_average(arr, target_length):
    input_length = arr.shape[0]
    if input_length <= target_length:
        return arr

    output = np.zeros((target_length, *arr.shape[1:]))

    blockSize_1 = int(input_length / target_length)
    blockSize_2 = blockSize_1 + 1
    n_blocks_2 = input_length % target_length
    n_blocks_1 = target_length - n_blocks_2

    start = 0
    for _ in range(n_blocks_1):
        end = start + blockSize_1
        output[_] = np.mean(arr[start:end], axis=0)
        start = end

    for _ in range(n_blocks_1, target_length):
        end = start + blockSize_2
        output[_] = np.mean(arr[start:end], axis=0)
        start = end

    return output


if param == "all":
    print("all confs, no blocking")
    # 在这里插入你的代码

    # 如果第5个参数是一个数字，执行另一段代码
elif param.isdigit():
    num = int(param)
    print(f"block into {num} effective confs")
    # 在这里插入你的代码
    pp2pt = block_average(pp2pt, num)
    a4p2pt = block_average(a4p2pt, num)
    wwpp2pt = block_average(wwpp2pt, num)

else:
    print("无法识别的参数，请输入'all'或一个数字。")

Ncnfg = pp2pt.shape[0]
# print(Ncnfg)
T = pp2pt.shape[1]
T_hlf = T // 2
# ? average forward/backward propgation, for better symmetry, and keep the 0 point
pp2pt = np.hstack(
    (pp2pt[:, 0:1], (pp2pt[:, 1 : T_hlf + 1] + pp2pt[:, -1 : T_hlf - 1 : -1]) / 2)
)
wwpp2pt = np.hstack(
    (wwpp2pt[:, 0:1], (wwpp2pt[:, 1 : T_hlf + 1] + wwpp2pt[:, -1 : T_hlf - 1 : -1]) / 2)
)
a4p2pt = np.hstack(
    (a4p2pt[:, 0:1], (a4p2pt[:, 1 : T_hlf + 1] - a4p2pt[:, -1 : T_hlf - 1 : -1]) / 2)
)

# n_bootstrap = Ncnfg*10
n_bootstrap = 4000


def bootstrap_resample(data, indices):
    data_sample = data[indices, :]
    data_sample_mean = np.mean(data_sample, axis=0)
    return data_sample_mean


pp2pt_bootstrap = np.zeros((n_bootstrap, pp2pt.shape[1]))
wwpp2pt_bootstrap = np.zeros((n_bootstrap, wwpp2pt.shape[1]))
a4p2pt_bootstrap = np.zeros((n_bootstrap, a4p2pt.shape[1]))


def check_and_create_file(nensemble):
    # dir_path = "/Users/hubl/baryon_all_data_analysis/bootstrap_and_plot_unite_index/indices"
    dir_path = "./bootstrap_and_plot_light_quark_h5/indices"
    # file_name = f'indices_nens{nensemble}.npy'
    file_name = f"indices_nens{nensemble}.npy"
    file_path = os.path.join(dir_path, file_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.isfile(file_path):
        data = np.load(file_path)
    else:
        indices = np.random.choice(
            np.arange(pp2pt.shape[0]), size=(n_bootstrap, pp2pt.shape[0]), replace=True
        )
        np.save(file_path, indices)
        data = indices

    return data


indices = check_and_create_file(nensemble)
print(indices.shape)

for i in range(n_bootstrap):
    pp2pt_bootstrap[i, :] = bootstrap_resample(pp2pt, indices[i, :])
    wwpp2pt_bootstrap[i, :] = bootstrap_resample(wwpp2pt, indices[i, :])
    a4p2pt_bootstrap[i, :] = bootstrap_resample(a4p2pt, indices[i, :])


def check_and_print_nan(array, array_name):
    if np.isnan(array).any():
        print(f"Warning: {array_name} contains NaN value(s).")


pp2pt_cntrl = np.mean(pp2pt_bootstrap, axis=0)  # jack-knife mean
pp2pt_cov = np.cov(np.transpose(pp2pt_bootstrap, axes=(1, 0)))  # jack-knife covariance
# pp2pt_err = np.sqrt(Ncnfg-1)*np.std(pp2pt_bootstrap,axis=0,ddof=1) # jack-knife std
pp2pt_err = np.std(pp2pt_bootstrap, axis=0, ddof=1)  # jack-knife std
# print(pp2pt_cntrl)
# print(pp2pt_err)
# exit()

wwpp2pt_cntrl = np.mean(wwpp2pt_bootstrap, axis=0)  # jack-knife mean
wwpp2pt_cov = np.cov(
    np.transpose(wwpp2pt_bootstrap, axes=(1, 0))
)  # jack-knife covariance
# wwpp2pt_err = np.sqrt(Ncnfg-1)*np.std(wwpp2pt_bootstrap,axis=0,ddof=1)
wwpp2pt_err = np.std(wwpp2pt_bootstrap, axis=0, ddof=1)

a4p2pt_cntrl = np.mean(a4p2pt_bootstrap, axis=0)  # jack-knife mean
a4p2pt_cov = np.cov(
    np.transpose(a4p2pt_bootstrap, axes=(1, 0))
)  # jack-knife covariance
# a4p2pt_err = np.sqrt(Ncnfg-1)*np.std(a4p2pt_bootstrap,axis=0,ddof=1)
a4p2pt_err = np.std(a4p2pt_bootstrap, axis=0, ddof=1)

pp2pt_pw_ratio_bootstrap = np.sqrt(pp2pt_bootstrap / wwpp2pt_bootstrap)
pp2pt_pw_ratio_cntrl = np.mean(pp2pt_pw_ratio_bootstrap, axis=0)  # jack-knife mean
pp2pt_pw_ratio_cov = np.cov(
    np.transpose(pp2pt_pw_ratio_bootstrap, axes=(1, 0))
)  # jack-knife covariance
# pp2pt_pw_ratio_err = np.sqrt(Ncnfg-1)*np.std(pp2pt_pw_ratio_bootstrap,axis=0,ddof=1)
pp2pt_pw_ratio_err = np.std(pp2pt_pw_ratio_bootstrap, axis=0, ddof=1)

a4p_pp_ratio_bootstrap = np.zeros((Ncnfg, T - 2))
# Create a shifted view of a4p2pt_bootstrap
a4p2pt_bootstrap_left = a4p2pt_bootstrap[:, : T_hlf - 1]
a4p2pt_bootstrap_right = a4p2pt_bootstrap[:, 2 : T_hlf + 1]
# Create a shifted view of pp2pt_bootstrap
pp2pt_bootstrap_center = pp2pt_bootstrap[:, 1:T_hlf]
# print(pp2pt_bootstrap_center[0,:])
# print(a4p2pt_bootstrap_left[0,:])
# print(a4p2pt_bootstrap_right[0,:])
# Compute the result
a4p_pp_ratio_bootstrap = (
    a4p2pt_bootstrap_left - a4p2pt_bootstrap_right
) / pp2pt_bootstrap_center
a4p_pp_ratio_cntrl = np.mean(a4p_pp_ratio_bootstrap, axis=0)  # jack-knife mean
a4p_pp_ratio_cov = np.cov(
    np.transpose(a4p_pp_ratio_bootstrap, axes=(1, 0))
)  # jack-knife covariance
# a4p_pp_ratio_err = np.sqrt(Ncnfg-1)*np.std(a4p_pp_ratio_bootstrap,axis=0,ddof=1)
a4p_pp_ratio_err = np.std(a4p_pp_ratio_bootstrap, axis=0, ddof=1)

t_ary = np.array(range(0, T_hlf + 1))
# T_strt,T_end,ini_p0=read_ini(nensemble,ii,"../find_ini_joint_fit/json_hand/pion_found_ini.json")
T_strt, T_end, ini_p0 = read_ini(
    nensemble, ii_uni, "params/fit_init_values/"+baryon+"_fpi_mq_mPS_found_ini.json"
)

tsep_dctnry = {
    "combined": np.concatenate(
        (t_ary[T_strt:T_end], t_ary[T_strt:T_end], t_ary[T_strt:T_end]), axis=0
    )
}

# a4p_pp_ratio start at 1 and end at T_hlf thus -1 in index
combined_data = np.concatenate(
    (
        a4p_pp_ratio_bootstrap[:, T_strt - 1 : T_end - 1],
        pp2pt_pw_ratio_bootstrap[:, T_strt:T_end],
        pp2pt_bootstrap[:, T_strt:T_end],
    ),
    axis=1,
)
combined_data_cntrl = np.mean(combined_data, axis=0)  # jack-knife mean
combined_data_err = np.std(combined_data, axis=0, ddof=1)
combined_data_cov = np.cov(
    np.transpose(combined_data, axes=(1, 0))
)  # jack-knife covariance
# fake cov in corder not to consider covs between different channels
fake_combined_cov = np.zeros((3 * (T_end - T_strt), 3 * (T_end - T_strt)))
fake_combined_cov[: T_end - T_strt, : T_end - T_strt] = combined_data_cov[
    : T_end - T_strt, : T_end - T_strt
]
fake_combined_cov[
    T_end - T_strt : 2 * (T_end - T_strt), T_end - T_strt : 2 * (T_end - T_strt)
] = combined_data_cov[
    T_end - T_strt : 2 * (T_end - T_strt), T_end - T_strt : 2 * (T_end - T_strt)
]
fake_combined_cov[
    2 * (T_end - T_strt) : 3 * (T_end - T_strt),
    2 * (T_end - T_strt) : 3 * (T_end - T_strt),
] = combined_data_cov[
    2 * (T_end - T_strt) : 3 * (T_end - T_strt),
    2 * (T_end - T_strt) : 3 * (T_end - T_strt),
]

ffake_combined_cov = np.zeros_like(combined_data_cov)
np.fill_diagonal(ffake_combined_cov, combined_data_cov.diagonal())

np.set_printoptions(threshold=sys.maxsize, linewidth=800, precision=10, suppress=True)

fake_dctnry = {"combined": gv.gvar(combined_data_cntrl, fake_combined_cov)}
c2pt_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
nocov_c2pt_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
ffake_dctnry = {"combined": gv.gvar(combined_data_cntrl, ffake_combined_cov)}

def ft_mdls(t_dctnry, p):
    mdls = {}
    ts = t_dctnry["combined"][2 * (T_end - T_strt) : 3 * (T_end - T_strt)]
    if ii < 6:
        mdls["combined"] = np.concatenate(
            (
                ZP[nensemble]
                / ZA[nensemble]
                * p["mR"]
                * np.ones(T_end - T_strt)
                * np.sinh(p["mPS"])
                / p["mPS"]
                * 4,
                (p["mPS"] ** 2)
                / (2 * ZP[nensemble] * p["mR"] * np.sqrt(p["Zwp"]))
                * np.ones(T_end - T_strt)
                * p["fpi"],
                p["Zwp"]
                / (2 * p["mPS"])
                * np.exp(-p["mPS"] * T_hlf)
                * np.cosh(p["mPS"] * (ts - T_hlf))
                * 2,
            ),
            axis=0,
        )
    else:
        mdls["combined"] = np.concatenate(
            (
                ZP[nensemble]
                / ZA[nensemble]
                * np.ones(T_end - T_strt)
                * np.sinh(p["mPS"])
                / p["mPS"]
                * 4
                * p["mR"]
                + p["c1"] * np.exp(-p["DeltaE"] * ts),
                (p["mPS"] ** 2)
                / (2 * ZP[nensemble] * p["mR"] * np.sqrt(p["Zwp"]))
                * np.ones(T_end - T_strt)
                * p["fpi"]
                + p["c2"] * np.exp(-p["DeltaE"] * ts),
                p["Zwp"]
                / (2 * p["mPS"])
                * np.exp(-p["mPS"] * T_hlf)
                * np.cosh(p["mPS"] * (ts - T_hlf))
                + p["Zwp2"]
                / (2 * (p["mPS"] + p["DeltaE"]))
                * np.exp(-(p["mPS"] + p["DeltaE"]) * T_hlf)
                * np.cosh((p["mPS"] + p["DeltaE"]) * (ts - T_hlf)),
            ),
            axis=0,
        )
    return mdls


if nensemble == 5 or nensemble == 10:
    fit = lsqfit.nonlinear_fit(
        data=(tsep_dctnry, nocov_c2pt_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True
        # data=(tsep_dctnry, ffake_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True
    )  # ? excute the fit
else:
    fit = lsqfit.nonlinear_fit(
        data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True
    )  # ? excute the fit


def print_dict(data):
    for key, value in data.items():
        print(f"{key}: {value}")


data = {
    # 'mR': fit.p['mR'].mean * fm2GeV / alttc,
    # 'mPS': fit.p['mPS'].mean * fm2GeV / alttc,
    "mPS^2": (fit.p["mPS"].mean * fm2GeV / alttc) ** 2,
    "fpi": fit.p["fpi"].mean * fm2GeV / alttc,
    # 'Zwp': fit.p['Zwp'].mean * (fm2GeV / alttc)**4,
    # 'chi2': fit.chi2 / fit.dof,
    # 'Q': fit.Q
}

output_txt = (
    fit.format(True)
    + "lmass="
    + str(ii)
    + " smass="
    + str(jj)
    + " T_start="
    + str(T_strt)
    + "   T_end="
    + str(T_end)
    + "   baryon="
    + baryon
    + "\nmPS = "
    + str(fit.p["mPS"] * fm2GeV / alttc)
    + "GeV name="
    + name[nensemble]
    + " Ncnfg="
    + str(Ncnfg)
    + " #ensemble="
    + str(nensemble)
)
print(output_txt)

img = Image.new(
    mode="RGB", size=(1000, (output_txt.count("\n") + 1) * 29 + 5), color="black"
)

# 检查操作系统
if platform.system() == "Darwin":
    fnt = ImageFont.truetype("Monaco.ttf", 25)
elif platform.system() == "Linux":
    # 获取当前系统的主机名
    hostname = os.uname().nodename


    # 根据主机名选择字体
    if hostname == "LQCD":
        # 如果主机名是LQCD，使用Monaco字体
        fnt = ImageFont.truetype("/home/hubl/.fonts/Monaco.ttf", 25)
    else:
        # 否则使用DejaVuSansMono字体
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 25
        )

ImageDraw.Draw(img).text((0, 0), output_txt, font=fnt, fill=(255, 255, 255))
if not os.path.exists("./temp"):
    os.makedirs("./temp")
img.save(
    "./temp/"
    + name[nensemble]
    + "_mu"
    + sys.argv[5]
    + "_ms"
    + sys.argv[6]
    + "_rs"
    + sys.argv[7]
    + "_"
    + baryon
    + "_fit_params.png"
)

mb_mr_temp = {
    "mR": np.zeros((n_bootstrap), dtype=float),
    "mPS": np.zeros((n_bootstrap), dtype=float),
    "fpi": np.zeros((n_bootstrap), dtype=float),
    "Zwp": np.zeros((n_bootstrap), dtype=float),
    "chi2": np.zeros((n_bootstrap), dtype=float),
    "Q": np.zeros((n_bootstrap), dtype=float),
}

def single_fit(
    i,
    cov_enabled,
    combined_data,
    combined_data_cov,
    combined_data_err,
    tsep_dctnry,
    ft_mdls,
    ini_p0,
    fm2GeV,
    alttc_samples,
):
    # if cov_enabled:
    if nensemble==5 or nensemble==10 or not cov_enabled:
        err_dctnry = {"combined": gv.gvar(combined_data[i, :], combined_data_err)}
    else:
        err_dctnry = {"combined": gv.gvar(combined_data[i, :], combined_data_cov)}

    fit = lsqfit.nonlinear_fit(
        data=(tsep_dctnry, err_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True
    )

    t_lst = fit.data[0]['combined']
    return {
        "mR": fit.p["mR"].mean * fm2GeV / alttc_samples[i],
        "mPS": fit.p["mPS"].mean * fm2GeV / alttc_samples[i],
        "fpi": fit.p["fpi"].mean * fm2GeV / alttc_samples[i],
        "Zwp": fit.p["Zwp"].mean * (fm2GeV / alttc_samples[i]) ** 4,
        "chi2": fit.chi2 / fit.dof,
        "Q": fit.Q,
        'a4p_pp_ratio_dat' : np.array([c.mean for c in fit.data[1]['combined'][0:T_end-T_strt]]),
        'a4p_pp_ratio_fit_fcn' : np.array([c.mean for c in fit.fcn({'combined':t_lst}, fit.p)['combined'][0:T_end-T_strt]]),
        'pp2pt_pw_ratio_dat' : np.array([c.mean for c in fit.data[1]['combined'][(T_end-T_strt):2*(T_end-T_strt)]]),
        'pp2pt_pw_ratio_fit_fcn' : np.array([c.mean for c in fit.fcn({'combined':t_lst}, fit.p)['combined'][(T_end-T_strt):2*(T_end-T_strt)]]),
        'pp2pt_dat' : np.array([c.mean for c in fit.data[1]['combined'][2*(T_end-T_strt):3*(T_end-T_strt)]]),
        'pp2pt_fit_fcn' : np.array([c.mean for c in fit.fcn({'combined':t_lst}, fit.p)['combined'][2*(T_end-T_strt):3*(T_end-T_strt)]]),
        't_lst': t_lst
}

def parallel_fit_tqdm(
    n_bootstrap,
    n_jobs,
    cov_enabled,
    combined_data,
    combined_data_cov,
    combined_data_err,
    tsep_dctnry,
    ft_mdls,
    ini_p0,
    fm2GeV,
    alttc_samples,
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_fit)(
            i,
            cov_enabled,
            combined_data,
            combined_data_cov,
            combined_data_err,
            tsep_dctnry,
            ft_mdls,
            ini_p0,
            fm2GeV,
            alttc_samples,
        )
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress", unit="bootstrap")
    )
    mb_mr_temp = {
        "mR": np.array([res["mR"] for res in results]),
        "mPS": np.array([res["mPS"] for res in results]),
        "fpi": np.array([res["fpi"] for res in results]),
        "Zwp": np.array([res["Zwp"] for res in results]),
        "chi2": np.array([res["chi2"] for res in results]),
        "Q": np.array([res["Q"] for res in results]),
    }
    # return mb_mr_temp
    t_lst=results[0]['t_lst']
    # print(t_lst)
    res_cntrl_err = {}
    for k in ['a4p_pp_ratio_dat', 'a4p_pp_ratio_fit_fcn', 'pp2pt_pw_ratio_dat', 'pp2pt_pw_ratio_fit_fcn', 'pp2pt_dat', 'pp2pt_fit_fcn', 'mPS']:
        res_array = np.array([res[k] for res in results])
        res_cntrl_err[k + '_cntrl'] = np.mean(res_array, axis=0)
        res_cntrl_err[k + '_err'] = np.std(res_array, axis=0, ddof=1)
    return mb_mr_temp, res_cntrl_err, t_lst


mb_mr_temp, res_cntrl_err, t_lst = parallel_fit_tqdm(
    n_bootstrap=n_bootstrap,
    n_jobs=-1,
    cov_enabled=cov_enabled,
    combined_data=combined_data,
    combined_data_cov=combined_data_cov,
    combined_data_err=combined_data_err,
    tsep_dctnry=tsep_dctnry,
    ft_mdls=ft_mdls,
    ini_p0=ini_p0,
    fm2GeV=fm2GeV,
    alttc_samples=alttc_samples,
)
# print(mb_mr_temp)
print(gv.gvar(res_cntrl_err['mPS_cntrl'],res_cntrl_err['mPS_err']))
# print(res_cntrl_err['mPS_cntrl'])
# print(res_cntrl_err['mPS_err'])
# exit(
pydata_path=baryon_meson_h5_prefix+"/pydata_eff_mass/"

if cov_enabled:
    bootstrap_resample_data = pydata_path+baryon+f"_fpi_mq_mPS_bootstrap_cov_{param}.npy"
else:
    bootstrap_resample_data = pydata_path+baryon+f"_fpi_mq_mPS_bootstrap_no_cov_{param}.npy"
# lock_path = bootstrap_resample_data + ".lock"
dir_name, file_name = os.path.split(bootstrap_resample_data)
lock_file = os.path.join(dir_name, '.'+file_name+'.lock')
from copy import deepcopy

with FileLock(lock_file):
    try:
        with open(bootstrap_resample_data, "rb") as f:
            mb_mr_data = np.load(f, allow_pickle=True).item()
    except FileNotFoundError:
        # n_ensembles = 12
        # ensemble_keys = [f'ensemble{i}' for i in range(n_ensembles)]
        # default_keys = {i: None for i in range(3)}
        n_ensembles = 20
        ensemble_keys = [f"ensemble{i}" for i in range(n_ensembles)]
        # default_keys = {
        #     i: {j: -1 * np.ones(n_bootstrap) for j in range(9)} for i in range(9)
        # }
        default_keys = {i: {j: -1*np.ones(n_bootstrap) for j in range(3,6)} for i in range(3)}

        mb_mr_data = {
            # 'mB': {key: deepcopy(default_keys) for key in ensemble_keys},
            "T_strt": {key: deepcopy(default_keys) for key in ensemble_keys},
            "T_end": {key: deepcopy(default_keys) for key in ensemble_keys},
            "mR": {key: deepcopy(default_keys) for key in ensemble_keys},
            "mPS": {key: deepcopy(default_keys) for key in ensemble_keys},
            "fpi": {key: deepcopy(default_keys) for key in ensemble_keys},
            "Zwp": {key: deepcopy(default_keys) for key in ensemble_keys},
            "chi2": {key: deepcopy(default_keys) for key in ensemble_keys},
            "Q": {key: deepcopy(default_keys) for key in ensemble_keys},
        }

mb_mr_data["T_strt"][f"ensemble{nensemble}"][ii][jj] = T_strt
mb_mr_data["T_end"][f"ensemble{nensemble}"][ii][jj] = T_end
mb_mr_data["mR"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["mR"]
mb_mr_data["mPS"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["mPS"]
mb_mr_data["fpi"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["fpi"]
mb_mr_data["Zwp"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["Zwp"]
mb_mr_data["chi2"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["chi2"]
mb_mr_data["Q"][f"ensemble{nensemble}"][ii][jj] = mb_mr_temp["Q"]

# print(gv.gvar(res_cntrl_err['mPS_cntrl'],res_cntrl_err['mPS_err']))
# exit()
with FileLock(lock_file):
    with open(bootstrap_resample_data, "wb") as f:
        np.save(f, mb_mr_data)
# os.system('./c2pt_meff_avg.py '+(str(sys.argv[1])+' '+str(sys.argv[3])+' '+str(sys.argv[2])+' "'+str(gv.gvar(res_cntrl_err['mPS_cntrl'],res_cntrl_err['mPS_err']))+'"'+' '+str(T_strt)+' '+str(T_end))+' '+str(sys.argv[5])+' '+str(sys.argv[7])+' '+'mu'+sys.argv[5]+'_ms'+sys.argv[6]+' '+baryon+' '+str(jj)+' '+str(int(cov_enabled)))
mp_gvar = gv.gvar(res_cntrl_err["mPS_cntrl"], res_cntrl_err["mPS_err"])
run_meff_avg_light_quark(
    prefix,
    ii,
    nensemble,
    str(mp_gvar),
    T_strt,
    T_end-1,
    tag,
    baryon,
    jj,
    cov_enabled,
    baryon_meson_h5_prefix
)
# print(res_cntrl_err["pp2pt_fit_fcn_cntrl"])
t_ary = t_lst[0:T_end-T_strt]
# a4p2pt_y_lim_strt,a4p2pt_y_lim_end=read_ylimit(nensemble,ii,"./json/dimensionless/a4p2pt_plot_limit.json")
# pp2pt_pw_ratio_y_lim_strt,pp2pt_pw_ratio_y_lim_end=read_ylimit(nensemble,ii,"./json/dimensionless/pp2pt_pw_ratio_plot_limit.json")
create_plot(0.8/alttc,t_ary=t_ary,
            x=np.array(range(2, T_hlf+1)),
            y=a4p_pp_ratio_cntrl/4,
            y_err=a4p_pp_ratio_err/4,
            # y_lim_strt=a4p2pt_y_lim_strt,
            # y_lim_end=a4p2pt_y_lim_end,
            y_lim_strt=0,
            y_lim_end=0,
            xlabel=r'$t/a$',
            ylabel=r'$\frac{C_{2,wp}^{A_4P}(t-a)-C_{2,wp}^{A_4P}(t+a)}{C_{2,wp}^{PP}}$',
            label1=r"frwrd/bckwrd avg. $\frac{C_{2,wp}^{A_4P}(t-a)-C_{2,wp}^{A_4P}(t+a)}{C_{2,wp}^{PP}}$",
            label2="best fit",
            prefix=name[nensemble]+"_"+prefix+'_'+baryon+"_a4p_pp_ratio_fit",
            T_hlf=T_hlf,
            fit_fcn=res_cntrl_err['a4p_pp_ratio_fit_fcn_cntrl']/4,
            fit_fcn_err=res_cntrl_err['a4p_pp_ratio_fit_fcn_err']/4)

create_plot(0.8/alttc,t_ary,
            x=np.array(range(0, T_hlf+1)),
            y=pp2pt_pw_ratio_cntrl,
            y_err=pp2pt_pw_ratio_err,
            # y_lim_strt=pp2pt_pw_ratio_y_lim_strt,
            # y_lim_end=pp2pt_pw_ratio_y_lim_end,
            y_lim_strt=0,
            y_lim_end=0,
            xlabel=r'$t/a$',
            ylabel=r'$\sqrt{\frac{C_{2,wp}^{PP}}{C_{2,ww}^{PP}}}$',
            label1=r"frwrd/bckwrd avg. $\sqrt{\frac{C_{2,wp}^{PP}}{C_{2,ww}^{PP}}}$",
            label2="best fit",
            prefix=name[nensemble]+"_"+prefix+'_'+baryon+"_pp2pt_pw_ratio_fit",
            T_hlf=T_hlf,
            fit_fcn=res_cntrl_err['pp2pt_pw_ratio_fit_fcn_cntrl'],
            fit_fcn_err=res_cntrl_err['pp2pt_pw_ratio_fit_fcn_err'])

create_plot(0.8/alttc,t_ary,
            x=np.array(range(0, T_hlf+1)),
            y=pp2pt_cntrl,
            y_err=pp2pt_err,
            y_lim_strt=0,
            y_lim_end=0,
            xlabel=r'$t/a$',
            ylabel=r'$C_{2,wp}^{PP}$',
            label1=r"frwrd/bckwrd avg. $C_{2,wp}^{PP}$",
            label2="best fit",
            prefix=name[nensemble]+"_"+prefix+'_'+baryon+"_pp2pt_fit",
            T_hlf=T_hlf,
            fit_fcn=res_cntrl_err['pp2pt_fit_fcn_cntrl'],
            fit_fcn_err=res_cntrl_err['pp2pt_fit_fcn_err'],
            yscale="log")

im0 = Image.open('./temp/'+name[nensemble]+"_"+prefix+"_"+baryon+'_pp2pt_fit.png')
im1 = Image.open('./temp/'+name[nensemble]+"_"+prefix+"_"+baryon+'_m_eff.png')
im2 = Image.open('./temp/'+name[nensemble]+"_"+prefix+"_"+baryon+'_fit_params.png')
im3 = Image.open('./temp/'+name[nensemble]+"_"+prefix+"_"+baryon+'_a4p_pp_ratio_fit.png')
im4 = Image.open('./temp/'+name[nensemble]+"_"+prefix+"_"+baryon+'_pp2pt_pw_ratio_fit.png')
tmp0=get_concat_h_resize(im0, im1)
tmp2=get_concat_h_cut(im3, im4)
tmp3=get_concat_v_cut(tmp0, tmp2)
tmp1=get_concat_h_resize(tmp3, im2)
if not os.path.exists('./temp/combined_pngs'):
    os.makedirs('./temp/combined_pngs')
tmp1.save("./temp/combined_pngs/"+name[nensemble]+"_"+prefix+"_"+baryon+".pdf")
