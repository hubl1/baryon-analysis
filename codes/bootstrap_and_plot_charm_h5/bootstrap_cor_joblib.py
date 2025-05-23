#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import gvar as gv
import lsqfit
import sys
import json
from PIL import Image, ImageFont, ImageDraw
import os
import re
import warnings
from filelock import FileLock
from tqdm import tqdm
from joblib import Parallel, delayed
import platform

from c2pt_meff_avg import run_meff_avg_charm
from save_windows import (
    read_ini_v2,
)

from params.params import alttc_gv, alttc_omega_gv,ms_arr, fm2GeV,name
from utils.c2pt_io_h5 import h5_get_pp2pt
from utils.create_plot import create_plot
from utils.image_concat import (
    get_concat_v_resize,
    get_concat_h_resize,
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

prefix = sys.argv[1]
ii = int(sys.argv[2])
nensemble = int(sys.argv[3])
mB = float(sys.argv[4])
tag = sys.argv[5]
baryon = sys.argv[6]
cov_enabled = bool(int(sys.argv[7]))
jj = int(sys.argv[8])
ii_uni = int(sys.argv[9])
jj_uni = int(sys.argv[10])
kk = int(sys.argv[11])
baryon_meson_h5_prefix = sys.argv[12]

param = "400"

two_state_enabled = True
alttc = gv.mean(alttc_gv[nensemble])
if baryon == 'LAMBDA_C':
    modified_prefix=re.sub(r"ms-0\.\d+", f'ms{ms_arr[nensemble]:.4f}', prefix)
    pp2pt_loc, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{modified_prefix}.wp.{baryon}.h5')
    # print(f'python3 ./find_ini_2pt.py ../baryon_meson_data_h5/{tag}/c2pt_{modified_prefix}.wp.{baryon}.h5 ', nensemble,baryon)
else:
    pp2pt_loc, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{prefix}.wp.{baryon}.h5')
    # print(f'python3 ./find_ini_2pt.py ../baryon_meson_data_h5/{tag}/c2pt_{prefix}.wp.{baryon}.h5 ', nensemble,baryon)

with open(os.path.expanduser("./params/ensemble_used_confs_common.json"), 'r', encoding='utf-8') as f:
    common_conf_data = json.load(f)
# print(confs,common_conf_data[f'ensemble{nensemble}'])
index_conf=[ i for i in range(len(confs)) if confs[i] in common_conf_data[f'ensemble{nensemble}']]
pp2pt=pp2pt_loc[index_conf,:]

Ncnfg = pp2pt.shape[0]
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

else:
    print("无法识别的参数，请输入'all'或一个数字。")
# print(Ncnfg)
T = pp2pt.shape[1]
T_hlf = T // 2
# ? average forward/backward propgation, for better symmetry, and keep the 0 point
pp2pt = np.hstack(
    (pp2pt[:, 0:1], (pp2pt[:, 1 : T_hlf + 1] + pp2pt[:, -1 : T_hlf - 1 : -1]) / 2)
)
n_bootstrap = 4000

def bootstrap_resample(data, indices):
    data_sample = data[indices, :]
    data_sample_mean = np.mean(data_sample, axis=0)
    return data_sample_mean

pp2pt_bootstrap = np.zeros((n_bootstrap, pp2pt.shape[1]))

def check_and_create_file(nensemble):
    dir_path = "./bootstrap_and_plot_light_quark_h5/indices"
    file_name = f'indices_nens{nensemble}.npy'
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

for i in range(n_bootstrap):
    pp2pt_bootstrap[i, :] = bootstrap_resample(pp2pt, indices[i, :])

pp2pt_cntrl = np.mean(pp2pt_bootstrap, axis=0)  # jack-knife mean
pp2pt_cov = np.cov(np.transpose(pp2pt_bootstrap, axes=(1, 0)))  # jack-knife covariance
# pp2pt_err = np.sqrt(Ncnfg-1)*np.std(pp2pt_bootstrap,axis=0,ddof=1) # jack-knife std
pp2pt_err = np.std(pp2pt_bootstrap, axis=0, ddof=1)  # jack-knife std
pp2pt_bootstrap_center = pp2pt_bootstrap[:, 1:T_hlf]

t_ary = np.array(range(0, T_hlf + 1))
# T_strt,T_end,ini_p0=read_ini(nensemble,ii,"./json/huzc_window_ini.json")
# print(os.path.expanduser("~"))
T_strt, T_end, ini_p0, two_state_enabled = read_ini_v2(
    nensemble,
    ii_uni,
    kk,
    os.path.expanduser("params/fit_init_values/" + baryon + "_found_ini.json"),
    two_state_enabled,
)

print(two_state_enabled)
# exit()
print(ii, jj, T_strt, T_end, ini_p0)
# exit()
tsep_dctnry = {
    "combined": np.concatenate(
        (t_ary[T_strt:T_end], t_ary[T_strt:T_end], t_ary[T_strt:T_end]), axis=0
    )
}
baryon_tsep_dctnry = {"baryon": t_ary[T_strt:T_end]}

# ffake_combined_cov=np.zeros_like(combined_data_cov)
# np.fill_diagonal(ffake_combined_cov, combined_data_cov.diagonal())

baryon_data = pp2pt_bootstrap[:, T_strt:T_end]
baryon_data_cntrl = np.mean(baryon_data, axis=0)  # jack-knife mean
baryon_data_err = np.std(baryon_data, axis=0, ddof=1)
baryon_data_cov = np.cov(
    np.transpose(baryon_data, axes=(1, 0))
)  # jack-knife covariance

np.set_printoptions(threshold=sys.maxsize, linewidth=800, precision=10, suppress=True)

if cov_enabled:
    baryon_dctnry = {"baryon": gv.gvar(baryon_data_cntrl, baryon_data_cov)}
else:
    baryon_dctnry = {"baryon": gv.gvar(baryon_data_cntrl, baryon_data_err)}

if not two_state_enabled:

    def baryon_ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry["baryon"]
        mdls["baryon"] = p["c1"] * np.exp(p["mp"] * (-ts))
        return mdls

else:

    def baryon_ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry["baryon"]
        mdls["baryon"] = p["c1"] * np.exp(p["mp"] * (-ts)) + p["c2"] * np.exp(
            p["mp2"] * (-ts)
        )
        return mdls


fit = lsqfit.nonlinear_fit(
    data=(baryon_tsep_dctnry, baryon_dctnry),
    svdcut=1e-6,
    fcn=baryon_ft_mdls,
    p0=ini_p0,
    debug=True,
)  # ? excute the fit

output_txt = (
    fit.format(True)
    + "lmass="
    + str(ii)
    + " smass="
    + str(jj)
    + " cmass="
    + str(kk)
    + " T_start="
    + str(T_strt)
    + "   T_end="
    + str(T_end)
    + "   baryon="
    + baryon
    + "\nmp = "
    + str(fit.p["mp"] * fm2GeV / alttc)
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
    "./temp/" + name[nensemble] + "_" + prefix + "_" + baryon + "_fit_params_real.png"
)

if not two_state_enabled:
    mb_mr_temp = {
        "mp": np.zeros((n_bootstrap), dtype=float),
        "c1": np.zeros((n_bootstrap), dtype=float),
    }
else:
    mb_mr_temp = {
        "mp": np.zeros((n_bootstrap), dtype=float),
        "c1": np.zeros((n_bootstrap), dtype=float),
        "mp2": np.zeros((n_bootstrap), dtype=float),
        "c2": np.zeros((n_bootstrap), dtype=float),
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
    alttc,
):
    if cov_enabled:
        err_dctnry = {"baryon": gv.gvar(combined_data[i, :], combined_data_cov)}
    else:
        err_dctnry = {"baryon": gv.gvar(combined_data[i, :], combined_data_err)}

    # fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, err_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True)
    while True:
        try:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fit = lsqfit.nonlinear_fit(
                data=(tsep_dctnry, err_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True
            )
            break
        except:
            continue

    t_ary = fit.data[0]["baryon"][0 : T_end - T_strt]
    t_lst = fit.data[0]["baryon"]
    if not two_state_enabled:
        return {
            "c1": fit.p["c1"].mean,
            "mp": fit.p["mp"].mean * fm2GeV / alttc,
            "chi2": fit.chi2 / fit.dof,
            "Q": fit.Q,
            "pp2pt_dat": np.array([c.mean for c in fit.data[1]["baryon"]]),
            "pp2pt_fit_fcn": np.array(
                [c.mean for c in fit.fcn({"baryon": t_lst}, fit.p)["baryon"]]
            ),
            "t_lst": t_lst,
        }
    else:
        return {
            "c1": fit.p["c1"].mean,
            "mp": fit.p["mp"].mean * fm2GeV / alttc,
            "c2": fit.p["c2"].mean,
            "mp2": fit.p["mp2"].mean * fm2GeV / alttc,
            "chi2": fit.chi2 / fit.dof,
            "Q": fit.Q,
            "pp2pt_dat": np.array([c.mean for c in fit.data[1]["baryon"]]),
            "pp2pt_fit_fcn": np.array(
                [c.mean for c in fit.fcn({"baryon": t_lst}, fit.p)["baryon"]]
            ),
            "t_lst": t_lst,
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
    alttc,
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
            alttc,
        )
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress", unit="bootstrap")
    )
    if not two_state_enabled:
        mb_mr_temp = {
            "c1": np.array([res["c1"] for res in results]),
            "mp": np.array([res["mp"] for res in results]),
            "chi2": np.array([res["chi2"] for res in results]),
            "Q": np.array([res["Q"] for res in results]),
        }
    else:
        mb_mr_temp = {
            "c1": np.array([res["c1"] for res in results]),
            "mp": np.array([res["mp"] for res in results]),
            "c2": np.array([res["c2"] for res in results]),
            "mp2": np.array([res["mp2"] for res in results]),
            "chi2": np.array([res["chi2"] for res in results]),
            "Q": np.array([res["Q"] for res in results]),
        }
    t_lst = results[0]["t_lst"]
    # print(t_lst)
    res_cntrl_err = {}
    for k in ["pp2pt_dat", "pp2pt_fit_fcn", "mp"]:
        res_array = np.array([res[k] for res in results])
        res_cntrl_err[k + "_cntrl"] = np.mean(res_array, axis=0)
        res_cntrl_err[k + "_err"] = np.std(res_array, axis=0, ddof=1)
    return mb_mr_temp, res_cntrl_err, t_lst


mb_mr_temp, res_cntrl_err, t_lst = parallel_fit_tqdm(
    n_bootstrap=n_bootstrap,
    n_jobs=-1,
    cov_enabled=cov_enabled,
    combined_data=baryon_data,
    combined_data_cov=baryon_data_cov,
    combined_data_err=baryon_data_err,
    tsep_dctnry=baryon_tsep_dctnry,
    ft_mdls=baryon_ft_mdls,
    ini_p0=ini_p0,
    fm2GeV=fm2GeV,
    alttc=alttc,
)

print(res_cntrl_err["mp_cntrl"])
print(res_cntrl_err["mp_err"])
# exit()
pydata_path=baryon_meson_h5_prefix+"/pydata_eff_mass/"
if not os.path.exists(pydata_path):
    os.makedirs(pydata_path)
from copy import deepcopy

if cov_enabled:
    bootstrap_resample_data = pydata_path + baryon + "_bootstrap_cov_400.npy"
else:
    bootstrap_resample_data = pydata_path + baryon + "_bootstrap_no_cov_400.npy"

# lock_path = bootstrap_resample_data+'.lock'
dir_name, file_name = os.path.split(bootstrap_resample_data)
lock_file = os.path.join(dir_name, "." + file_name + ".lock")

with FileLock(lock_file):
    try:
        with open(bootstrap_resample_data, "rb") as f:
            mb_mr_data = np.load(f, allow_pickle=True).item()
    except FileNotFoundError:
        n_ensembles = 20
        ensemble_keys = [f"ensemble{i}" for i in range(n_ensembles)]
        default_keys = {
            i: {
                j: {k: -1 * np.ones(n_bootstrap) for k in range(6, 9)}
                for j in range(3, 6)
            }
            for i in range(3)
        }  # if not two_state_enabled:
        if False:
            mb_mr_data = {
                "mB": {key: deepcopy(default_keys) for key in ensemble_keys},
                "T_strt": {key: deepcopy(default_keys) for key in ensemble_keys},
                "T_end": {key: deepcopy(default_keys) for key in ensemble_keys},
                "mp": {key: deepcopy(default_keys) for key in ensemble_keys},
                "c1": {key: deepcopy(default_keys) for key in ensemble_keys},
                "chi2": {key: deepcopy(default_keys) for key in ensemble_keys},
                "Q": {key: deepcopy(default_keys) for key in ensemble_keys},
            }
        else:
            mb_mr_data = {
                "mB": {key: deepcopy(default_keys) for key in ensemble_keys},
                "T_strt": {key: deepcopy(default_keys) for key in ensemble_keys},
                "T_end": {key: deepcopy(default_keys) for key in ensemble_keys},
                "mp": {key: deepcopy(default_keys) for key in ensemble_keys},
                "c1": {key: deepcopy(default_keys) for key in ensemble_keys},
                "mp2": {key: deepcopy(default_keys) for key in ensemble_keys},
                "c2": {key: deepcopy(default_keys) for key in ensemble_keys},
                "chi2": {key: deepcopy(default_keys) for key in ensemble_keys},
                "Q": {key: deepcopy(default_keys) for key in ensemble_keys},
            }

# Update the values for each ii and jj combination.
# print('ii,jj=',ii,jj)
mb_mr_data["mB"][f"ensemble{nensemble}"][ii][jj][kk] = mB
mb_mr_data["T_strt"][f"ensemble{nensemble}"][ii][jj][kk] = T_strt
mb_mr_data["T_end"][f"ensemble{nensemble}"][ii][jj][kk] = T_end
mb_mr_data["mp"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["mp"]
mb_mr_data["c1"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["c1"]
if two_state_enabled:
    mb_mr_data["mp2"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["mp2"]
    mb_mr_data["c2"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["c2"]
mb_mr_data["chi2"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["chi2"]
mb_mr_data["Q"][f"ensemble{nensemble}"][ii][jj][kk] = mb_mr_temp["Q"]
# print(mb_mr_data['mR'][f'ensemble{nensemble}'][ii][jj])
# print(mb_mr_data['mp'][f'ensemble{nensemble}'][ii][jj])

with FileLock(lock_file):
    with open(bootstrap_resample_data, "wb") as f:
        np.save(f, mb_mr_data)

mp_gvar = gv.gvar(res_cntrl_err["mp_cntrl"], res_cntrl_err["mp_err"])

run_meff_avg_charm(
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
    two_state_enabled,
    kk,
    baryon_meson_h5_prefix
)

t_ary = t_lst[0 : T_end - T_strt]
create_plot(
    0.8 / alttc,
    t_ary,
    x=np.array(range(0, T_hlf + 1)),
    y=pp2pt_cntrl,
    y_err=pp2pt_err,
    y_lim_strt=0,
    y_lim_end=0,
    # y_lim_strt=res_cntrl_err['mp_cntrl']-0.2,
    # y_lim_end=res_cntrl_err['mp_cntrl']+0.2,
    xlabel=r"$t/a$",
    ylabel=r"$C_{2,wp}^{PP}$",
    label1=r"frwrd/bckwrd avg. $C_{2,wp}^{PP}$",
    label2="best fit",
    prefix=name[nensemble] + "_" + prefix + "_" + baryon + "_fit_real",
    T_hlf=T_hlf,
    fit_fcn=res_cntrl_err["pp2pt_fit_fcn_cntrl"],
    fit_fcn_err=res_cntrl_err["pp2pt_fit_fcn_err"],
    yscale="log",
)


im0 = Image.open(
    "./temp/" + name[nensemble] + "_" + prefix + "_" + baryon + "_fit_real.png"
)
im1 = Image.open(
    "./temp/" + name[nensemble] + "_" + prefix + "_" + baryon + "_m_eff_real.png"
)
im2 = Image.open(
    "./temp/" + name[nensemble] + "_" + prefix + "_" + baryon + "_fit_params_real.png"
)
tmp0 = get_concat_v_resize(im0, im1)
tmp1 = get_concat_h_resize(tmp0, im2)
if not os.path.exists("./temp/combined_pngs"):
    os.makedirs("./temp/combined_pngs")
tmp1.save("./temp/combined_pngs/" + name[nensemble] + "_" + prefix + "_" + baryon + ".pdf")
