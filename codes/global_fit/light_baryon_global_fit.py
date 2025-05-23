#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import pandas as pd
import lsqfit
import sys
from pathlib import Path

import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import os

# from params import alttc_gv, Z_P_gv1, Z_P_gv2, Z_A_gv,alttc_omega_gv, a_fm
from params.params import (
    a_fm,
    Z_V,
    Z_A_Z_V,
    a_fm_independent,
    a_fm_correlated,
    name
)
from params.plt_labels import plot_configurations

# alttc_gv=alttc_omega_gv

# np.set_printoptions(threshold=sys.maxsize, linewidth=800, precision=10, suppress=True)
plt.style.use("utils/science.mplstyle")

baryon = sys.argv[1]
error_source = sys.argv[2]
lqcd_data_path= sys.argv[4]

if error_source == "alttc_sys":
    a_fm_independent += gv.sdev(a_fm_correlated)
    a_fm_correlated += gv.sdev(a_fm_correlated)

selected_ensemble_indices = [*range(8), 8, 9, 10, 11, 12, 14]
# selected_ensemble_indices=range(15)
# selected_ensemble_indices=[0,1,2,4,5,6,7,9,10,14]
alttc_index = np.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 1, 3, 5, 2])
alttc_gv = a_fm[[alttc_index[idx] for idx in selected_ensemble_indices]]
# Z_A_gv=np.array(Z_A_gv[selected_ensemble_indices])
alttc_gv_new = a_fm_independent[[alttc_index[idx] for idx in selected_ensemble_indices]]
# Z_P_gv1, Z_P_gv2, Z_A_gv
ml_col_indices = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
ms_col_indices = np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1]) + 3
ns_info = [24, 24, 32, 32, 48, 48, 32, 48, 32, 48, 48, 28, 36, 48, 64]
Z_A_gv = (Z_A_Z_V * Z_V)[[alttc_index[idx] for idx in selected_ensemble_indices]]

markers = ["d", "o", ".", "8", "x", "+", "v", "^", "<", ">", "s"]

cov_enabled = True
additive_Ca = '--additive_Ca' in sys.argv
CasQ = not additive_Ca
native_fpi = False
Ca4Q = False
n_bootstrap = 4000


def package_joint_fit_data(path_to_pydata, particle, key, cov_enabled, quark):
    selected_ensembles = ["ensemble" + str(idx) for idx in selected_ensemble_indices]

    baryon_data = {}
    if cov_enabled:
        baryon_data[particle] = np.load(
            path_to_pydata + particle + "_bootstrap_cov_400.npy",
            allow_pickle=True,
        )[()]
    else:
        baryon_data[particle] = np.load(
            path_to_pydata + particle + "_bootstrap_no_cov_400.npy",
            allow_pickle=True,
        )[()]

    # print(baryon_data)
    fit_array_cntrl = {}
    fit_array_cov = {}
    fit_array_boot = {}

    # selected_ensembles = list(baryon_data[particle][key].keys())
    # print(selected_ensembles)
    # ms_col_indices = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

    for ii in selected_ensemble_indices:
        ensemble_key = "ensemble" + str(ii)
        ensemble_array_list = []
        for ml_col in range(3):
            for ms_col in range(3, 6):
                # 根据quark的值计算条件
                condition = False
                if quark == "s":
                    condition = ml_col == ml_col_indices[ii]
                elif quark == "l":
                    condition = ms_col == ms_col_indices[ii]
                elif quark == "a":
                    condition = True
                # if ms_col == ms_col_indices[ii]:
                if condition:
                    # print(ii, ml_col, ms_col)
                    # print(ensemble_values)
                    # print(ensemble_key)
                    # print(baryon_data[particle])
                    ensemble_values = baryon_data[particle][key][ensemble_key]
                    ensemble_array_list.append(ensemble_values[ml_col][ms_col])
                    # print(ensemble_key,np.array(ensemble_array_list).shape)
                    ensemble_array = np.array(ensemble_array_list)
        # print(ensemble_array)
        fit_array_cntrl[ensemble_key] = np.mean(np.array(ensemble_array), axis=1)
        fit_array_cov[ensemble_key] = np.cov(
            np.transpose(np.array(ensemble_array), axes=(0, 1))
        )
        fit_array_err = np.std(np.array(ensemble_array), axis=0, ddof=1)
        fit_array_boot[ensemble_key] = np.array(ensemble_array)

    fit_array_combined_boot = np.concatenate(
        [fit_array_boot[ensemble_key] for ensemble_key in selected_ensembles]
    )
    print(particle, fit_array_cntrl)

    # return data_obs
    return fit_array_combined_boot


def predict_y_slop(x_values, y_values, x_to_predict, degree=1):
    polynomial = np.poly1d(np.polyfit(x_values, y_values, 1))
    y_predicted = polynomial(x_to_predict)
    y_predicted_slop = np.polyder(polynomial)(x_to_predict)
    y_predicted_slop1 = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
    y_predicted_slop2 = (y_values[2] - y_values[0]) / (x_values[2] - x_values[0])
    y_predicted_slop3 = (y_values[2] - y_values[1]) / (x_values[2] - x_values[1])
    return y_predicted, y_predicted_slop


m_pi_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "PION_fpi_mq_mPS",
    "mPS",
    True,
    "l",
)
metas_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "ETA_S_fpi_mq_mPS",
    "mPS",
    True,
    "s",
)
ml_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "PION_fpi_mq_mPS",
    "mR",
    True,
    "l",
)
ms_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "ETA_S_fpi_mq_mPS",
    "mR",
    True,
    "s",
)
fpi_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "PION_fpi_mq_mPS",
    "fpi",
    True,
    "l",
)
mp_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    baryon,
    "mp",
    True,
    "a",
)

# print(mp_v_bt[:, 0])


alttc_cntrl = np.array(
    [
        np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in alttc_gv
        for _ in range(3)
    ]
)
alttc_cntrl_9 = np.array(
    [
        np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in alttc_gv
        for _ in range(9)
    ]
)
# a_fm_samples = np.array(
#     [
#         np.random.normal(loc=g.mean, scale=g.sdev * np.sqrt(1), size=n_bootstrap)
#         # np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
#         for g in a_fm_new
#         # for g in a_fm_new[alttc_index[selected_ensemble_indices]]
#     ]
# )

a_fm_samples = np.array(
    [
        # np.random.normal(loc=g.mean, scale=g.sdev * np.sqrt(1.8), size=n_bootstrap)
        np.random.normal(loc=g.mean, scale=g.sdev, size=n_bootstrap)
        # np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in a_fm_independent
        # for g in a_fm_new
        # for g in a_fm
    ]
)

normalized_gaussian = np.random.normal(loc=0, scale=1, size=n_bootstrap)
a_fm_correlated_error = np.array(
    [normalized_gaussian * g.sdev for g in a_fm_correlated]
)
if error_source == "alttc_stat":
    a_fm_samples = np.repeat(np.mean(a_fm_samples, axis=1)[:, np.newaxis], 4000, axis=1)

a_fm_samples += a_fm_correlated_error

ZA_fm_samples = np.array(
    [
        np.random.normal(loc=g.mean, scale=g.sdev * np.sqrt(1.8), size=n_bootstrap)
        # np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in Z_A_gv
        # for g in a_fm_new[alttc_index[selected_ensemble_indices]]
    ]
)
# print(a_fm_samples[:, 0])
alttc_samples = np.array(
    [
        a_fm_samples[alttc_index[idx]]
        for idx in selected_ensemble_indices
        for _ in range(3)
    ]
)
alttc_samples_9 = np.array(
    [
        a_fm_samples[alttc_index[idx]]
        for idx in selected_ensemble_indices
        for _ in range(9)
    ]
)
ZA_samples = np.array(
    [
        ZA_fm_samples[alttc_index[idx]]
        for idx in selected_ensemble_indices
        for _ in range(3)
    ]
)
# alttc_samples= np.array([ a_fm_samples[idx] for idx in range(len(selected_ensemble_indices)) for _ in range(3)])
# print(alttc_samples[:, 0])
# exit()
# print((alttc_samples)[:, 0])
mp_v_bt *= alttc_cntrl_9 / alttc_samples_9
m_pi_v_bt *= alttc_cntrl / alttc_samples
metas_v_bt *= alttc_cntrl / alttc_samples

for i, j in enumerate(selected_ensemble_indices):
    print(
        name[j],
        gv.gvar(
            np.mean(m_pi_v_bt[np.arange(i * 3, i * 3 + 3), :], axis=1),
            np.std(m_pi_v_bt[np.arange(i * 3, i * 3 + 3), :], axis=1),
        ),
    )

# exit()
# mp_v_bt *= alttc_cntrl/0.1973
# m_pi_v_bt *= alttc_cntrl/0.1973
# metas_v_bt *= alttc_cntrl/0.1973
fpi_v_bt *= ZA_samples * alttc_cntrl / alttc_samples
# print(gv.gvar(np.mean(m_pi_v_bt,axis=1),np.std(m_pi_v_bt,axis=1)))
# np.save('./m_pi_bootstrap_dimensionless.npy',m_pi_v_bt)
# np.save('./m_etas_bootstrap_dimensionless.npy',metas_v_bt)
# print(gv.gvar(np.mean(ms_v_bt,axis=1),np.std(ms_v_bt,axis=1)))
# exit()

ml_indices = np.array(
    [i * 3 + ml_col_indices[index] for i, index in enumerate(selected_ensemble_indices)]
).repeat(3)
ms_indices = np.array(
    [
        i * 3 + ms_col_indices[index] - 3
        for i, index in enumerate(selected_ensemble_indices)
    ]
).repeat(3)
# print(ml_indices)
# print(ms_indices)
# exit()
m_pi_sea_bt = m_pi_v_bt[ml_indices, :]
# if baryon == 'OMEGA':
#     m_pi_v_bt=m_pi_sea_bt
if error_source == "mpi":
    m_pi_sea_bt = np.repeat(np.mean(m_pi_sea_bt, axis=1)[:, np.newaxis], 4000, axis=1)
metas_sea_bt = metas_v_bt[ms_indices, :]
if error_source == "metas":
    metas_sea_bt = np.repeat(np.mean(metas_sea_bt, axis=1)[:, np.newaxis], 4000, axis=1)
ml_sea_bt = ml_v_bt[ml_indices, :]
ms_sea_bt = ms_v_bt[ms_indices, :]
tsep_dctnry = {"combined": np.arange(len(selected_ensemble_indices) * 3)}
ns = [ns_info[index] for index in selected_ensemble_indices]

etas_phys = 0.68963
mp_v_bt_etas069 = -1 * np.ones_like(metas_v_bt)
mp_v_bt_etas069_slop = -1 * np.ones_like(metas_v_bt)
for i in range(mp_v_bt_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(3):
            (
                mp_v_bt_etas069[n * 3 + ii, i],
                mp_v_bt_etas069_slop[n * 3 + ii, i],
            ) = predict_y_slop(
                metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                mp_v_bt[(9 * n + np.arange(3) + 3 * ii), i],
                etas_phys**2,
                degree=2,
            )

# print(gv.gvar(np.mean(mp_v_bt_etas069_slop,axis=1),np.std(mp_v_bt_etas069_slop,axis=1)))
# exit()
Hms = mp_v_bt_etas069_slop * etas_phys**2

combined_data = mp_v_bt_etas069
# print(combined_data)
# print(combined_data[:,0:2])
# exit()
combined_data_cntrl = np.mean(combined_data, axis=1)  # jack-knife mean
# print(combined_data.shape)
combined_data_err = np.std(combined_data, axis=1, ddof=1)  # jack-knife mean
# combined_cov = np.cov(np.transpose(pp2pt_jcknf,axes=(1,0))) # jack-knife covariance
# combined_data_cov = np.cov(combined_data)*chi2  # jack-knife covariance
combined_data_cov = np.cov(combined_data)
# n = combined_data_cov.shape[0]  # 33
# group_size = 3  # 每组的大小
# # 遍历矩阵中的所有元素
# for i in range(n):
#     for j in range(n):
#         # 如果 i 和 j 不在同一组，则将协方差设置为0
#         if i // group_size != j // group_size:
#             combined_data_cov[i, j] = 0

if cov_enabled == True:
    data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
else:
    data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
N_f = 2

# print("------------------")

# Assuming m_pi_phys is the physical pion mass
m_pi_phys = 0.135  # Adjust this value to the correct one
ms_phys = 0.0922  # Adjust this value to the correct one
fpi_phys = 0.122  # Adjust this value to the correct one
# print(fpi_v_bt[:, 0])


def Fpi_article(m_pi_sq, a):
    # Constants provided
    F = 0.08659
    alpha_4 = 0.342
    alpha_5 = -0.38
    d_pi_a = -5.56
    # f0 = 0.086
    Lambda_chi = 4 * np.pi * F

    # Calculate ell_4
    ell_4 = np.log(Lambda_chi**2 / m_pi_phys**2) + 0.5 * (alpha_5 + 2 * alpha_4)

    # Calculate F_pi
    F_pi = F * (
        1 - (m_pi_sq / Lambda_chi**2) * (np.log(m_pi_sq / m_pi_phys**2) - ell_4)
    )
    # *(1+d_pi_a*a**2)

    return np.sqrt(2) * F_pi
    # return 0.122
    # return np.sqrt(2)*f0*(1+m_pi_sq*1.7)


def vol_a_mdls(idct, p):
    mdls = {}
    i = idct["i"]
    m_pi_v = m_pi_v_bt[:, i]
    m_pi_sea = m_pi_sea_bt[:, i]
    metas_sea = metas_sea_bt[:, i]
    # print(metas_sea)
    # ms_sea = ms_sea_bt[:, i]
    # ml_sea = ml_sea_bt[:, i]
    # ml_v = ml_v_bt[:, i]
    # print(ml_v)
    # fpi=np.mean(fpi_v_bt,axis=1)
    # fpi=fpi_phys
    a = alttc_samples[:, i]
    if native_fpi:
        fpi = fpi_v_bt[:, i]
    else:
        fpi = np.array([Fpi_article(m_pi_v[j] ** 2, a[j]) for j in range(a.shape[0])])
    # print(fpi)
    # fpi = fpi_phys*alttc_cntrl[:,i]/alttc_samples[:,i]
    # m_pi_v = np.mean(m_pi_v_bt,axis=1)
    # m_pi_sea = np.mean(m_pi_sea_bt,axis=1)
    # ms_sea = np.mean(ms_sea_bt,axis=1)
    # fpi=np.mean(fpi_v_bt,axis=1)
    # a = gv.mean(alttc_gv_new).repeat(3)
    L = np.array(ns).repeat(3) * a
    m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
    # print(p["C5"].shape,ms_sea.shape)
    # print(m_pi_v)

    M0 = p["M0"]

    if baryon == "PROTON":
        Cv3 = (
            -(p["gA"] ** 2 - 4 * p["gA"] * p["g1"] - 5 * p["g1"] ** 2)
            * np.pi
            / (3 * (4 * np.pi * fpi) ** 2)
        )
        Cpq3 = (
            -(8 * p["gA"] ** 2 + 4 * p["gA"] * p["g1"] + 5 * p["g1"] ** 2)
            * np.pi
            / (3 * (4 * np.pi * fpi) ** 2)
        )
    elif baryon == 'DELTA':
        Cv3 = p["Cv3"]
        Cpq3 = p["Cpq3"]

    if baryon in ['PROTON', 'DELTA']:
        M_corr = (
            +p["Cv2"] * (m_pi_v**2 - m_pi_phys**2)
            + p["Cpq2"] * (m_pi_pq**2 - m_pi_phys**2)
            + Cv3 * (m_pi_v**3 - m_pi_phys**3)
            + Cpq3 * (m_pi_pq**3 - m_pi_phys**3)
        )
    elif baryon == 'OMEGA': 
        M_corr = (
            + p["Cs2"] * (m_pi_sea**2 - m_pi_phys**2)
        )
    else:
        M_corr = (
            +p["Cv2"] * (m_pi_v**2 - m_pi_phys**2)
            + p["Cpq2"] * (m_pi_pq**2 - m_pi_phys**2)
        )

    M_corr_s = p["C5"] * (metas_sea**2 - etas_phys**2)
    M_s = M_corr_s + M_corr
    M_s *= (1 + p["Cas"] * a**2) if CasQ else 1

    # M_corr_L = p["C4"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L)
    M_corr_L = p["C4"] * (m_pi_v**2 / L) * np.exp(-m_pi_v * L)

    M_corr_a = p["Ca2"] * a**2
    if Ca4Q:
        M_corr_a += p["Ca4"] * a**4

    mdls["combined"] = np.array(M_s + M_corr_L + M_corr_a + M0)
    return mdls


def fit_with_initial_values(initial_params):
    fit_vol = lsqfit.nonlinear_fit(
        data=({"i": 0}, data_dctnry),
        svdcut=1e-8,
        fcn=vol_a_mdls,
        p0=initial_params,
        debug=True,
    )
    return initial_params, fit_vol.chi2 / fit_vol.dof, fit_vol.Q


# 初始参数范围
if baryon != 'OMEGA':
    param_ranges = {
        "M0": (0.5, 1.5),
        "Cv2": (0, 1),
        "Cpq2": (0, 1),
        "C4": (-1, 1),
        "Ca2": (-1.5, 1.5),
        "C5": (-1, 1),
    }
    if baryon == "PROTON":
        param_ranges["gA"] = (0.1, 1.5)
        param_ranges["g1"] = (-1, 1)
    elif baryon == "DELTA":
        param_ranges["Cv3"] = (0, 1.5)
        param_ranges["Cpq3"] = (-1, 1)
else:
    param_ranges = {
        "M0": (0.5, 1.5),
        "Cs2": (0, 1),
        "C4": (-1, 1),
        "Ca2": (-1.5, 1.5),
        "C5": (-1, 1),
    }
if CasQ:
    param_ranges["Cas"] = (-1, 1)
if Ca4Q:
    param_ranges["Ca4"] = (-1, 1)

n_trials = 1000
best_params = None
best_chi2_dof = float("inf")
best_Q = 0


def generate_trial_params():
    return {
        param: np.random.uniform(low, high)
        for param, (low, high) in param_ranges.items()
    }


# 并行地尝试不同的参数初值，并进行拟合
results = Parallel(n_jobs=-1)(
    delayed(fit_with_initial_values)(generate_trial_params()) for _ in range(n_trials)
)

# 从结果中找到最佳拟合
for params, chi2_dof, Q in results:
    if chi2_dof < best_chi2_dof and Q > best_Q:
        best_chi2_dof = chi2_dof
        best_Q = Q
        best_params = params

print("Best Initial Parameters:", best_params)
print("Best chi2/dof:", best_chi2_dof)
print("Best Q:", best_Q)
# exit()
fit = lsqfit.nonlinear_fit(
    data=({"i": 0}, data_dctnry),
    svdcut=1e-8,
    fcn=vol_a_mdls,
    p0=best_params,
    debug=True,
)  # ? excute the fit
print(fit.chi2 / fit.dof)
fit_chi2 = fit.chi2 / fit.dof


# 替换函数
def replace_keys_with_names(text, names):
    lines = text.split("\n")
    for i, name in enumerate(names):
        replace = False
        new_lines = []
        for line in lines:
            if "--------------------------------------" in line:
                replace = True
            if replace:
                new_lines.append(
                    line.replace(f" {i} ", f"{i} " + name).replace(
                        "combined", "        "
                    )
                )
            else:
                new_lines.append(line)
        lines = new_lines
    return "\n".join(lines)


new_text = replace_keys_with_names(
    fit.format(True), (np.array(name)[selected_ensemble_indices]).repeat(3)
)
# print(fit.format(True))  # ? print out fit results
print(new_text)

def M(m_pi_v, m_pi_sea, ms_sea, a, L, params):
    M_0 = params["M0"]
    return M_0


# Calculate M at the physical point
M_phys = M(m_pi_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p)
# print(M_phys)
print(f"M at the physical point is: {fit.p['M0']} GeV")
# exit()


def single_fit(
    i,
    cov_enabled,
    combined_data,
    combined_data_cov,
    combined_data_err,
    tsep_dctnry,
    ft_mdls,
    ini_p0,
):
    if cov_enabled:
        err_dctnry = {"combined": gv.gvar(combined_data[:, i], combined_data_cov)}
    else:
        err_dctnry = {"combined": gv.gvar(combined_data[:, i], combined_data_err)}

    # fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, err_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = lsqfit.nonlinear_fit(
            data=({"i": i}, err_dctnry),
            svdcut=1e-8,
            fcn=vol_a_mdls,
            p0=ini_p0,
            debug=True,
        )

    if baryon != 'OMEGA':
        return_dict = {
            "M0": fit.p["M0"].mean,
            "Cv2": fit.p["Cv2"].mean,
            "Cpq2": fit.p["Cpq2"].mean,
            "Ca2": fit.p["Ca2"].mean,
            "C4": fit.p["C4"].mean,
            "C5": fit.p["C5"].mean,
            # "gA": fit.p["gA"].mean,
            # "g1": fit.p["g1"].mean,
            "M_phys": M(m_pi_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p).mean,
            "chi2": fit.chi2 / fit.dof,
            "Q": fit.Q,
        }
        if baryon == "PROTON":
            return_dict["g1"] = fit.p["g1"].mean
            return_dict["gA"] = fit.p["gA"].mean
        elif baryon == 'DELTA':
            return_dict["Cv3"] = fit.p["Cv3"].mean
            return_dict["Cpq3"] = fit.p["Cpq3"].mean
    else:
        return_dict = {
            "M0": fit.p["M0"].mean,
            "Cs2": fit.p["Cs2"].mean,
            "Ca2": fit.p["Ca2"].mean,
            "C4": fit.p["C4"].mean,
            "C5": fit.p["C5"].mean,
            "M_phys": M(m_pi_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p).mean,
            "chi2": fit.chi2 / fit.dof,
            "Q": fit.Q,
        }
    # if not additive_Ca:
    #     return_dict["Ca20"] = fit.p["Ca20"].mean
    # return_dict["Ca2s"] = fit.p["Ca2s"].mean
    if CasQ:
        return_dict["Cas"] = fit.p["Cas"].mean
    if Ca4Q:
        return_dict["Ca4"] = fit.p["Ca4"].mean

    return return_dict


# LEC = single_fit(1,
#     cov_enabled=cov_enabled,
#     combined_data=combined_data,
#     combined_data_cov=combined_data_cov,
#     combined_data_err=combined_data_err,
#     tsep_dctnry=tsep_dctnry,
#     ft_mdls=vol_a_mdls,
#     ini_p0=best_params,
# )


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
        )
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress", unit="bootstrap")
    )
    if baryon != 'OMEGA':
        LEC_temp = {
            "M0": np.array([res["M0"] for res in results]),
            "Cv2": np.array([res["Cv2"] for res in results]),
            "Cpq2": np.array([res["Cpq2"] for res in results]),
            "Ca2": np.array([res["Ca2"] for res in results]),
            "C4": np.array([res["C4"] for res in results]),
            "C5": np.array([res["C5"] for res in results]),
            "M_phys": np.array([res["M_phys"] for res in results]),
            "chi2": np.array([res["chi2"] for res in results]),
            "Q": np.array([res["Q"] for res in results]),
        }
        if baryon == "PROTON":
            LEC_temp["g1"] = np.array([res["g1"] for res in results])
            LEC_temp["gA"] = np.array([res["gA"] for res in results])
        elif baryon == "DELTA":
            LEC_temp["Cv3"] = np.array([res["Cv3"] for res in results])
            LEC_temp["Cpq3"] = np.array([res["Cpq3"] for res in results])
    else:
        LEC_temp = {
            "M0": np.array([res["M0"] for res in results]),
            "Cs2": np.array([res["Cs2"] for res in results]),
            "Ca2": np.array([res["Ca2"] for res in results]),
            "C4": np.array([res["C4"] for res in results]),
            "C5": np.array([res["C5"] for res in results]),
            "M_phys": np.array([res["M_phys"] for res in results]),
            "chi2": np.array([res["chi2"] for res in results]),
            "Q": np.array([res["Q"] for res in results]),
        }

    # if not additive_Ca:
    #     LEC_temp["Ca20"] = np.array([res["Ca20"] for res in results])
    # LEC_temp["Ca2s"] = np.array([res["Ca2s"] for res in results])

    if CasQ:
        LEC_temp["Cas"] = np.array([res["Cas"] for res in results])
    if Ca4Q:
        LEC_temp["Ca4"] = np.array([res["Ca4"] for res in results])

    return LEC_temp


n_bootstrap = 4000
LEC_mass = parallel_fit_tqdm(
    n_bootstrap=n_bootstrap,
    n_jobs=-1,
    cov_enabled=cov_enabled,
    combined_data=combined_data,
    combined_data_cov=combined_data_cov,
    combined_data_err=combined_data_err,
    tsep_dctnry=tsep_dctnry,
    ft_mdls=vol_a_mdls,
    ini_p0=best_params,
)
print(LEC_mass)

# 创建一个新的字典来存储结果
LEC_gvar = {}


def derivatives_Mcorr(m_pi_v, m_pi_sea, m_pi_phys, p, Cv3, Cpq3):
    # Compute m_pi_pq
    m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)

    # Compute partial derivatives of m_pi_pq
    dm_pi_pq_d_m_pi_v = m_pi_v / np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
    dm_pi_pq_d_m_pi_sea = m_pi_sea / np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)

    # Compute derivatives of M_corr
    dM_d_m_pi_v = (
        2 * p["Cv2"] * m_pi_v
        + 2 * p["Cpq2"] * m_pi_pq * dm_pi_pq_d_m_pi_v
        + 3 * Cv3 * m_pi_v**2
        + 3 * Cpq3 * m_pi_pq**2 * dm_pi_pq_d_m_pi_v
    )

    dM_d_m_pi_sea = (
        2 * p["Cpq2"] * m_pi_pq * dm_pi_pq_d_m_pi_sea
        + 3 * Cpq3 * m_pi_pq**2 * dm_pi_pq_d_m_pi_sea
    )

    return dM_d_m_pi_v, dM_d_m_pi_sea


Hml_sea_bt = np.zeros(n_bootstrap)
Hml_v_bt = np.zeros(n_bootstrap)
for i in range(n_bootstrap):
    p = {key: LEC_mass[key][i] for key in LEC_mass}
    fpi = Fpi_article(m_pi_phys**2, 0)
    if baryon != 'OMEGA':
        if baryon == "PROTON":
            Cv3 = (
                -(p["gA"] ** 2 - 4 * p["gA"] * p["g1"] - 5 * p["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
            Cpq3 = (
                -(8 * p["gA"] ** 2 + 4 * p["gA"] * p["g1"] + 5 * p["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
        elif baryon == 'DELTA':
            Cv3 = p["Cv3"]
            Cpq3 = p["Cpq3"]
        else:
            Cv3 = 0
            Cpq3 = 0

        dM_d_m_pi_v, dM_d_m_pi_sea = derivatives_Mcorr(
            m_pi_phys, m_pi_phys, m_pi_phys, p, Cv3, Cpq3
        )
        Hml_sea_bt[i] = m_pi_phys / 2 * dM_d_m_pi_sea
        Hml_v_bt[i] = m_pi_phys / 2 * dM_d_m_pi_v
    else:
        Hml_v_bt[i] = 0
        Hml_sea_bt[i] = m_pi_phys**2 * p['Cs2']

print(gv.gvar(np.mean(Hml_v_bt), np.std(Hml_v_bt)))
print(gv.gvar(np.mean(Hml_sea_bt), np.std(Hml_sea_bt)))

file_path = f"Hml_v_sea_decomposition_addCa_{not CasQ}.json"
if os.path.exists(file_path):
    # 如果文件存在，读取现有内容
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
else:
    # 如果文件不存在，初始化一个空字典
    data = {}

data[baryon] = {}
# 更新字典
data[baryon]["Hml_v"] = str(gv.gvar(np.mean(Hml_v_bt), np.std(Hml_v_bt)))
data[baryon]["Hml_sea"] = str(gv.gvar(np.mean(Hml_sea_bt), np.std(Hml_sea_bt)))

# 将更新后的字典保存回json文件
# with open(file_path, "w") as json_file:
#     json.dump(data, json_file, indent=4)

data_Hml_v = {
    "fit_fcn": [np.mean(Hml_v_bt)],
    "fit_fcn_err": [np.std(Hml_v_bt)],
    "fit_chi2": fit_chi2,
}

data_Hml_s = {
    "fit_fcn": [np.mean(Hml_sea_bt)],
    "fit_fcn_err": [np.std(Hml_sea_bt)],
    "fit_chi2": fit_chi2,
}

outdir = Path(lqcd_data_path) / "pydata_global_fit"
outdir.mkdir(parents=True, exist_ok=True)  # 自动创建目录
outdir2 = Path(lqcd_data_path) / "figures"
outdir2.mkdir(parents=True, exist_ok=True)  # 自动创建目录

outfile_v = outdir / f"{baryon}_Hmlv_data_addCa_{not CasQ}_{error_source}.npy"
outfile_s = outdir / f"{baryon}_Hmls_data_addCa_{not CasQ}_{error_source}.npy"

np.save(outfile_v, data_Hml_v)
np.save(outfile_s, data_Hml_s)

# 对于LEC中的每一个参数
for param, values in LEC_mass.items():
    # 计算均值和标准差
    mean = np.mean(values)
    std_dev = np.std(values)

    # 将均值和标准差转换为gvar的形式
    LEC_gvar[param] = gv.gvar(mean, std_dev)

# print(LEC)
# exit()

# 打印结果
print("LEC using bootstrap:")
for param, gvar in LEC_gvar.items():
    if param != 'chi2':
        print(f"{param}: {gvar}")

# print(LEC_gvar[''])
# pp = LEC_gvar

a = gv.mean(alttc_gv_new).repeat(3)
L = np.array(ns).repeat(3) * a


def compute_pq_corr(pp, m_pi_sea, m_pi_v, metas_sea, alttc_loc=0):
    fpi = np.array([Fpi_article(m_pi_v[i] ** 2, 0) for i in range(m_pi_v.shape[0])])

    if baryon != 'OMEGA':
        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        if baryon == "PROTON":
            Cpq3 = (
                -(8 * pp["gA"] ** 2 + 4 * pp["gA"] * pp["g1"] + 5 * pp["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
        elif baryon == 'DELTA':
            Cpq3 = pp["Cpq3"]
        else:
            Cpq3 = 0
        term1 = (
            -pp["Cpq2"] * m_pi_pq**2
            + pp["Cpq2"] * m_pi_v**2
            - Cpq3 * m_pi_pq**3
            + Cpq3 * m_pi_v**3
        ) * ((1 + pp["Cas"] * a**2) if CasQ else 1)
    else:
        term1 = 0
    # common_factor = np.pi / (3 * (4 * np.pi * fpi) ** 2)

    # term3 = -pp["C4"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L)
    term3 = -pp["C4"] * (m_pi_v**2 / L) * np.exp(-m_pi_v * L)

    term4 = (
        -pp["C5"]
        * (metas_sea**2 - etas_phys**2)
        * ((1 + pp["Cas"] * a**2) if CasQ else 1)
    )

    if additive_Ca:
        return term1 + term3 + term4
    else:
        # return (term1 + term2)*(1+pp["Ca2"]*alttc_loc**2) + term4*(1+pp["Ca2s"]*alttc_loc**2) + term3
        return (term1 + term3 + term4) * (1 + pp["Ca2"] * alttc_loc**2)


def compute_phys_corr(pp, m_pi_sea, m_pi_v, metas_sea, alttc_loc=0):
    fpi = np.array([Fpi_article(m_pi_v[i] ** 2, 0) for i in range(m_pi_v.shape[0])])
    # print(fpi)
    if baryon != 'OMEGA':
        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        if baryon == "PROTON":
            Cpq3 = (
                -(8 * pp["gA"] ** 2 + 4 * pp["gA"] * pp["g1"] + 5 * pp["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
            Cv3 = (
                -(pp["gA"] ** 2 - 4 * pp["gA"] * pp["g1"] - 5 * pp["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
        elif baryon == 'DELTA':
            Cpq3 = pp["Cpq3"]
            Cv3 = pp["Cv3"]
        else:
            Cpq3 = 0
            Cv3 = 0
        term1 = (
            -pp["Cv2"] * m_pi_v**2
            + pp["Cv2"] * m_pi_phys**2
            - pp["Cpq2"] * m_pi_pq**2
            + pp["Cpq2"] * m_pi_phys**2
            - Cv3 * m_pi_v**3
            + Cv3 * m_pi_phys**3
            - Cpq3 * m_pi_pq**3
            + Cpq3 * m_pi_phys**3
        ) * ((1 + pp["Cas"] * a**2) if CasQ else 1)
    else:
        term1= -pp["Cs2"] * m_pi_sea**2 + pp["Cs2"] * m_pi_phys**2

    # term2 = -pp["C4"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L)
    term2 = -pp["C4"] * (m_pi_v**2 / L) * np.exp(-m_pi_v * L)

    term3 = (
        -pp["C5"]
        * (metas_sea**2 - etas_phys**2)
        * ((1 + pp["Cas"] * a**2) if CasQ else 1)
    )

    # term5 = -(pp["C1"]+pp['C2']) * (m_pi_v**2 - m_pi_phys**2)
    # term5 = -(pp["C2"] + pp["C1"]) * (m_pi_v**2 - m_pi_phys**2)

    if additive_Ca:
        return term1 + term2 + term3
    else:
        # return (term1 + term2  + term5)*(1+pp["Ca2"]*alttc_loc**2) + term4*(1+pp["Ca2s"]*alttc_loc**2) + term3
        return (term1 + term2 + term3) * (1 + pp["Ca2"] * alttc_loc**2)


mp_v_bt_etas069_corr = np.array(
    [
        compute_phys_corr(
            {key: LEC_mass[key][i] for key in LEC_mass},
            m_pi_sea_bt[:, i],
            m_pi_v_bt[:, i],
            metas_sea_bt[:, i],
            alttc_samples[:, i],
        )
        for i in range(mp_v_bt_etas069.shape[1])
    ]
).T

mp_v_bt_etas069_pq_corr = np.array(
    [
        compute_pq_corr(
            {key: LEC_mass[key][i] for key in LEC_mass},
            m_pi_sea_bt[:, i],
            m_pi_v_bt[:, i],
            metas_sea_bt[:, i],
            alttc_samples[:, i],
        )
        for i in range(mp_v_bt_etas069.shape[1])
    ]
).T

mp_v_bt_etas069_phys = mp_v_bt_etas069 + mp_v_bt_etas069_corr
mp_v_bt_etas069_uni = mp_v_bt_etas069 + mp_v_bt_etas069_pq_corr
# print('aaaaaaaa',mp_v_bt_etas069[:,0],mp_v_bt_etas069_corr[:,0],mp_v_bt_etas069_phys[:,0])

# print(combined_data_corr)
combined_data_physical_cntrl = np.mean(mp_v_bt_etas069_phys, axis=1)
combined_data_physical_err = np.std(mp_v_bt_etas069_phys, axis=1)
# Plot
combined_data_cntrl = np.mean(mp_v_bt_etas069_uni, axis=1)
combined_data_err = np.std(mp_v_bt_etas069_uni, axis=1)
combined_data_uncorrected_cntrl = np.mean(mp_v_bt_etas069, axis=1)
combined_data_uncorrected_err = np.std(mp_v_bt_etas069, axis=1)

# combined_data_cntrl = combined_data_physical_cntrl
if baryon != 'OMEGA':
    m_pi_v_mean_square = np.mean(m_pi_v_bt**2, axis=1)
else:
    m_pi_v_mean_square = np.mean(m_pi_sea_bt**2, axis=1)


# 开始绘图
# plt.figure(figsize=(15, 10), dpi=100)
plt.figure(figsize=(5 * 0.9, 4 * 0.9), dpi=200)
ax = plt.gca()

errorbar_kwargs = {
    "linewidth": 0.8,
    "elinewidth": 0.8,
    "capthick": 1,
    "capsize": 2,
    "mew": 0.8,
    "linestyle": "none",
    "fillstyle": "none",
    "markersize": 5,
}

colors_info = [
    "r",
    "orange",
    "b",
    "purple",
    "g",
]
colors = [colors_info[alttc_index[idx]] for idx in selected_ensemble_indices]
markers = ["^", "<", "v", ">", "x", "s", "d", "o", "8", "p", "*", "h", "H", ".", "+"]
# for i in range(3*13):
# alttc_index = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 1, 3, 5, 2]
alttc_index_selected = alttc_index[selected_ensemble_indices]
# for i in range(len(selected_ensemble_indices)*3):
for j in range(6):
    indices_of_alttc_j = np.where(alttc_index_selected == j)[0]

    for ii in indices_of_alttc_j:
        for i in range(ii * 3, ii * 3 + 3):
            if i % 3 == 0:
                # i 能被3整除时，添加label
                ax.errorbar(
                    m_pi_v_mean_square[i],
                    combined_data_cntrl[i],
                    yerr=combined_data_err[i],
                    color=colors[i // 3],
                    label=name[selected_ensemble_indices[i // 3]],
                    marker=markers[i // 3],
                    **errorbar_kwargs,
                )
            else:
                # i 不能被3整除时，不添加label
                ax.errorbar(
                    m_pi_v_mean_square[i],
                    combined_data_cntrl[i],
                    yerr=combined_data_err[i],
                    color=colors[i // 3],
                    marker=markers[i // 3],
                    **errorbar_kwargs,
                )
            # ax.errorbar(
            #     m_pi_v_mean_square[i]+0.001,
            #     combined_data_uncorrected_cntrl[i],
            #     yerr=combined_data_uncorrected_err[i],
            #     color='black',
            #     marker=markers[i // 3],
            #     **errorbar_kwargs,
            # )


fpi_mean = np.mean(fpi_v_bt, axis=1)


def compute_M_proton(m_pi_square, a, LEC, fpi):
    m_pi = np.sqrt(m_pi_square)
    fpi = Fpi_article(m_pi_square, a)
    # fpi=fpi_phys
    M_0 = LEC["M0"]
    if baryon != 'OMEGA':
        if baryon == "PROTON":
            Cv3 = (
                -(LEC["gA"] ** 2 - 4 * LEC["gA"] * LEC["g1"] - 5 * LEC["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )
            Cpq3 = (
                -(8 * LEC["gA"] ** 2 + 4 * LEC["gA"] * LEC["g1"] + 5 * LEC["g1"] ** 2)
                * np.pi
                / (3 * (4 * np.pi * fpi) ** 2)
            )

        elif baryon == 'DELTA':
            Cv3 = LEC["Cv3"]
            Cpq3 = LEC["Cpq3"]
        else:
            Cv3 = 0
            Cpq3 = 0
        M_corr = (
            LEC["Cv2"] * (m_pi**2 - m_pi_phys**2)
            + LEC["Cpq2"] * (m_pi**2 - m_pi_phys**2)
            + Cv3 * (m_pi**3 - m_pi_phys**3)
            + Cpq3 * (m_pi**3 - m_pi_phys**3)
        ) * ((1 + LEC["Cas"] * a**2) if CasQ else 1)
    else:
        M_corr = ( LEC["Cs2"] * (m_pi**2 - m_pi_phys**2)) * ((1 + LEC["Cas"] * a**2) if CasQ else 1)

    if Ca4Q:
        M_corr += LEC["Ca2"] * a**2 + LEC["Ca4"] * a**4
    else:
        M_corr += LEC["Ca2"] * a**2
    return M_0 + M_corr


# 使用样本的平均值和标准差来计算上下限
def compute_limits(samples):
    mean = np.mean(samples, axis=1)
    err = np.std(samples, axis=1)
    return mean - err, mean + err


####plot for article######
fit_a_2 = np.linspace(0, 0.013, 400)
num_samples = LEC_mass["M0"].shape[0]

M_extra_plot_samples = -np.ones((fit_a_2.size, num_samples))
for idx, a2 in enumerate(fit_a_2):
    M_extra_plot_samples[idx] = [
        compute_M_proton(
            m_pi_phys**2,
            np.sqrt(a2),
            {key: LEC_mass[key][i] for key in LEC_mass},
            0.122,
        )
        for i in range(num_samples)
    ]
    # M_samples[idx,:]=np.zeros(4000)
M_extra_plot_mean = np.mean(M_extra_plot_samples, axis=1)
M_extra_plot_err = np.std(M_extra_plot_samples, axis=1, ddof=1)
# 将数据保存为一个字典
data = {
    "combined_data_Ds_phy": combined_data_physical_cntrl[
        ml_indices[3 * np.arange(len(selected_ensemble_indices))]
    ],
    "combined_data_err": combined_data_physical_err[
        ml_indices[3 * np.arange(len(selected_ensemble_indices))]
    ],
    "labels": name,
    "colors": colors,
    "markers": markers,
    "a_2": gv.mean(alttc_gv_new) ** 2,
    "fit_a_2": fit_a_2,
    "fit_chi2": fit_chi2,
    "fit_fcn": M_extra_plot_mean,
    "fit_fcn_err": M_extra_plot_err,
}

# 保存数据到.npy文件
np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_mass_extra_plot_data_addCa_{not CasQ}_{error_source}.npy", data)
####plot for article######


# m_pi_square = np.linspace(0.0145, 0.15, 400)
m_pi_square = np.linspace(0.0001, 0.15, 100)
a_values = [0.1053, 0.0887, 0.0775, 0.0682, 0.0519]
colorss = ["red", "orange", "blue", "purple", "green"]

results = {}
for a, color in zip(a_values, colorss):
    # 根据LEC['M0']的长度来确定bootstrap样本的数量
    num_samples = LEC_mass["M0"].shape[0]
    # print(num_samples)
    # print(fpi)
    # print(fpi.shape)

    # 创建一个空的二维数组来存储结果
    M_samples = np.empty((m_pi_square.size, num_samples))

    # 对于每个m_pi_square值，计算所有的bootstrap样本
    for idx, m in enumerate(m_pi_square):
        M_samples[idx] = [
            compute_M_proton(
                m, a, {key: LEC_mass[key][i] for key in LEC_mass}, fpi_phys
            )
            # compute_M_proton(m, a, {key: LEC[key][i] for key in LEC}, fpi[i])
            for i in range(num_samples)
        ]

    # 现在你可以沿着axis=1取均值
    M_mean = np.mean(M_samples, axis=1)
    M_std = np.std(M_samples, axis=1)

    M_lower, M_upper = compute_limits(M_samples)

    # plt.fill_between(m_pi_square, M_lower, M_upper, color=color, alpha=0.2)
    plt.plot(m_pi_square, M_mean, color=color, linestyle="-")
    # 绘制带状区域
    # plt.fill_between(m_pi_square, M_lower, M_upper, color=color, alpha=0.4, label=f"a = {a}fm")
    plt.fill_between(
        m_pi_square,
        M_lower,
        M_upper,
        color=color,
        alpha=0.18,
        label=f"a = {a}fm",
        edgecolor="none",
    )
    # 绘制线条
    (line,) = plt.plot(m_pi_square, M_mean, color=color)
    # 将结果转换为 gvar 格式
    mpi_gvars = {
        str(m): gv.gvar(mean, std) for m, mean, std in zip(m_pi_square, M_mean, M_std)
    }

    # 存储到字典中
    results[f"a={a}"] = mpi_gvars


# 格距为0时的处理
# M_samples_zero = np.array(
#     [
#         compute_M_proton(m, 0, {key: LEC_mass[key][i] for key in LEC_mass}, fpi_phys)
#         for m in m_pi_square
#         for i in range(LEC_mass["M0"].shape[0])
#     ]
# )

# 创建一个空的二维数组来存储结果
M_samples_zero = np.empty((m_pi_square.size, num_samples))

# 对于每个m_pi_square值，计算所有的bootstrap样本（对于a=0）
for idx, m in enumerate(m_pi_square):
    M_samples_zero[idx] = [
        compute_M_proton(m, 0, {key: LEC_mass[key][i] for key in LEC_mass}, fpi_phys)
        for i in range(num_samples)
    ]


# 现在你可以沿着axis=1取均值
M_mean_zero = np.mean(M_samples_zero, axis=1)
M_std_zero = np.std(M_samples_zero, axis=1)
M_lower_zero, M_upper_zero = compute_limits(M_samples_zero)
print(M_samples_zero.shape,M_mean_zero,M_std_zero)
# np.save('./Nucleon_mass_continuum_bootstrap_sample.npy',M_samples_zero)
# np.save('./Pion_mass_square.npy',m_pi_square)
# exit()

mpi_gvars = {
    str(m): gv.gvar(mean, std)
    for m, mean, std in zip(m_pi_square, M_mean_zero, M_std_zero)
}

# 存储到字典中
results[f"a={0}"] = mpi_gvars

# 将结果写入 JSON 文件
# with open("results.json", "w") as f:
#     json.dump(results, f, default=lambda obj: str(obj), indent=4)

# fit the sigma term using a linear fit
# 找到 m_pi^2 = 0.02 附近的点进行线性拟合
# 这里我们尽量选择最接近 0.02 的10个点
target = 0.135**2
num_points = 10  # 希望用于拟合的点的数量
half_points = num_points // 2

# 根据距离排序，获取最接近的索引
sorted_indices = np.argsort(np.abs(m_pi_square - target))

# 如果可用点数少于所需的10个点，则使用所有点
if len(sorted_indices) < num_points:
    selected_indices = sorted_indices
else:
    # 找到最接近的点作为中心
    center_idx = sorted_indices[0]
    selected_indices = sorted_indices[: max(half_points, center_idx + 1)]  # 确保中心点包含在内
    selected_indices = np.unique(
        np.concatenate([selected_indices, sorted_indices[:num_points]])
    )  # 扩展并去重

mass_data = {
    "fit_fcn": [np.mean(M_samples_zero[center_idx])],
    "fit_fcn_err": [np.std(M_samples_zero[center_idx])],
    "labels": name[selected_ensemble_indices],
    "colors": colors,
    "markers": markers,
    "fit_chi2": fit_chi2,
    # "a_2": a_2_list,
    # "fit_a_2": fit_a_2,
    # "fit_fcn": M_mean,
    # "fit_fcn_err": M_err,
}

# 保存数据到.npy文件
np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_mass_data_addCa_{not CasQ}_{error_source}.npy", mass_data)
# 用于存储每个样本的斜率
slopes = []

# 对每个bootstrap样本进行线性拟合
for sample in range(num_samples):
    # 收集每个样本在指定 m_pi_square 点的值
    y_samples = [M_samples_zero[idx][sample] for idx in selected_indices]
    x_samples = m_pi_square[selected_indices]

    # 进行线性拟合并获取斜率
    slope, intercept = np.polyfit(x_samples, y_samples, 1)
    slopes.append(slope)

# 计算斜率的平均值和标准差
slope_mean = np.mean(slopes)
slope_std = np.std(slopes)
# Hml = np.array(slopes) * 0.135**2
Hml=Hml_sea_bt+Hml_v_bt

print("Estimated slop: ", gv.gvar(slope_mean, slope_std))
print(
    "Estimated sigma term: ", gv.gvar(slope_mean, slope_std) * 0.135**2 * 1000, "MeV"
)
data = {
    "fit_fcn": [slope_mean * 0.135**2],
    "fit_fcn_err": [slope_std * 0.135**2],
    "labels": name[selected_ensemble_indices],
    "colors": colors,
    "markers": markers,
    "fit_chi2": fit_chi2,
    # "a_2": a_2_list,
    # "fit_a_2": fit_a_2,
    # "fit_fcn": M_mean,
    # "fit_fcn_err": M_err,
}

# 保存数据到.npy文件
np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_Hml_data_addCa_{not CasQ}_{error_source}.npy", data)

# ========================== yang comparision =====================
yang_comparision = False
if yang_comparision == True:
    targets = np.linspace(0, 0.15, 40)  # 定义目标范围

    num_points = 10  # 希望用于拟合的点的数量
    half_points = num_points // 2

    # 用于存储每个 target 的结果
    slope_results = []

    for target in targets:
        # 根据距离排序，获取最接近的索引
        sorted_indices = np.argsort(np.abs(m_pi_square - target))

        # 如果可用点数少于所需的10个点，则使用所有点
        if len(sorted_indices) < num_points:
            selected_indices = sorted_indices
        else:
            # 找到最接近的点作为中心
            center_idx = sorted_indices[0]
            selected_indices = sorted_indices[
                : max(half_points, center_idx + 1)
            ]  # 确保中心点包含在内
            selected_indices = np.unique(
                np.concatenate([selected_indices, sorted_indices[:num_points]])
            )  # 扩展并去重

        # 用于存储每个样本的斜率
        slopes = []

        # 对每个bootstrap样本进行线性拟合
        for sample in range(num_samples):
            # 收集每个样本在指定 m_pi_square 点的值
            y_samples = [M_samples_zero[idx][sample] for idx in selected_indices]
            x_samples = m_pi_square[selected_indices]

            # 进行线性拟合并获取斜率
            slope, intercept = np.polyfit(x_samples, y_samples, 1)
            slopes.append(slope)

        # 计算斜率的平均值和标准差
        slope_mean = np.mean(slopes)
        slope_std = np.std(slopes)

        # 将结果存储到列表中
        slope_results.append(
            {
                "target": target,
                "slope_mean": slope_mean,
                "slope_std": slope_std,
                # 'sigma_term': gv.gvar(slope_mean, slope_std) * target * 1000  # MeV
                "sigma_term": slope_mean * target * 1000,  # MeV
            }
        )

        print(f"Target: {target}")
        print("Estimated slope: ", gv.gvar(slope_mean, slope_std))
        print(
            "Estimated sigma term: ",
            gv.gvar(slope_mean, slope_std) * target * 1000,
            "MeV",
        )

    np.save("{lqcd_data_path}/Hm_data_addCa_{not CasQ}_{error_source}.npy", slope_results)
    # 提取 target, sigma_term, 和标准差
    targets = [result["target"] for result in slope_results]
    sigma_terms = [result["sigma_term"] for result in slope_results]
    sigma_term_errors = [
        result["slope_std"] * result["target"] * 1000 for result in slope_results
    ]

    # # 创建图形
    # plt.figure(figsize=(10, 6), dpi=100)
    # plt.plot(targets, sigma_terms, label='Sigma Term', color='blue')
    # plt.fill_between(targets,
    #                  np.array(sigma_terms) - np.array(sigma_term_errors),
    #                  np.array(sigma_terms) + np.array(sigma_term_errors),
    #                  color='blue', alpha=0.3)

    # targets = np.array(targets)
    # sigma_terms = np.array(sigma_terms)
    # sigma_term_errors = np.array(sigma_term_errors)

    # 保存数据到文件
    np.savez(
        "plot_data.npz",
        targets=targets,
        sigma_terms=sigma_terms,
        sigma_term_errors=sigma_term_errors,
    )

    print("Data saved to 'plot_data.npz'")
# ========================== yang comparision =====================

# plt.xlabel(r'$m^2_{\pi} (\mathrm{GeV}^2)$')
# plt.ylabel('Estimated $\sigma$ Term (MeV)')
# # plt.title('Sigma Term vs Target with Error Bars')
# plt.legend()
# plt.grid(True)

# # 保存图像
# plt.savefig('./sigma_term_vs_mpi.pdf')
# plt.show()


plt.plot(m_pi_square, M_mean_zero, "k--")
plt.fill_between(
    m_pi_square,
    M_lower_zero,
    M_upper_zero,
    color="grey",
    alpha=0.75,
    label="continuum",
    edgecolor="none",
    # zorder=10,
)

m_pi_value = 0.135
# BMW band
if baryon == "PROTON":
    # Load the data from CSV files
    lower_data = pd.read_csv(
        "global_fit/data-BMW/lower.csv", header=None, names=["m_pi_squared", "M_rho"]
    )
    upper_data = pd.read_csv(
        "global_fit/data-BMW/upper.csv", header=None, names=["m_pi_squared", "M_rho"]
    )

    # Fit polynomials to the data from each CSV file
    lower_fit_coeff = np.polyfit(lower_data["m_pi_squared"], lower_data["M_rho"], deg=4)
    upper_fit_coeff = np.polyfit(upper_data["m_pi_squared"], upper_data["M_rho"], deg=4)

    # Generate x values for the fit curves over the range of m_pi_squared values
    fit_x = np.linspace(
        min(lower_data["m_pi_squared"].min(), upper_data["m_pi_squared"].min()),
        max(lower_data["m_pi_squared"].max(), upper_data["m_pi_squared"].max()),
        300,
    )
    lower_fit_y = np.polyval(lower_fit_coeff, fit_x)
    upper_fit_y = np.polyval(upper_fit_coeff, fit_x)
    plt.fill_between(
        fit_x,
        lower_fit_y,
        upper_fit_y,
        hatch="////",
        # hatch="*",
        edgecolor="magenta",
        facecolor="none",
        label="Borsanyi 20",
    )

# BMW band

# yang band
if baryon == "PROTON":
    # Load the data from CSV files
    lower_data = pd.read_csv(
        "global_fit/data-yang/lower.csv", header=None, names=["m_pi_squared", "M_rho"]
    )
    upper_data = pd.read_csv(
        "global_fit/data-yang/upper.csv", header=None, names=["m_pi_squared", "M_rho"]
    )

    # Fit polynomials to the data from each CSV file
    lower_fit_coeff = np.polyfit(lower_data["m_pi_squared"], lower_data["M_rho"], deg=4)
    upper_fit_coeff = np.polyfit(upper_data["m_pi_squared"], upper_data["M_rho"], deg=4)

    # Generate x values for the fit curves over the range of m_pi_squared values
    fit_x = np.linspace(
        min(lower_data["m_pi_squared"].min(), upper_data["m_pi_squared"].min()),
        max(lower_data["m_pi_squared"].max(), upper_data["m_pi_squared"].max()),
        300,
    )
    lower_fit_y = np.polyval(lower_fit_coeff, fit_x)
    upper_fit_y = np.polyval(upper_fit_coeff, fit_x)
    plt.fill_between(
        fit_x,
        # lower_fit_y-0.02,
        # upper_fit_y-0.02,
        lower_fit_y * 0.98,
        upper_fit_y * 0.98,
        # color="lightblue",
        label="$\chi$QCD 18, rescaled",
        # zorder=0,
        hatch="\\\\\\",
        edgecolor="cyan",
        facecolor="none",
    )

# yang band


# 在你的绘图代码之前或之后加入以下代码
m_pi_squared_value = m_pi_value**2

# 添加表示M_p = 0.931的横线
M_p_value = float(plot_configurations[baryon]["mass"]) / 1000
# 计算交点并在此位置放置星星标记
intersection_x = m_pi_squared_value
intersection_y = M_p_value
xlim = plt.xlim()
ylim = plt.ylim()

if xlim[0] <= intersection_x <= xlim[1] and ylim[0] <= intersection_y <= ylim[1]:
    plt.scatter(
        intersection_x, intersection_y, color="red", marker="*", zorder=5, s=50
    )  # s设置星星的大小，zorder确保星星在其他元素上面
plt.axvline(
    x=m_pi_squared_value,
    color="grey",
    linestyle="--",
    # linewidth=1.5,
    label="$m_{{\pi}}^2 = (0.135\mathrm{GeV})^2$",
)

plt.axhline(
    y=M_p_value,
    color="grey",
    linestyle="-.",
    # linewidth=1.5,
    label=f"{plot_configurations[baryon]['label_x'].replace('(GeV)','')} = {M_p_value:.3f}"
    + "$\mathrm{GeV}$",
)

# 设置图形的标题，坐标轴标签等
# plt.title('Proton Mass vs Pion Mass squared', fontsize=35, pad=20)
plt.xlabel("$m_{\pi}^2(\mathrm{GeV^2})$")
y_label = plot_configurations[baryon]["label_x"].replace("GeV", "\mathrm{GeV}")
plt.ylabel(f"{y_label}")

# ax.tick_params(axis="both", colors="black")
# ax.tick_params(which="major", direction="in")
# ax.minorticks_on()
# ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)

# for spine in ax.spines.values():
#     spine.set_color("black")
#     spine.set_linewidth(3)

# ax.legend(loc="best", frameon=False, fontsize=25, ncol=2)
# ax.legend(loc="upper left", frameon=False, fontsize=25, ncol=2)
# legend1=ax.legend(loc="upper left", labels=name,frameon=False, ncol=2)
handles, labels = ax.get_legend_handles_labels()
num_right = len(labels) - len(selected_ensemble_indices)
legend1 = ax.legend(
    handles[num_right:], labels[num_right:], loc="upper left", frameon=False, ncol=3
)
ax.add_artist(legend1)
# print(handles)
ax.legend(
    handles[:num_right], labels[:num_right], loc="lower right", frameon=False, ncol=2
)

plt.tight_layout()
if baryon == 'PROTON':
    plt.savefig(f"{lqcd_data_path}/final_results/{baryon}_mass_addCa_{not CasQ}_{error_source}.pdf")
else:
    plt.savefig(f"{lqcd_data_path}/figures/{baryon}_mass_addCa_{not CasQ}_{error_source}.pdf")
    # plt.savefig(f"{lqcd_data_path}/figures/{baryon}_{data_name}_mass_addCa_{not CasQ}.pdf")
# plt.show()


def Hm_data_fit_and_plot(data_name, combined_data, ylabel, Ca4Q):
    # Hms+=LEC_mass['C1']*m_pi_phys**2+LEC_mass['C2']*etas_phys**2
    # print((LEC_mass['C1']*m_pi_phys**2+LEC_mass['C2']*etas_phys**2))
    # print(Hm.shape)
    # np.savetxt(f'./{baryon}_Hms_sea.txt',np.array([np.mean(LEC_mass['C2']*etas_phys**2),np.std(LEC_mass['C2']*etas_phys**2)]))
    # print(Hm[:,0])
    # exit()
    # combined_data = mc_v_etas069
    combined_data_cntrl = np.mean(combined_data, axis=1)
    combined_data_err = np.std(combined_data, axis=1, ddof=1)
    combined_data_cov = np.cov(combined_data)

    if cov_enabled == True:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
        # data_dctnry = {"combined": gv.gvar(combined_data[:,0], combined_data_cov)}
    else:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
    N_f = 2
    # print(data_dctnry)
    # exit()

    ms_phys = 0.0978
    ml_phys = 0.003571
    mc_phys = 1.085
    # m_pi_phys = 0.13498  # Adjust this value to the correct one
    m_pi_phys = 0.135  # Adjust this value to the correct one
    # D_S_phys=1968.34/1000
    # D_S_phys = 1962.9 / 1000

    # print(mc_v_etas690_Ds_phys[np.arange(13)*3,0])
    # exit( )
    def vol_a_mdls(idct, ppp):
        mdls = {}
        i = idct["i"]
        # print(i)
        # i=100
        m_pi_sea = m_pi_sea_bt[np.arange(len(selected_ensemble_indices)) * 3, i]
        metas_sea = metas_sea_bt[np.arange(len(selected_ensemble_indices)) * 3, i]
        ms_sea = ms_sea_bt[np.arange(len(selected_ensemble_indices)) * 3, i]
        ml_sea = ml_sea_bt[np.arange(len(selected_ensemble_indices)) * 3, i]

        # fpi=np.mean(fpi_v_bt,axis=1)
        # a = gv.mean(alttc_gv_new)
        a = alttc_samples[np.arange(len(selected_ensemble_indices)) * 3, i]
        L = np.array(ns) * a
        # pc1_arr=np.array([p['C1(0.105)'],p['C1(0.077)'],p['C1(0.05)'],p['C1(0.088)'],p['C1(0.068)']])
        # pc1=pc1_arr[alttc_index].repeat(3)
        M_baryon = (
            ppp["M0"]
            # + ppp["C1"] * (Ds_v_etas069[:, i] - D_S_phys)
            # + pc1 * (mc_v_etas069[:, i] - mc_phys)
            # + ppp["C1"] * (mc_v_etas690_Ds_phys[np.arange(13)*3, i] - mc_phys)
            # * (1 + ppp["C1a2"] * a**2)
            # + ppp["C2"] * (ms_sea - ms_phys)
            # + ppp["C3"] * (ml_sea - ml_phys)
            + ppp["C1"] * (m_pi_sea**2 - m_pi_phys**2)
            + ppp["C2"] * (metas_sea**2 - etas_phys**2)
            # + ppp["C6"] * (mc_v_mean - mc_phys_mean)
            # + ppp["C4"] * (m_jpsi_mean - m_jpsi_phys)
            + ppp["CL"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L)
            + ppp["Ca2"] * a**2
        )
        if Ca4Q:
            M_baryon += ppp["Ca4"] * a**4
        # M_baryon = (
        #     p["M0"]
        #     + p["C1"] * (ms_v - ms_phys)
        #     + p["C2"] * (ms_sea - ms_phys)
        #     + p["C3"] * (ml_sea - ml_phys)
        #     + p["C4"] * (ml_sea / L) * np.exp(-m_pi_sea * L))*(1+ p["C5"] * a**2)
        mdls["combined"] = np.array(M_baryon)
        # print(p["C4"] * (ml_sea / L) * np.exp(-m_pi_sea * L))
        return mdls

    # 初始参数范围
    param_ranges = {
        "M0": (0, 2),
        "C1": (-3, 3),
        # "C1a2": (-3, 3),
        # "C1(0.105)": (-3, 3),
        # "C1(0.077)": (-3, 3),
        # "C1(0.05)": (-3, 3),
        # "C1(0.088)": (-3, 3),
        # "C1(0.068)": (-3, 3),
        "C2": (-3, 3),
        # "C3": (-3, 3),
        "CL": (-3, 3),
        "Ca2": (-3, 3),
    }
    if Ca4Q:
        param_ranges["Ca4"] = (-3, 3)

    def generate_trial_params():
        return {
            param: np.random.uniform(low, high)
            for param, (low, high) in param_ranges.items()
        }

    n_trials = 10000
    # best_params = None
    best_params = generate_trial_params()
    best_chi2_dof = float("inf")
    best_Q = 0

    def fit_with_initial_values(initial_params):
        if cov_enabled == True:
            data_dctnry_loc = {
                "combined": gv.gvar(combined_data_cntrl, combined_data_cov)
            }
        else:
            data_dctnry_loc = {
                "combined": gv.gvar(combined_data_cntrl, combined_data_err)
            }
        fit_vol = lsqfit.nonlinear_fit(
            data=({"i": 0}, data_dctnry_loc),
            svdcut=1e-12,
            fcn=vol_a_mdls,
            p0=initial_params,
            debug=True,
        )
        return (
            {key: gv.mean(value) for key, value in (fit_vol.p).items()},
            fit_vol.chi2 / fit_vol.dof,
            fit_vol.Q,
        )

    # 并行地尝试不同的参数初值，并进行拟合
    results = Parallel(n_jobs=-1)(
        delayed(fit_with_initial_values)(generate_trial_params())
        for _ in range(n_trials)
    )

    # 从结果中找到最佳拟合
    for params, chi2_dof, Q in results:
        if chi2_dof < best_chi2_dof and Q > best_Q:
            best_chi2_dof = chi2_dof
            best_Q = Q
            best_params = params

    # print("Best Initial Parameters:", best_params)
    # print("Best chi2/dof:", best_chi2_dof)
    # print("Best Q:", best_Q)
    # best_params['M0']=0
    # best_params['C1']=0
    # best_params['C2']=0
    # best_params['C3']=0
    # best_params['C4']=0
    # best_params['C5']=0
    # best_params['C6']=0
    fit = lsqfit.nonlinear_fit(
        data=({"i": 0}, data_dctnry),
        svdcut=1e-12,
        fcn=vol_a_mdls,
        p0=best_params,
        debug=True,
    )  # ? excute the fit
    fit_chi2 = fit.chi2 / fit.dof

    # print(data_name)
    # print(fit.format(True))  # ? print out fit results
    # print(fit_vol.format(True))  # ? print out fit results
    # print(fit.p)
    # exit()
    # exit()

    def M(ms_v, m_pi_sea, ms_sea, a, l, params):
        """
        calculate the mass using the provided equation.

        parameters:
            m_pi_v: float
                pion mass.
            m_pi_sea: float
                sea pion mass.
            a: float
                lattice spacing.
            l: float
                lattice size.
            params: dict
                parameters obtained from the fit.

        returns:
            m: float
                calculated mass.
        """
        m_0 = params["M0"]
        c1 = params["C1"]
        c2 = params["C2"]
        # c3 = params["C3"]
        # c4 = params["c4"]
        # c5 = params["c5"]
        # c6 = params["c6"]
        # f_pi = 0.130  # you can adjust this value if it's different

        ta = c1 * m_pi_phys**2 + c2 * etas_phys**2
        # term2 = 1 + c5 * a**2
        # term3 = c4 * (ml_sea / l) * np.exp(-m_pi_sea * L)

        return ta

    # Assuming m_pi_phys is the physical pion mass
    # m_pi_phys = 0.135  # Adjust this value to the correct one
    # ms_phys = 0.0922
    # ms_phys = 0.1  # Adjust this value to the correct one

    # Calculate M at the physical point
    # M_phys = M(ms_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p)
    # print(M_phys)
    # print(f"M at the physical point is: {M_phys} GeV")

    def single_fit(
        i,
        cov_enabled,
        combined_data,
        combined_data_cov,
        combined_data_err,
        tsep_dctnry,
        ft_mdls,
        ini_p0,
    ):
        if cov_enabled:
            err_dctnry = {"combined": gv.gvar(combined_data[:, i], combined_data_cov)}
            # err_dctnry = {"combined": gv.gvar(combined_data[:, 100], combined_data_cov)}
        else:
            err_dctnry = {"combined": gv.gvar(combined_data[:, i], combined_data_err)}
            # err_dctnry = {"combined": gv.gvar(combined_data[:, 100], combined_data_err)}

        # fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, err_dctnry), fcn=ft_mdls, p0=ini_p0, debug=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fit = lsqfit.nonlinear_fit(
                data=({"i": i}, err_dctnry),
                # data=({"i": 100}, err_dctnry),
                svdcut=1e-12,
                fcn=vol_a_mdls,
                p0=ini_p0,
                debug=True,
            )
            # print(i,fit.format(True))
        # print(fit.chi2/fit.dof)
        # 使用字典推导式构造基础字典
        result_dict = {key: fit.p[key].mean for key in param_ranges.keys()}
        # result_dict["trace_anomaly"] = M(
        #     ms_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p
        # ).mean
        result_dict["M_phys"] = fit.p["M0"].mean
        result_dict["chi2"] = fit.chi2 / fit.dof
        result_dict["Q"] = fit.Q

        return result_dict

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
            )
            for i in tqdm(
                range(n_bootstrap), desc="Bootstrap Progress", unit="bootstrap"
            )
        )
        LEC_temp = {
            key: np.array([res[key] for res in results]) for key in param_ranges.keys()
        }
        for key in ["M_phys", "chi2", "Q"]:
            LEC_temp[key] = np.array([res[key] for res in results])
        return LEC_temp

    n_bootstrap = 4000
    LEC = parallel_fit_tqdm(
        n_bootstrap=n_bootstrap,
        n_jobs=-1,
        cov_enabled=cov_enabled,
        combined_data=combined_data,
        combined_data_cov=combined_data_cov,
        combined_data_err=combined_data_err,
        tsep_dctnry=tsep_dctnry,
        ft_mdls=vol_a_mdls,
        ini_p0=best_params,
    )

    # print(LEC.keys())
    # # 创建一个新的字典来存储结果
    LEC_gvar = {}

    # 对于LEC中的每一个参数
    for param, values in LEC.items():
        # 计算均值和标准差
        mean = np.mean(values)
        std_dev = np.std(values)

        # 将均值和标准差转换为gvar的形式
        LEC_gvar[param] = gv.gvar(mean, std_dev)

    # 打印结果
    # print("7th dropped: ", dropped, "cov_enabled: ", cov_enabled)
    # print("LEC using bootstrap:")
    # for param, gvar in LEC_gvar.items():
    #     print(f"{param}: {gvar}")

    # exit()
    # print(LEC_gvar[''])
    pp = LEC_gvar
    ml_phys = 0.003381
    # print(pp)
    m_pi_v = np.mean(m_pi_v_bt, axis=1)
    m_pi_sea = np.mean(m_pi_sea_bt, axis=1)
    # ms_sea = ms_sea_bt[:, i]
    ms_sea = np.mean(ms_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
    ml_sea = np.mean(ml_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
    m_pi_sea = np.mean(m_pi_sea_bt, axis=1)[
        np.arange(len(selected_ensemble_indices)) * 3
    ]
    metas_sea = np.mean(metas_sea_bt, axis=1)[
        np.arange(len(selected_ensemble_indices)) * 3
    ]
    ms_v = np.mean(ms_v_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
    m_pi_sea = np.mean(m_pi_sea_bt, axis=1)[
        np.arange(len(selected_ensemble_indices)) * 3
    ]
    # print(ms_sea.shape)
    # fpi=np.mean(fpi_v_bt,axis=1)
    a = gv.mean(alttc_gv_new)
    L = np.array(ns) * a

    combined_data_corr = (
        -pp["C1"] * (m_pi_sea**2 - m_pi_phys**2)
        - pp["C2"] * (metas_sea**2 - etas_phys**2)
        # + LEC_mass_gvar["C1"] * m_pi_phys**2
        # + LEC_mass_gvar["C2"] * etas_phys**2
        - (pp["CL"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L))
    )
    if not additive_Ca:
        if Ca4Q:
            combined_data_corr * (1 + pp["Ca2"] * a**2 + pp["Ca4"] * a**4)
        else:
            combined_data_corr * (1 + pp["Ca2"] * a**2)

    # combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)  +np.mean(LEC_mass['C2']*etas_phys**2)
    # combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr) +np.mean(LEC_mass['C2']*etas_phys**2) +np.mean(LEC_mass['C1']*m_pi_phys**2)
    combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)
    # print("aaaaaaaaa")
    # print(combined_data_corr)
    # exit()
    # print(ms_v_mean, combined_data_cntrl, combined_data_Ds_phy)
    # combined_data_Ds_phy_list = []
    # combined_data_err_list = []
    labels_list = name[selected_ensemble_indices]
    # colors_list = []
    markers_list = markers
    # a_2_list = []
    a_2_list = gv.mean(alttc_gv_new) ** 2

    def closest_key(a, LEC):
        # 提取所有的 'C1(...)' 键，并分解以得到其中的数值部分
        keys = [key for key in LEC.keys() if key.startswith("C1(")]
        values = [float(key[3:-1]) for key in keys]  # 提取括号中的数值

        # 将提供的 a 值与可用的键中的值进行比较，找到最接近的
        closest_value = min(values, key=lambda x: abs(x - a))

        # 格式化回 'C1(...)' 的键格式
        closest_key = f"C1({closest_value})"
        return closest_key

    def compute_M_baryon(a, LEC, fpi):
        M_baryon = (
            LEC["M0"]
            # + LEC_mass["C1"] * m_pi_phys**2
            # + LEC_mass["C2"] * etas_phys**2
            # + pc1 * (mDs - mc_phys) * (1 + LEC["C1a2"] * a**2)
            # + LEC["C2"] * (ms_sea - ms_phys)
            # + LEC["C3"] * (ml_sea - ml_phys)
            # + LEC["C5"] * a**2+ LEC["C6"] * a**4)
            + LEC["Ca2"] * a**2
        )
        if Ca4Q:
            M_baryon += LEC["Ca4"] * a**4
        # print(M_baryon)
        return M_baryon

    # 使用样本的平均值和标准差来计算上下限
    def compute_limits(samples):
        mean = np.mean(samples, axis=1)
        err = np.std(samples, axis=1)
        return mean - err, mean + err

    fit_a_2 = np.linspace(0, 0.013, 400)
    num_samples = LEC["M0"].shape[0]
    # for key in LEC.keys():
    #     print(LEC[key].shape)
    # for key in LEC:
    #     print(f"Key: {key}, Length of data: {len(LEC[key])}")
    #     if len(LEC[key]) <= i:
    #         print(f"Error: Index {i} out of range for key '{key}'")
    # print(num_samples)
    M_samples = -np.ones((fit_a_2.size, num_samples))
    # M_samples = np.zeros(4000)
    # print(M_samples)
    # exit()
    for idx, a2 in enumerate(fit_a_2):
        M_samples[idx] = [
            compute_M_baryon(
                np.sqrt(a2),
                {key: LEC[key][i] for key in LEC},
                0.122,
            )
            for i in range(num_samples)
        ]
        # M_samples[idx,:]=np.zeros(4000)
    M_mean = np.mean(M_samples, axis=1)
    M_err = np.std(M_samples, axis=1, ddof=1)
    # 将数据保存为一个字典
    data = {
        "combined_data_Ds_phy": combined_data_cntrl,
        "combined_data_err": combined_data_err,
        "labels": labels_list,
        "colors": colors,
        "markers": markers_list,
        "a_2": a_2_list,
        "fit_a_2": fit_a_2,
        "fit_fcn": M_mean,
        "fit_fcn_err": M_err,
        "fit_chi2": fit_chi2,
    }

    # 保存数据到.npy文件
    np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_{data_name}_data_addCa_{not CasQ}_{error_source}.npy", data)

    # 设置画图参数
    errorbar_kwargs = {
        "linewidth": 0,
        "elinewidth": 2,
        "capthick": 5,
        "capsize": 5,
        "mew": 1,
        "linestyle": "none",
        "fillstyle": "none",
    }

    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(15, 10), dpi=140)
    ax = plt.axes([0.2, 0.15, 0.78, 0.8])
    # 计算位移
    displacements = np.zeros(len(data["a_2"]))  # 初始化位移数组
    unique_a2 = np.unique(data["a_2"])  # 获取所有独特的 a^2 值

    # 对于每一个独特的 a^2 值，找出所有对应的索引，并分配位移
    for value in unique_a2:
        indices = np.where(data["a_2"] == value)[0]
        n = len(indices)
        # 中心对齐位移
        offsets = np.linspace(-0.00010 * (n - 1), 0.00010 * (n - 1), n)
        displacements[indices] = offsets

    # 创建图形
    # plt.figure(figsize=(10, 8))
    # ax = plt.axes()

    # 绘图
    for i in range(len(data["combined_data_err"])):
        # for i in range(13):
        # print(data['combined_data_Ds_phy'][i])
        # print(data['a_2'][i] + displacements[i])
        ax.errorbar(
            data["a_2"][i] + displacements[i],
            data["combined_data_Ds_phy"][i],
            yerr=data["combined_data_err"][i],
            color=data["colors"][i],
            label=data["labels"][i],
            marker=data["markers"][i],
            markersize=15,
            **errorbar_kwargs,
        )
    ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="--", linewidth=2)
    ax.fill_between(
        data["fit_a_2"],
        data["fit_fcn"] - data["fit_fcn_err"],
        data["fit_fcn"] + data["fit_fcn_err"],
        alpha=0.5,
        color="grey",
    )

    # Labels and legends
    ax.set_xlim([-0.0003, 0.013])
    # ax.set_ylim([0.0,1.20])
    ax.set_xlabel(r"$a^2 [\mathrm{fm}^2]$", fontsize=35, labelpad=3)
    ax.set_ylabel(r"$H_m\, [\mathrm{GeV}]$", fontsize=35, labelpad=16)
    ax.legend(loc="best", frameon=False, fontsize=20, ncol=3)
    ax.tick_params(axis="both", which="major", direction="in", width=3, length=12)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    quark_content = str(plot_configurations[baryon]["quark_content"])
    mass = str(plot_configurations[baryon]["mass"])
    spin_parity = str(plot_configurations[baryon]["spin_parity"])
    baryon_lat = plot_configurations[baryon]["label_y"].replace("(GeV)", "")
    # 添加带框的注释
    # textstr = f"{(plot_configurations[baryon]['label_y']).replace('(GeV)', '')} ({mass})  {spin_parity} {quark_content}"
    textstr = f"{baryon_lat} ({mass}) {spin_parity} {quark_content} "
    # print(textstr)
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    props = dict(boxstyle="square", facecolor="white", alpha=0.3)
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=25,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3)

    # 显示图表
    # plt.show()
    # plt.savefig(f"{lqcd_data_path}/figures/{baryon}_{data_name}_mass_addCa_{not CasQ}.pdf")


print("========================== Hms part =====================")
Hm_data_fit_and_plot(
    "Hmsv", Hms[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_msv\, $", Ca4Q
)

Hm_data_fit_and_plot(
    "Hmss",
    Hms[np.arange(len(selected_ensemble_indices)) * 3, :] * 0
    + LEC_mass["C5"] * etas_phys**2,
    r"$H_mss\, $",
    Ca4Q,
)

Hms += LEC_mass["C5"] * etas_phys**2
Hm_data_fit_and_plot(
    "Hms", Hms[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_ms\, $", Ca4Q
)
# print(np.mean(Hms[np.arange(len(selected_ensemble_indices)) * 3, :], axis=1))

Hm = Hml + Hms
Hm_data_fit_and_plot(
    "Hm", Hm[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m\, $", Ca4Q
)
