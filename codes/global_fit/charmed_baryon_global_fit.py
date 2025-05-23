#!/usr/bin/env python3.11
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import scipy.linalg as linalg
import sys
from pathlib import Path

# import seaborn as sns
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.parallel import parallel_backend

from params.params import a_fm, a_fm_independent, a_fm_correlated, name
# from utils.create_plot import create_gpt_plot
from params.plt_labels import plot_configurations
# import itertools
import json
import os
from light_quark_fit import fit_light_quark_like_proton

np.set_printoptions(threshold=sys.maxsize, linewidth=800, precision=10, suppress=True)
plt.style.use("utils/physrev.mplstyle")

#hadrons with one or more valance light quark
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

markers = ["d", "o", ".", "8", "x", "+", "v", "^", "<", ">", "s", "s", "s"]
cov_enabled = False
# cov_enabled = False
n_bootstrap = 4000
additive_Ca = "--additive_Ca" in sys.argv
baryon = sys.argv[1]
error_source = sys.argv[2]
lqcd_data_path = sys.argv[4]
print(a_fm_independent)
if error_source == "alttc_sys":
    a_fm_independent += gv.sdev(a_fm_correlated)
    a_fm_correlated += gv.sdev(a_fm_correlated)

if baryon in charm_valance_light:
    num_quark_combination = 27
else:
    num_quark_combination = 9

# additive_Ca = True
selected_ensemble_indices = [*range(8), 8, 9, 10, 11, 12, 14]
alttc_index = np.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 1, 3, 5, 2])
alttc_gv = a_fm[[alttc_index[idx] for idx in selected_ensemble_indices]]
# Z_A_gv=np.array(Z_A_gv[selected_ensemble_indices])
alttc_gv_new = a_fm_independent[[alttc_index[idx] for idx in selected_ensemble_indices]]
# Z_P_gv1, Z_P_gv2, Z_A_gv
ml_col_indices = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
ms_col_indices = np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1]) + 3
mc_col_indices = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]) + 6
ns_info = [24, 24, 32, 32, 48, 48, 32, 48, 32, 48, 48, 28, 36, 48, 64]

pion_uni_indices=np.array([9*n+ml_col_indices[n]*3+i for n in range(len(selected_ensemble_indices)) for i in range(3) ])
print(pion_uni_indices)

ml_indices = np.array(
    [i * 3 + ml_col_indices[index] for i, index in enumerate(selected_ensemble_indices)]
).repeat(3)
ms_indices = np.array(
    [
        i * 3 + ms_col_indices[index] - 3
        for i, index in enumerate(selected_ensemble_indices)
    ]
).repeat(3)

ns = [ns_info[index] for index in selected_ensemble_indices]

def package_joint_fit_data(path_to_pydata, particle_name, key, cov_enabled, quark):
    selected_ensembles = ["ensemble" + str(idx) for idx in selected_ensemble_indices]
    ordered_particles = [
        particle_name,
    ]

    baryon_data = {}
    for particle in ordered_particles:
        # print(particle)
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

    for ii, ensemble_key in enumerate(selected_ensembles):
        # ml_col = ml_col_indices[ii]
        # ms_col = ms_col_indices[ii]
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
                    for particle in ordered_particles:
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
    # print(fit_array_cntrl)

    # return data_obs
    return fit_array_combined_boot


def package_joint_fit_data_lsc(path_to_pydata, particle_name, key, cov_enabled, quark):
    selected_ensembles = ["ensemble" + str(idx) for idx in selected_ensemble_indices]
    ordered_particles = [
        particle_name,
    ]

    baryon_data = {}
    for particle in ordered_particles:
        # print(particle)
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

    for ii, ensemble_key in enumerate(selected_ensembles):
        # ml_col = ml_col_indices[ii]
        # ms_col = ms_col_indices[ii]
        ensemble_array_list = []
        for ml_col in range(3):
            for ms_col in range(3, 6):
                for mc_col in range(6, 9):
                    # 根据quark的值计算条件
                    condition = False
                    if quark == "s":
                        condition = (
                            ml_col == ml_col_indices[ii]
                            and mc_col == mc_col_indices[ii]
                        )
                    elif quark == "l":
                        condition = (
                            ms_col == ms_col_indices[ii]
                            and mc_col == mc_col_indices[ii]
                        )
                    elif quark == "nl":
                        condition = ml_col == ml_col_indices[ii]
                    elif quark == "c":
                        condition = (
                            ms_col == ms_col_indices[ii]
                            and ml_col == ml_col_indices[ii]
                        )
                    elif quark == "a":
                        condition = True
                    # if ms_col == ms_col_indices[ii]:
                    if condition:
                        # print(ii, ml_col, ms_col, mc_col)
                        # print(ensemble_values)
                        # print(ensemble_key)
                        for particle in ordered_particles:
                            # print(baryon_data[particle])
                            ensemble_values = baryon_data[particle][key][ensemble_key]
                            # ensemble_array_list.append(ensemble_values[ml_col][ms_col])
                            ensemble_array_list.append(
                                ensemble_values[ml_col][ms_col][mc_col]
                            )
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
    # print(fit_array_cntrl)

    # return data_obs
    return fit_array_combined_boot


m_pi_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "PION_fpi_mq_mPS",
    "mPS",
    True,
    "l",
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
metas_v_bt = package_joint_fit_data(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "ETA_S_fpi_mq_mPS",
    "mPS",
    True,
    "s",
)

Ds_v_bt_pq = package_joint_fit_data_lsc(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "D_S_fpi_mq_mPS",
    "mPS",
    True,
    "nl",
)

msc_v_bt_pq = package_joint_fit_data_lsc(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
    "D_S_fpi_mq_mPS",
    "mR",
    True,
    "nl",
)

print(np.mean(Ds_v_bt_pq, axis=1).shape)
print(np.mean(msc_v_bt_pq, axis=1))
print(np.mean(metas_v_bt, axis=1))
print(np.mean(ms_v_bt, axis=1))
print(np.mean(m_pi_v_bt, axis=1))
print(np.mean(ml_v_bt, axis=1))

# if error_source == '2pt_sys':
#     suffix='h5_s'
# else:
#     suffix='h5'
suffix = "h5"

if baryon == "D_S":
    m_baryon_v_bt_pq = -np.ones_like(Ds_v_bt_pq)
    m_baryon_v_bt_pq = Ds_v_bt_pq
else:
    if baryon in charm_valance_light:
        quarks = "a"
    else:
        quarks = "nl"
    m_baryon_v_bt_pq = package_joint_fit_data_lsc(
    os.path.expanduser(f"{lqcd_data_path}/precomputed_pydata_eff_mass/"),
        baryon,
        "mp",
        True,
        quarks,
    )
# print(m_baryon_v_bt_pq[9*3+np.arange(9),0])
# print(m_baryon_v_bt_pq[np.arange(9),0])
print(m_baryon_v_bt_pq[:, 0].shape)
print(m_baryon_v_bt_pq[:, 0])
# Z_A_gv = (Z_A_Z_V * Z_V)[[alttc_index[idx] for idx in selected_ensemble_indices]]
# Z_P_gv = (Z_P_Z_V * Z_V)[[alttc_index[idx] for idx in selected_ensemble_indices]]
print(m_baryon_v_bt_pq.shape)
print(alttc_gv, alttc_gv.shape)
# print(Z_V_gv)
# print(Z_A_gv)
# print(Z_P_gv)
# exit()
alttc_cntrl = np.array(
    [
        np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in alttc_gv
        for _ in range(num_quark_combination)
    ]
)

a_fm_samples = np.array(
    [
        # np.random.normal(loc=g.mean, scale=g.sdev * np.sqrt(1.8), size=n_bootstrap)
        np.random.normal(loc=g.mean, scale=g.sdev, size=n_bootstrap)
        # np.random.normal(loc=g.mean, scale=0, size=n_bootstrap)
        for g in a_fm_independent
    ]
)

normalized_gaussian = np.random.normal(loc=0, scale=1, size=n_bootstrap)
a_fm_correlated_error = np.array(
    [normalized_gaussian * g.sdev for g in a_fm_correlated]
)


if error_source == "alttc_stat":
    a_fm_samples = np.repeat(np.mean(a_fm_samples, axis=1)[:, np.newaxis], 4000, axis=1)

a_fm_samples += a_fm_correlated_error

# print(np.cov(a_fm_samples[[0,1],:]))
print(np.cov(a_fm_samples[:, :]))
# exit()

# Z_A_Z_P_fm_samples=np.array(
#     [
#         np.random.normal(loc=g.mean, scale=g.sdev, size=n_bootstrap)
#         for g in Z_A_Z_V/Z_P_Z_V
#     ]
# )


alttc_samples = np.array(
    [
        a_fm_samples[alttc_index[idx]]
        for idx in selected_ensemble_indices
        for _ in range(num_quark_combination)
    ]
)
alttc_samples_3 = np.array(
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
# Z_A_Z_P_samples= np.array([ Z_A_Z_P_fm_samples[alttc_index[idx]] for idx in selected_ensemble_indices for _ in range(3)])
# print(Z_A_Z_P_samples[:,:2])
# exit()
alttc_cntrl_3 = np.array(
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
# exit()
msc_v_bt_pq *= alttc_cntrl_9 / alttc_samples_9
ms_v_bt *= alttc_cntrl_3 / alttc_samples_3
ml_v_bt *= alttc_cntrl_3 / alttc_samples_3
metas_v_bt *= alttc_cntrl_3 / alttc_samples_3

Ds_v_bt_pq *= alttc_cntrl_9 / alttc_samples_9
# metac_v_bt_pq *= alttc_cntrl / alttc_samples
print(alttc_cntrl.shape, alttc_samples.shape, m_baryon_v_bt_pq.shape)
m_baryon_v_bt_pq *= alttc_cntrl / alttc_samples

m_pi_sea_bt = m_pi_v_bt[ml_indices, :]
if error_source == "mpi":
    m_pi_sea_bt = np.repeat(np.mean(m_pi_sea_bt, axis=1)[:, np.newaxis], 4000, axis=1)
ml_sea_bt = ml_v_bt[ml_indices, :]
ms_sea_bt = ms_v_bt[ms_indices, :]
metas_sea_bt = metas_v_bt[ms_indices, :]
if error_source == "metas":
    metas_sea_bt = np.repeat(np.mean(metas_sea_bt, axis=1)[:, np.newaxis], 4000, axis=1)
tsep_dctnry = {"combined": np.arange(len(selected_ensemble_indices))}

def predict_y(x_values, y_values, x_to_predict, degree=2):
    # coefficients = np.polyfit(x_values, y_values, degree)
    coefficients = np.polyfit(x_values, y_values, 1)
    # 创建多项式函数
    polynomial = np.poly1d(coefficients)
    # 使用多项式函数预测y值
    y_predicted = polynomial(x_to_predict)
    return y_predicted


# def predict_y_slop(x_values, y_values, x_to_predict,printQ, degree=1):
def predict_y_slop(x_values, y_values, x_to_predict, degree=1):
    # 拟合多项式并生成多项式函数
    polynomial = np.poly1d(np.polyfit(x_values, y_values, 1))
    # 预测y值
    y_predicted = polynomial(x_to_predict)
    # 计算斜率
    y_predicted_slop = np.polyder(polynomial)(x_to_predict)
    # y_predicted_slop = np.polyder(polynomial)(x_values[0])
    # y_predicted_slop = (y_values[2]-y_values[1])/(x_values[2]-x_values[1])
    y_predicted_slop1 = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
    y_predicted_slop2 = (y_values[2] - y_values[0]) / (x_values[2] - x_values[0])
    y_predicted_slop3 = (y_values[2] - y_values[1]) / (x_values[2] - x_values[1])
    # if printQ == True:
    # print(y_predicted_slop,y_predicted_slop2)
    # print(y_predicted_slop)
    return y_predicted, y_predicted_slop


# etas_phys = 0.6874
etas_phys = 0.68963
D_S_phys = 1966.7 / 1000
if error_source == "D_S":
    D_S_phys += 1.5 / 1000

m_pi_phys = 0.135  # Adjust this value to the correct one
# print(metas_v_bt.shape)

Ds_v_etas069 = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
msc_v_etas069 = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
ms_v_etas069 = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
ml_v_pion_phys = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
mc_v_etas690_Ds_phys = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))

if baryon not in charm_valance_light:
    m_baryon_v_etas069 = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
    m_baryon_v_etas069_Ds_phys_slop = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys_ms_phys_slop = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_etas069_Ds_phys = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys_ms_phys = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys = -np.ones((3 * len(selected_ensemble_indices), n_bootstrap))
else:
    m_baryon_v_etas069 = -np.ones((9 * len(selected_ensemble_indices), n_bootstrap))
    m_baryon_v_etas069_pion_uni = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_etas069_pion_uni_Ds_phys = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_etas069_pion_uni_Ds_phys_slop = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys = -np.ones((9 * len(selected_ensemble_indices), n_bootstrap))
    m_baryon_v_mc_phys_etas069 = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys_pion_uni = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    # m_baryon_v_mc_phys_pion_uni = -np.ones(
    #     (3 * len(selected_ensemble_indices), n_bootstrap)
    # )
    # m_baryon_v_mc_phys_etas069_pion_uni = -np.ones(
    #     (3 * len(selected_ensemble_indices), n_bootstrap)
    # )
    # m_baryon_v_mc_phys_etas069_pion_uni_slop = -np.ones(
    #     (3 * len(selected_ensemble_indices), n_bootstrap)
    # )
    m_baryon_v_mc_phys_pion_uni_ms_phys = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )
    m_baryon_v_mc_phys_pion_uni_ms_phys_slop = -np.ones(
        (3 * len(selected_ensemble_indices), n_bootstrap)
    )

for i in range(Ds_v_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(3):
            Ds_v_etas069[n * 3 + ii, i] = predict_y(
                metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                Ds_v_bt_pq[(9 * n + 3 * np.arange(3) + ii), i],
                etas_phys**2,
                degree=2,
            )

for i in range(Ds_v_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(3):
            msc_v_etas069[n * 3 + ii, i] = 2 * predict_y(
                metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                msc_v_bt_pq[(9 * n + 3 * np.arange(3) + ii), i],
                etas_phys**2,
                degree=2,
            )

# print(msc_v_bt_pq[:, 0])
for i in range(Ds_v_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(3):
            ms_v_etas069[n * 3 + ii, i] = predict_y(
                metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                ms_v_bt[range(3 * n, 3 * n + 3), i],
                etas_phys**2,
                degree=2,
            )

mc_v_etas069 = msc_v_etas069 - ms_v_etas069

for i in range(Ds_v_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(3):
            mc_v_etas690_Ds_phys[n * 3 + ii, i] = predict_y(
                Ds_v_etas069[range(3 * n, 3 * n + 3), i],
                mc_v_etas069[range(3 * n, 3 * n + 3), i],
                D_S_phys,
                degree=2,
            )

if baryon not in charm_valance_light:
    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):  # ii is charm index
                m_baryon_v_etas069[n * 3 + ii, i] = predict_y(
                    metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                    m_baryon_v_bt_pq[(9 * n + 3 * np.arange(3) + ii), i],
                    etas_phys**2,
                    degree=2,
                )
else:
    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):  # ii is charm index
                for jj in range(3):  # jj is light index
                    m_baryon_v_etas069[n * 9 + ii + 3 * jj, i] = predict_y(
                        metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                        m_baryon_v_bt_pq[(27 * n + 3 * np.arange(3) + ii + 9 * jj), i],
                        etas_phys**2,
                        degree=2,
                    )

if baryon not in charm_valance_light:
    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):
                m_baryon_v_mc_phys[n * 3 + ii, i] = predict_y(
                    mc_v_etas069[range(3 * n, 3 * n + 3), i],
                    m_baryon_v_bt_pq[(9 * n + np.arange(3) + 3 * ii), i],
                    mc_v_etas690_Ds_phys[3 * n, i],
                    degree=2,
                )
else:
    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):  # ii is light index
                for jj in range(3):  # jj is strange index
                    m_baryon_v_mc_phys[n * 9 + jj + 3 * ii, i] = predict_y(
                        mc_v_etas069[range(3 * n, 3 * n + 3), i],
                        m_baryon_v_bt_pq[(27 * n + np.arange(3) + 3 * jj + 9*ii), i],
                        mc_v_etas690_Ds_phys[3 * n, i],
                        degree=2,
                    )

# print(np.mean(m_baryon_v_bt_pq,axis=1))
# print(np.mean(m_baryon_v_mc_phys,axis=1))
# exit()

if baryon in charm_valance_light:
    # for i in range(Ds_v_etas069.shape[1]):
    #     for n in range(len(selected_ensemble_indices)):
    #         for ii in range(3):  # ii is charm index
    #             m_baryon_v_etas069_pion_phys[n * 3 + ii, i] = predict_y(
    #                 m_pi_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
    #                 m_baryon_v_etas069[(9 * n + 3 * np.arange(3) + ii), i],
    #                 m_pi_phys**2,
    #                 degree=2,
    #             )

    m_baryon_v_etas069_pion_uni=m_baryon_v_etas069[pion_uni_indices, :]

    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):  # ii is charm index
                m_baryon_v_mc_phys_etas069[n * 3 + ii, i] = predict_y(
                    metas_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
                    m_baryon_v_mc_phys[(9 * n + np.arange(3) + 3*ii), i],
                    etas_phys**2,
                    degree=2,
                )
    # for i in range(Ds_v_etas069.shape[1]):
    #     for n in range(len(selected_ensemble_indices)):
    #         for ii in range(3):  # ii is charm index
    #             m_baryon_v_mc_phys_pion_phys[n * 3 + ii, i] = predict_y(
    #                 m_pi_v_bt[range(3 * n, 3 * n + 3), i] ** 2,
    #                 m_baryon_v_mc_phys[(9 * n + 3*np.arange(3) + ii), i],
    #                 m_pi_phys**2,
    #                 degree=2,
    #             )

    m_baryon_v_mc_phys_pion_uni=m_baryon_v_mc_phys[pion_uni_indices,:]

print('btpq',m_baryon_v_bt_pq[np.arange(9*11,9*12),0])
print('etas69',m_baryon_v_etas069[np.arange(3*11,3*12),0])
# print('etas69pionuni',m_baryon_v_etas069_pion_uni[:,0])
# exit()
for i in range(Ds_v_etas069.shape[1]):
    for n in range(len(selected_ensemble_indices)):
        for ii in range(1):
            if baryon not in charm_valance_light:
                (
                    m_baryon_v_etas069_Ds_phys[n * 3 + ii, i],
                    m_baryon_v_etas069_Ds_phys_slop[n * 3 + ii, i],
                ) = predict_y_slop(
                    mc_v_etas069[range(3 * n, 3 * n + 3), i],
                    m_baryon_v_etas069[range(3 * n, 3 * n + 3), i],
                    # m_baryon_v_bt_pq[range(9 * n, 9 * n + 3), i],
                    mc_v_etas690_Ds_phys[n * 3, i],
                    degree=2,
                )
            else:
                (
                    m_baryon_v_etas069_pion_uni_Ds_phys[n * 3 + ii, i],
                    m_baryon_v_etas069_pion_uni_Ds_phys_slop[n * 3 + ii, i],
                ) = predict_y_slop(
                    mc_v_etas069[range(3 * n, 3 * n + 3), i],
                    m_baryon_v_etas069_pion_uni[range(3 * n, 3 * n + 3), i],
                    mc_v_etas690_Ds_phys[n * 3, i],
                    degree=2,
                )
                (
                    m_baryon_v_mc_phys_pion_uni_ms_phys[n * 3 + ii, i],
                    m_baryon_v_mc_phys_pion_uni_ms_phys_slop[n * 3 + ii, i],
                ) = predict_y_slop(
                    ms_v_bt[range(3 * n, 3 * n + 3), i],
                    m_baryon_v_mc_phys_pion_uni[range(3 * n, 3 * n + 3), i],
                    ms_v_etas069[n * 3, i],
                    degree=2,
                )
        for ii in range(1, 3):
            if baryon not in charm_valance_light:
                (
                    m_baryon_v_etas069_Ds_phys[n * 3 + ii, i],
                    m_baryon_v_etas069_Ds_phys_slop[n * 3 + ii, i],
                ) = (
                    m_baryon_v_etas069_Ds_phys[n * 3, i],
                    m_baryon_v_etas069_Ds_phys_slop[n * 3, i],
                )
            else:
                (
                    m_baryon_v_etas069_pion_uni_Ds_phys[n * 3 + ii, i],
                    m_baryon_v_mc_phys_pion_uni_ms_phys[n * 3 + ii, i],
                    m_baryon_v_etas069_pion_uni_Ds_phys_slop[n * 3 + ii, i],
                    m_baryon_v_mc_phys_pion_uni_ms_phys_slop[n * 3 + ii, i],
                ) = (
                    m_baryon_v_etas069_pion_uni_Ds_phys[n * 3, i],
                    m_baryon_v_mc_phys_pion_uni_ms_phys[n * 3, i],
                    m_baryon_v_etas069_pion_uni_Ds_phys_slop[n * 3, i],
                    m_baryon_v_mc_phys_pion_uni_ms_phys_slop[n * 3, i],
                )

# print(np.mean(m_baryon_v_etas069_pion_uni_Ds_phys[:, :], axis=1))
# print(np.mean(m_baryon_v_mc_phys_pion_uni_ms_phys[:, :], axis=1))
# print('etas69dsphys',m_baryon_v_etas069_pion_uni_Ds_phys[:,0])
# exit()
ml_sea_bt = ml_v_bt[ml_indices, :]
if baryon in charm_valance_light:
    np.savez(f"{baryon}_debug_fit_input.npz",
            baryon=baryon,
            m_baryon_v_mc_phys_etas069=m_baryon_v_mc_phys_etas069,
            m_pi_v_bt=m_pi_v_bt,
            m_pi_sea_bt=m_pi_sea_bt,
            metas_sea_bt=metas_sea_bt,
            alttc_samples_3=alttc_samples_3,
            selected_ensemble_indices=selected_ensemble_indices,
            ns=ns)
    Hm_light_bt, baryon_mass_bt=fit_light_quark_like_proton(baryon,m_baryon_v_mc_phys_etas069, m_pi_v_bt, m_pi_sea_bt, metas_sea_bt, alttc_samples_3, selected_ensemble_indices, ns, error_source, additive_Ca, lqcd_data_path)

if baryon not in charm_valance_light:
    for i in range(Ds_v_etas069.shape[1]):
        for n in range(len(selected_ensemble_indices)):
            for ii in range(3):
                (
                    m_baryon_v_mc_phys_ms_phys[n * 3 + ii, i],
                    m_baryon_v_mc_phys_ms_phys_slop[n * 3 + ii, i],
                ) = predict_y_slop(
                    ms_v_bt[range(3 * n, 3 * n + 3), i],
                    m_baryon_v_mc_phys[range(3 * n, 3 * n + 3), i],
                    ms_v_etas069[n * 3, i],
                    degree=2,
                )

if baryon not in charm_valance_light:
    Hmc = m_baryon_v_etas069_Ds_phys_slop * mc_v_etas690_Ds_phys
    Hms = m_baryon_v_mc_phys_ms_phys_slop * ms_v_etas069
    Hml = np.zeros_like(Hms)
else:
    Hmc = m_baryon_v_etas069_pion_uni_Ds_phys_slop * mc_v_etas690_Ds_phys
    Hms = m_baryon_v_mc_phys_pion_uni_ms_phys_slop * ms_v_etas069
    # Hml = m_baryon_v_mc_phys_etas069_pion_phys_slop * m_pi_phys**2


# MEc= m_baryon_v_etas069_Ds_phys_slop/Z_A_Z_P_samples
print()
# print((mc_v_etas690_Ds_phys*Z_A_Z_P_samples)[:,0])
# print(MEc[:,0])
# print(Hmc.shape)
# print(Hmc[:, 0])
# print(Hms.shape)
# print(Hms[:, 0])
# exit()
# np.savetxt(f"./{baryon}_Hm_saved.txt", np.mean(Hm[:, :],axis=1))
# 将数据保存为一个字典
# data_to_save = {
#     "combined_data_Ds_phy": np.mean(Hm[:, :],axis=1),
#     "combined_data_err": np.std(Hm[:, :],axis=1),
#     "a_2": a_2_list,
# }
# plt.figure(figsize=(8, 6))
# print(np.mean(m_baryon_v_etas069_Ds_phys_slop[30,:]),np.std(m_baryon_v_etas069_Ds_phys_slop[30,:]))
# print(np.mean(m_baryon_v_etas069_Ds_phys[30,:]),np.std(m_baryon_v_etas069_Ds_phys[30,:]))
# print(np.mean(m_baryon_v_etas069[30,:]),np.std(m_baryon_v_etas069[30,:]))
# for i in range(len(selected_ensemble_indices)):
#     plt.errorbar(
#         i,
#         np.mean(Hm[3 * i, :], axis=0),
#         np.std(Hm[3 * i, :], axis=0),
#     )
#     # plt.errorbar(i, np.mean(m_baryon_v_etas069_Ds_phys[3*i, :],axis=0),np.std(m_baryon_v_etas069_Ds_phys[3*i, :],axis=0))
#     # plt.errorbar(i, np.mean(m_baryon_v_etas069[3*i, :],axis=0),np.std(m_baryon_v_etas069[3*i, :],axis=0))
#     # plt.errorbar(a_2_list[i], np.mean(m_baryon_v_etas069_Ds_phys[3*i, :],axis=0),np.std(m_baryon_v_etas069_Ds_phys[3*i, :],axis=0),)
#     # plt.errorbar(a_2_list[i], np.mean(m_baryon_v_etas069_Ds_phys_slop[3*i, :],axis=0),np.std(m_baryon_v_etas069_Ds_phys_slop[3*i, :],axis=0),)
#     # plt.errorbar(a_2_list[i], np.mean(mc_v_etas690_Ds_phys[3*i, :],axis=0),np.std(mc_v_etas690_Ds_phys[3*i, :],axis=0),)
# plt.ylim(0.0, 1.5)


colors = [
    "r",
    "r",
    "r",
    "r",
    "r",
    "r",
    "b",
    "b",
    "b",
    "b",
    "g",
    "orange",
    "purple",
    "b",
]

markers = ["^", "<", "v", ">", "x", "s", "d", "o", "8", "p", "*", "h", "H", "."]
# combined_data = mc_v_etas069
# combined_data = Hm[np.arange(len(selected_ensemble_indices))*3,:]
if baryon not in charm_valance_light:
    combined_data = m_baryon_v_etas069_Ds_phys[
        np.arange(len(selected_ensemble_indices)) * 3, :
    ]
else:
    combined_data = m_baryon_v_etas069_pion_uni_Ds_phys[
        np.arange(len(selected_ensemble_indices)) * 3, :
    ]
combined_data_cntrl = np.mean(combined_data, axis=1)
# combined_data_cntrl += np.array([ -0.07 if i in range(11,12) else 0 for i in range(len(selected_ensemble_indices)) ])
combined_data_err = np.std(combined_data, axis=1, ddof=1)
combined_data_cov = np.cov(combined_data)

if cov_enabled == True:
    data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
    # data_dctnry = {"combined": gv.gvar(combined_data[:,0], combined_data_cov)}
else:
    data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
N_f = 2
print(data_dctnry)
# exit()

print("------------------")

ms_phys = 0.0978
ml_phys = 0.003571
mc_phys = 1.085
# m_pi_phys = 0.13498  # Adjust this value to the correct one
# D_S_phys=1968.34/1000
# D_S_phys = 1962.9 / 1000


print(mc_v_etas690_Ds_phys[np.arange(len(selected_ensemble_indices)) * 3, 0])


# exit( )
def vol_a_mdls(idct, p):
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
    a = alttc_samples_3[np.arange(len(selected_ensemble_indices)) * 3, i]
    L = np.array(ns) * a
    # pc1_arr=np.array([p['C1(0.105)'],p['C1(0.077)'],p['C1(0.05)'],p['C1(0.088)'],p['C1(0.068)']])
    # pc1=pc1_arr[alttc_index].repeat(3)
    M_baryon = (
        p["M0"]
        + p["C1"] * (m_pi_sea**2 - m_pi_phys**2)
        + p["C2"] * (metas_sea**2 - etas_phys**2)
        + p["CL"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L)
    )
    corr = p["Ca2"] * a**2 + p["Ca4"] * a**4
    if additive_Ca:
        M_baryon += corr
    else:
        M_baryon *= 1 + corr
    mdls["combined"] = np.array(M_baryon)
    return mdls


def fit_with_initial_values(initial_params):
    if cov_enabled == True:
        data_dctnry_loc = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
    else:
        data_dctnry_loc = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
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


# 初始参数范围
param_ranges = {
    "M0": (0, 2),
    "C1": (-3, 3),
    "C2": (-3, 3),
    # "C3": (-3, 3),
    "CL": (-3, 3),
    "Ca2": (-3, 3),
    "Ca4": (-3, 3),
}


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

print(fit.format(True))  # ? print out fit results
# print(fit_vol.format(True))  # ? print out fit results
print(fit.p)
chi2 = fit.chi2 / fit.dof
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
        # err_dctnry = {"combined": gv.gvar(combined_data[:, i]+np.array([ -0.07 if i in range(11,12) else 0 for i in range(13) ]), combined_data_cov)}
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
    result_dict["trace_anomaly"] = M(
        ms_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p
    ).mean
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
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress", unit="bootstrap")
    )
    LEC_temp = {
        key: np.array([res[key] for res in results]) for key in param_ranges.keys()
    }
    for key in ["M_phys", "chi2", "Q", "trace_anomaly"]:
        LEC_temp[key] = np.array([res[key] for res in results])
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
if baryon == 'D_S':
    LEC_mass["C1"]*=1e-6
    LEC_mass["C2"]*=1e-6
print(LEC_mass.keys())
print(np.mean(LEC_mass["C1"] * m_pi_phys**2), np.std(LEC_mass["C1"] * m_pi_phys**2))
print(np.mean(LEC_mass["C2"] * etas_phys**2), np.std(LEC_mass["C2"] * etas_phys**2))


# # 创建一个新的字典来存储结果
LEC_mass_gvar = {}

# 对于LEC中的每一个参数
for param, values in LEC_mass.items():
    # 计算均值和标准差
    mean = np.mean(values)
    std_dev = np.std(values)

    # 将均值和标准差转换为gvar的形式
    LEC_mass_gvar[param] = gv.gvar(mean, std_dev)

# 打印结果
# print("7th dropped: ", dropped, "cov_enabled: ", cov_enabled)
print("LEC using bootstrap:")
for param, gvar in LEC_mass_gvar.items():
    print(f"{param}: {gvar}")

# exit()
# print(LEC_gvar[''])
pp = LEC_mass_gvar
ml_phys = 0.003381
# print(pp)
m_pi_v = np.mean(m_pi_v_bt, axis=1)
m_pi_sea = np.mean(m_pi_sea_bt, axis=1)
# ms_sea = ms_sea_bt[:, i]
ms_sea = np.mean(ms_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
# ml_sea = np.mean(ml_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices))*3]
m_pi_sea = np.mean(m_pi_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
metas_sea = np.mean(metas_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
ms_v = np.mean(ms_v_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
m_pi_sea = np.mean(m_pi_sea_bt, axis=1)[np.arange(len(selected_ensemble_indices)) * 3]
# print(ms_sea.shape)
# fpi=np.mean(fpi_v_bt,axis=1)
a = gv.mean(alttc_gv_new)
L = np.array(ns) * a

combined_data_corr = (
    -pp["C1"] * (m_pi_sea**2 - m_pi_phys**2)
    - pp["C2"] * (metas_sea**2 - etas_phys**2)
    - (pp["CL"] * (m_pi_sea**2 / L) * np.exp(-m_pi_sea * L))
)

if not additive_Ca:
    combined_data_corr *= 1 + pp["Ca2"] * a**2 + pp["Ca4"] * a**4

combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)
# combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)
print(gv.mean(combined_data_corr))
ms_v_mean = np.mean(mc_v_etas069[np.arange(len(selected_ensemble_indices)) * 3], axis=1)
# print(m_pi_v_mean_square)
x_1053 = ms_v_mean[:18]
y_1053 = combined_data_cntrl[:18]
err_1053 = combined_data_err[:18]

x_0775 = ms_v_mean[18:30]
y_0775 = combined_data_cntrl[18:30]
err_0775 = combined_data_err[18:30]

x_0519 = ms_v_mean[30:33]
y_0519 = combined_data_cntrl[30:33]
err_0519 = combined_data_err[30:33]

x_0887 = ms_v_mean[33:36]
y_0887 = combined_data_cntrl[33:36]
err_0887 = combined_data_err[33:36]

x_0682 = ms_v_mean[36:39]
y_0682 = combined_data_cntrl[36:39]
err_0682 = combined_data_err[36:39]

# 开始绘图
plt.figure(figsize=(15, 10), dpi=100)
ax = plt.gca()

print("aaaaaaaaa")
print(ms_v_mean, combined_data_cntrl)
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


def compute_M_baryon(mDs, a, LEC, fpi):
    M_baryon = (
        LEC["M0"]
        # + LEC["C1"] * (mDs - D_S_phys)
        # + pc1 * (mDs - mc_phys) * (1 + LEC["C1a2"] * a**2)
        # + LEC["C2"] * (ms_sea - ms_phys)
        # + LEC["C3"] * (ml_sea - ml_phys)
        # + LEC["C5"] * a**2+ LEC["C6"] * a**4)
        # + LEC["Ca2"] * a**2
        # + LEC["Ca4"] * a**4
    )
    corr = LEC["Ca2"] * a**2 + LEC["Ca4"] * a**4
    if additive_Ca:
        M_baryon += corr
    else:
        M_baryon *= 1 + corr

    # print(M_baryon)
    return M_baryon


# 使用样本的平均值和标准差来计算上下限
def compute_limits(samples):
    mean = np.mean(samples, axis=1)
    err = np.std(samples, axis=1)
    return mean - err, mean + err


fit_a_2 = np.linspace(0, 0.013, 400)
num_samples = LEC_mass["M0"].shape[0]

M_samples = -np.ones((fit_a_2.size, num_samples))
for idx, a2 in enumerate(fit_a_2):
    M_samples[idx] = [
        compute_M_baryon(
            D_S_phys, np.sqrt(a2), {key: LEC_mass[key][i] for key in LEC_mass}, 0.122
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
    "d1_mpi2": LEC_mass_gvar["C1"].mean * m_pi_phys**2,
    "fit_chi2": chi2,
}

outdir = Path(lqcd_data_path) / "pydata_global_fit"
outdir.mkdir(parents=True, exist_ok=True)  # 自动创建目录
outdir2 = Path(lqcd_data_path) / "figures"
outdir2.mkdir(parents=True, exist_ok=True)  # 自动创建目录
# 保存数据到.npy文件
np.save(
    f"{outdir}/{baryon}_mass_uni_data_addCa_{additive_Ca}_{error_source}.npy",
    data,
)

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

# plt.figure(figsize=(10, 8))
plt.figure(figsize=(4, 3), dpi=140)
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
        **errorbar_kwargs,
    )
ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="--")
ax.fill_between(
    data["fit_a_2"],
    data["fit_fcn"] - data["fit_fcn_err"],
    data["fit_fcn"] + data["fit_fcn_err"],
    alpha=0.5,
    edgecolor="none",
    color="grey",
)


# Labels and legends
ax.set_xlim([-0.0003, 0.013])
# ax.set_ylim([0.0,1.20])
ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$", labelpad=3)
ax.set_ylabel(r"$m_H  (\mathrm{GeV})$", labelpad=16)
ax.legend(loc="best", frameon=False, ncol=3)
ax.minorticks_on()
plt.xticks()
plt.yticks()
quark_content = str(plot_configurations[baryon]["quark_content"])
mass = str(plot_configurations[baryon]["mass"])
spin_parity = str(plot_configurations[baryon]["spin_parity"])
baryon_lat = plot_configurations[baryon]["label_y"].replace("(GeV)", "")
# 添加带框的注释
# textstr = f"{(plot_configurations[baryon]['label_y']).replace('(GeV)', '')} ({mass})  {spin_parity} {quark_content}"
# trace_anomaly=gv.gvar(np.mean(LEC['M_phys']),np.std(LEC['M_phys'],ddof=1))
textstr = f"{baryon_lat} ({mass}) {spin_parity} {quark_content} "
print(textstr)
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
props = dict(boxstyle="square", facecolor="white", alpha=0.3)
ax.text(
    0.05,
    0.05,
    textstr,
    transform=ax.transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    bbox=props,
)

plt.savefig(f"{outdir2}/{baryon}_mass_addCa_{additive_Ca}_{error_source}.pdf")


def Hm_data_fit_and_plot(data_name, combined_data, ylabel, a4Q=True):
    combined_data_cntrl = np.mean(combined_data, axis=1)
    combined_data_err = np.std(combined_data, axis=1, ddof=1)
    combined_data_cov = np.cov(combined_data)

    if cov_enabled == True:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
        # data_dctnry = {"combined": gv.gvar(combined_data[:,0], combined_data_cov)}
    else:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}
    N_f = 2

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
        a = alttc_samples_3[np.arange(len(selected_ensemble_indices)) * 3, i]
        L = np.array(ns) * a
        # pc1_arr=np.array([ppp['C1(0.105)'],ppp['C1(0.077)'],ppp['C1(0.05)'],ppp['C1(0.088)'],ppp['C1(0.068)']])
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
        if a4Q:
            M_baryon += ppp["Ca4"] * a**4
        mdls["combined"] = np.array(M_baryon)
        return mdls

    # 初始参数范围
    param_ranges = {
        "M0": (0, 2),
        "C1": (-3, 3),
        "C2": (-3, 3),
        "CL": (-3, 3),
        "Ca2": (-3, 3),
    }
    if a4Q:
        param_ranges["Ca4"] = (-3, 3)

    def generate_trial_params():
        return {
            param: np.random.uniform(low, high)
            for param, (low, high) in param_ranges.items()
        }

    n_trials = 4000
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

    print("Best Initial Parameters:", best_params)
    print("Best chi2/dof:", best_chi2_dof)
    print("Best Q:", best_Q)
    fit = lsqfit.nonlinear_fit(
        data=({"i": 0}, data_dctnry),
        svdcut=1e-12,
        fcn=vol_a_mdls,
        p0=best_params,
        debug=True,
    )  # ? excute the fit

    print(data_name)
    print(fit.format(True))  # ? print out fit results
    # print(fit_vol.format(True))  # ? print out fit results
    print(fit.p)
    chi2 = fit.chi2 / fit.dof

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
        result_dict["trace_anomaly"] = M(
            ms_phys, m_pi_phys, ms_phys, 0, float("inf"), fit.p
        ).mean
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
        for key in ["M_phys", "chi2", "Q", "trace_anomaly"]:
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

    print(LEC.keys())
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
    print("LEC using bootstrap:")
    for param, gvar in LEC_gvar.items():
        print(f"{param}: {gvar}")

    if data_name in ["Hmsv", "Hmc"]:
        file_path = f"{data_name}_c1c2.json"
        if os.path.exists(file_path):
            # 如果文件存在，读取现有内容
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
        else:
            # 如果文件不存在，初始化一个空字典
            data = {}

        # 更新字典
        data[baryon] = {}
        data[baryon]["C1"] = str(LEC_gvar["C1"])
        data[baryon]["C2"] = str(LEC_gvar["C2"])

        # 将更新后的字典保存回json文件
        # with open(file_path, "w") as json_file:
        #     json.dump(data, json_file, indent=4)

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

    # combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)  +np.mean(LEC_mass['C2']*etas_phys**2)
    # combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr) +np.mean(LEC_mass['C2']*etas_phys**2) +np.mean(LEC_mass['C1']*m_pi_phys**2)
    combined_data_cntrl = combined_data_cntrl + gv.mean(combined_data_corr)
    print(gv.mean(combined_data_corr))
    ms_v_mean = np.mean(
        mc_v_etas069[np.arange(len(selected_ensemble_indices)) * 3], axis=1
    )
    # print(m_pi_v_mean_square)

    # 开始绘图
    plt.figure(figsize=(15, 10), dpi=100)
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

    def compute_M_baryon(mDs, a, LEC, LEC_mass, fpi):
        M_baryon = LEC["M0"] + LEC["Ca2"] * a**2 + (LEC["Ca4"] * a**4 if a4Q else 0)
        # print(M_baryon)
        return M_baryon

    # 使用样本的平均值和标准差来计算上下限
    def compute_limits(samples):
        mean = np.mean(samples, axis=1)
        err = np.std(samples, axis=1)
        return mean - err, mean + err

    fit_a_2 = np.linspace(0, 0.013, 400)
    num_samples = LEC["M0"].shape[0]
    M_samples = -np.ones((fit_a_2.size, num_samples))
    # exit()
    for idx, a2 in enumerate(fit_a_2):
        M_samples[idx] = [
            compute_M_baryon(
                D_S_phys,
                np.sqrt(a2),
                {key: LEC[key][i] for key in LEC},
                {key: LEC_mass[key][i] for key in LEC_mass},
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
        "fit_chi2": chi2,
    }

    # 保存数据到.npy文件
    np.save(
        f"{outdir}/{baryon}_{data_name}_data_addCa_{additive_Ca}_{error_source}.npy",
        data,
    )

    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(4, 3), dpi=140)
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
            **errorbar_kwargs,
        )
    ax.plot(data["fit_a_2"], data["fit_fcn"], color="grey", linestyle="--")
    ax.fill_between(
        data["fit_a_2"],
        data["fit_fcn"] - data["fit_fcn_err"],
        data["fit_fcn"] + data["fit_fcn_err"],
        alpha=0.5,
        edgecolor="none",
        color="grey",
    )

    # Labels and legends
    ax.set_xlim([-0.0003, 0.013])
    # ax.set_ylim([0.0,1.20])
    ax.set_xlabel(r"$a^2 (\mathrm{fm}^2)$", labelpad=3)
    ax.set_ylabel(ylabel + "(GeV)", labelpad=16)
    ax.legend(loc="best", frameon=False, ncol=3)
    # ax.tick_params(axis='both', which='major', direction='in', width=3, length=12)
    ax.minorticks_on()
    # ax.tick_params(axis="both", which="minor", direction="in", width=3, length=8)
    plt.xticks()
    plt.yticks()
    quark_content = str(plot_configurations[baryon]["quark_content"])
    mass = str(plot_configurations[baryon]["mass"])
    spin_parity = str(plot_configurations[baryon]["spin_parity"])
    baryon_lat = plot_configurations[baryon]["label_y"].replace("(GeV)", "")
    # 添加带框的注释
    # textstr = f"{(plot_configurations[baryon]['label_y']).replace('(GeV)', '')} ({mass})  {spin_parity} {quark_content}"
    # trace_anomaly=gv.gvar(np.mean(LEC['M_phys']),np.std(LEC['M_phys'],ddof=1))
    textstr = f"{baryon_lat} ({mass}) {spin_parity} {quark_content} "
    print(textstr)
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    props = dict(boxstyle="square", facecolor="white", alpha=0.3)
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    # 显示图表
    # plt.show()
    # plt.savefig(f"./pdfs/{baryon}_{data_name}_addCa_{additive_Ca}.pdf")


# exit()
if baryon not in charm_valance_light:
    baryon_mass =m_baryon_v_etas069_Ds_phys
    Hm = Hms + Hmc +Hml
    Hm += LEC_mass["C1"] * m_pi_phys**2 + LEC_mass["C2"] * etas_phys**2
    Hm_data_fit_and_plot(
        "Hm", Hm[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m\, $"
    )

    Ha = baryon_mass - Hm
    Hm_data_fit_and_plot(
        "Ha", Ha[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_a\, $"
    )

    Hag = baryon_mass - 1.295 * Hm
    Hm_data_fit_and_plot(
        "Hag", Hag[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_a^g\, $"
    )

    # Hm_data_fit_and_plot(
    #     "Hmlv",
    #     Hml[np.arange(len(selected_ensemble_indices)) * 3, :],
    #     r"$H_{mv}^l\, $",
    #     True,
    # )

    Hmls = LEC_mass["C1"] * (m_pi_phys**2 * np.ones_like(Hm))
    Hm_data_fit_and_plot(
        "Hmls", Hmls[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_{ms}^l\, $"
    )

    Hml += Hmls
    Hm_data_fit_and_plot(
        "Hml", Hml[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m^l\, $", True
    )
else:
    print('SGGGGGGG', Hms.shape,Hmc.shape,Hm_light_bt.shape)
    Hm = Hms + Hmc +Hm_light_bt
    Hm_data_fit_and_plot(
        "Hm", Hm[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m\, $"
    )

    Ha = baryon_mass_bt - Hm
    Hm_data_fit_and_plot(
        "Ha", Ha[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_a\, $"
    )

    Hag = baryon_mass_bt - 1.295 * Hm
    Hm_data_fit_and_plot(
        "Hag", Hag[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_a^g\, $"
    )

Hm_data_fit_and_plot(
    "Hmsv",
    Hms[np.arange(len(selected_ensemble_indices)) * 3, :],
    r"$H_{mv}^s\, $",
    True,
)

if baryon not in charm_valance_light:
    Hmss = LEC_mass["C2"] * etas_phys**2 * np.ones_like(Hms)
    Hm_data_fit_and_plot(
        "Hmss",
        Hmss[np.arange(len(selected_ensemble_indices)) * 3, :],
        r"$H_{ms}^s\, $",
        True,
    )

    Hms += Hmss
    Hm_data_fit_and_plot(
        "Hms", Hms[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m^s\, $", True
    )

Hm_data_fit_and_plot(
    "Hmc", Hmc[np.arange(len(selected_ensemble_indices)) * 3, :], r"$H_m^c\, $"
)
