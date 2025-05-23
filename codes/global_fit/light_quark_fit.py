import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import os
from params.params import name , a_fm_independent
from params.plt_labels import plot_configurations
plt.style.use("utils/physrev.mplstyle")

def fit_light_quark_like_proton(baryon,combined_data, m_pi_v_bt, m_pi_sea_bt, metas_sea_bt, alttc_samples, selected_ensemble_indices, ns, error_source, additive_Ca, lqcd_data_path):
    cov_enabled = True
    combined_data = combined_data
    combined_data_cntrl = np.mean(combined_data, axis=1)  # jack-knife mean
    combined_data_err = np.std(combined_data, axis=1, ddof=1)  # jack-knife mean
    combined_data_cov = np.cov(combined_data)

    if cov_enabled == True:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_cov)}
    else:
        data_dctnry = {"combined": gv.gvar(combined_data_cntrl, combined_data_err)}

    tsep_dctnry = {"combined": np.arange(len(selected_ensemble_indices) * 3)}

    # Assuming m_pi_phys is the physical pion mass
    m_pi_phys = 0.135  # Adjust this value to the correct one
    ms_phys = 0.0922  # Adjust this value to the correct one
    fpi_phys = 0.122  # Adjust this value to the correct one
    etas_phys = 0.68963
    # CasQ=False
    CasQ=not additive_Ca
    Ca4Q=True
    # error_source=None
    # additive_Ca=True
    ml_col_indices = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    ms_col_indices = np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1]) + 3
    ml_indices = np.array(
        [i * 3 + ml_col_indices[index] for i, index in enumerate(selected_ensemble_indices)]
    ).repeat(3)
    alttc_index = np.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 1, 3, 5, 2])
    alttc_gv_new = a_fm_independent[[alttc_index[idx] for idx in selected_ensemble_indices]]
    



    def vol_a_mdls(idct, p):
        mdls = {}
        i = idct["i"]
        m_pi_v = m_pi_v_bt[:, i]
        m_pi_sea = m_pi_sea_bt[:, i]
        metas_sea = metas_sea_bt[:, i]

        a = alttc_samples[:, i]
        L = np.array(ns).repeat(3) * a
        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)

        M0 = p["M0"]

        # Cv3 = p["Cv3"]
        # Cpq3 = p["Cpq3"]

        M_corr = (
            +p["Cv2"] * (m_pi_v**2 - m_pi_phys**2)
            + p["Cpq2"] * (m_pi_pq**2 - m_pi_phys**2)
            # + Cv3 * (m_pi_v**3 - m_pi_phys**3)
            # + Cpq3 * (m_pi_pq**3 - m_pi_phys**3)
        )

        M_corr_s = p["C5"] * (metas_sea**2 - etas_phys**2)
        M_s = M_corr_s + M_corr
        M_s *= (1 + p["Cas"] * a**2) if CasQ else 1

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
    param_ranges = {
        "M0": (0.5, 1.5),
        "Cv2": (0, 1),
        "Cpq2": (0, 1),
        "C4": (-1, 1),
        "Ca2": (-1.5, 1.5),
        "C5": (-1, 1),
    }
    # if baryon == "PROTON":
    #     param_ranges["gA"] = (0.1, 1.5)
    #     param_ranges["g1"] = (-1, 1)
    # else:
    #     param_ranges["Cv3"] = (0, 1.5)
    #     param_ranges["Cpq3"] = (-1, 1)
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
    print(f"M at the physical point is: {M_phys} GeV")

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
        # if baryon == "PROTON":
        #     return_dict["g1"] = fit.p["g1"].mean
        #     return_dict["gA"] = fit.p["gA"].mean
        # else:
        #     return_dict["Cv3"] = fit.p["Cv3"].mean
        #     return_dict["Cpq3"] = fit.p["Cpq3"].mean
        # if not additive_Ca:
        #     return_dict["Ca20"] = fit.p["Ca20"].mean
        # return_dict["Ca2s"] = fit.p["Ca2s"].mean
        if CasQ:
            return_dict["Cas"] = fit.p["Cas"].mean
        if Ca4Q:
            return_dict["Ca4"] = fit.p["Ca4"].mean

        return return_dict

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
        # if baryon == "PROTON":
        #     LEC_temp["g1"] = np.array([res["g1"] for res in results])
        #     LEC_temp["gA"] = np.array([res["gA"] for res in results])
        # else:
        #     LEC_temp["Cv3"] = np.array([res["Cv3"] for res in results])
        #     LEC_temp["Cpq3"] = np.array([res["Cpq3"] for res in results])

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
    # print(LEC_mass)

    # 创建一个新的字典来存储结果
    LEC_gvar = {}


    def derivatives_Mcorr(m_pi_v, m_pi_sea, m_pi_phys, p):
        # Compute m_pi_pq
        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        # Cv3 = p["Cv3"]
        # Cpq3 = p["Cpq3"]

        # Compute partial derivatives of m_pi_pq
        dm_pi_pq_d_m_pi_v = m_pi_v / np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        dm_pi_pq_d_m_pi_sea = m_pi_sea / np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)

        # Compute derivatives of M_corr
        dM_d_m_pi_v = (
            2 * p["Cv2"] * m_pi_v
            + 2 * p["Cpq2"] * m_pi_pq * dm_pi_pq_d_m_pi_v
            # + 3 * Cv3 * m_pi_v**2
            # + 3 * Cpq3 * m_pi_pq**2 * dm_pi_pq_d_m_pi_v
        )

        dM_d_m_pi_sea = (
            2 * p["Cpq2"] * m_pi_pq * dm_pi_pq_d_m_pi_sea
        #     + 3 * Cpq3 * m_pi_pq**2 * dm_pi_pq_d_m_pi_sea
        )

        return dM_d_m_pi_v, dM_d_m_pi_sea


    Hml_sea_bt = np.zeros(n_bootstrap)
    Hml_v_bt = np.zeros(n_bootstrap)
    Hms_sea_bt = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        p = {key: LEC_mass[key][i] for key in LEC_mass}
        # Cv3 = p["Cv3"]
        # Cpq3 = p["Cpq3"]

        dM_d_m_pi_v, dM_d_m_pi_sea = derivatives_Mcorr(
            m_pi_phys, m_pi_phys, m_pi_phys, p
        )
        Hml_sea_bt[i] = m_pi_phys / 2 * dM_d_m_pi_sea
        Hml_v_bt[i] = m_pi_phys / 2 * dM_d_m_pi_v
        Hms_sea_bt[i] = etas_phys**2*p['C5']

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
    data[baryon]["Hms_sea"] = str(gv.gvar(np.mean(Hms_sea_bt), np.std(Hms_sea_bt)))

    # # 将更新后的字典保存回json文件
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
    data_Hms_s = {
        "fit_fcn": [np.mean(Hms_sea_bt)],
        "fit_fcn_err": [np.std(Hms_sea_bt)],
        "fit_chi2": fit_chi2,
    }
    Hm_light_bt=Hms_sea_bt+Hml_v_bt+Hml_sea_bt

    np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_Hmlv_data_addCa_{not CasQ}_{error_source}.npy", data_Hml_v)
    np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_Hmls_data_addCa_{not CasQ}_{error_source}.npy", data_Hml_s)
    np.save(f"{lqcd_data_path}/pydata_global_fit/{baryon}_Hmss_data_addCa_{not CasQ}_{error_source}.npy", data_Hms_s)

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
        print(f"{param}: {gvar}")

    # print(LEC_gvar[''])
    # pp = LEC_gvar

#TO BE UPDATED
    a = gv.mean(alttc_gv_new).repeat(3)
    L = np.array(ns).repeat(3) * a


    def compute_pq_corr(pp, m_pi_sea, m_pi_v, metas_sea, alttc_loc=0):

        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        # Cpq3 = pp["Cpq3"]
        term1 = (
            -pp["Cpq2"] * m_pi_pq**2
            + pp["Cpq2"] * m_pi_v**2
            # - Cpq3 * m_pi_pq**3
            # + Cpq3 * m_pi_v**3
        ) * ((1 + pp["Cas"] * a**2) if CasQ else 1)

        # common_factor = np.pi / (3 * (4 * np.pi * fpi) ** 2)

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
        m_pi_pq = np.sqrt(m_pi_v**2 + m_pi_sea**2) / np.sqrt(2)
        # Cpq3 = pp["Cpq3"]
        # Cv3 = pp["Cv3"]
        term1 = (
            -pp["Cv2"] * m_pi_v**2
            + pp["Cv2"] * m_pi_phys**2
            - pp["Cpq2"] * m_pi_pq**2
            + pp["Cpq2"] * m_pi_phys**2
            # - Cv3 * m_pi_v**3
            # + Cv3 * m_pi_phys**3
            # - Cpq3 * m_pi_pq**3
            # + Cpq3 * m_pi_phys**3
        ) * ((1 + pp["Cas"] * a**2) if CasQ else 1)

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
            for i in range(n_bootstrap)
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
            for i in range(n_bootstrap)
        ]
    ).T

    mp_v_bt_etas069_phys = combined_data + mp_v_bt_etas069_corr
    mp_v_bt_etas069_uni = combined_data + mp_v_bt_etas069_pq_corr
    # print('aaaaaaaa',mp_v_bt_etas069[:,0],mp_v_bt_etas069_corr[:,0],mp_v_bt_etas069_phys[:,0])

    # print(combined_data_corr)
    combined_data_physical_cntrl = np.mean(mp_v_bt_etas069_phys, axis=1)
    combined_data_physical_err = np.std(mp_v_bt_etas069_phys, axis=1)
    # Plot
    combined_data_cntrl = np.mean(mp_v_bt_etas069_uni, axis=1)
    combined_data_err = np.std(mp_v_bt_etas069_uni, axis=1)
    # combined_data_uncorrected_cntrl = np.mean(mp_v_bt_etas069, axis=1)
    # combined_data_uncorrected_err = np.std(mp_v_bt_etas069, axis=1)

    # combined_data_cntrl = combined_data_physical_cntrl
    m_pi_v_mean_square = np.mean(m_pi_v_bt**2, axis=1)
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
    alttc_index = np.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 1, 3, 5, 2])
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




    def compute_M_proton(m_pi_square, a, LEC, fpi):
        m_pi = np.sqrt(m_pi_square)
        # fpi=fpi_phys
        M_0 = LEC["M0"]
        # Cv3 = LEC["Cv3"]
        # Cpq3 = LEC["Cpq3"]
        M_corr = (
            LEC["Cv2"] * (m_pi**2 - m_pi_phys**2)
            + LEC["Cpq2"] * (m_pi**2 - m_pi_phys**2)
            # + Cv3 * (m_pi**3 - m_pi_phys**3)
            # + Cpq3 * (m_pi**3 - m_pi_phys**3)
        ) * ((1 + LEC["Cas"] * a**2) if CasQ else 1)

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

    # # 将结果写入 JSON 文件
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

        # np.save("./npy_error_budget/Hm_data_addCa_{not CasQ}_{error_source}.npy", slope_results)
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
        # np.savez(
        #     "./npy_error_budget/plot_data.npz",
        #     targets=targets,
        #     sigma_terms=sigma_terms,
        #     sigma_term_errors=sigma_term_errors,
        # )

        # print("Data saved to 'plot_data.npz'")
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

    if M_p_value > 0:
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
    plt.savefig(f"{lqcd_data_path}/figures/{baryon}_mass_addCa_{not CasQ}_{error_source}.pdf")
    # plt.show()
    return Hm_light_bt, LEC_mass['M0']
