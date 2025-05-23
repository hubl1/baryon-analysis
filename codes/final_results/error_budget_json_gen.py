import json
import os
import numpy as np
import sys
from pathlib import Path


d1s = -0.0294
d2s = -0.0089
d1s_err = 0.0017
d2s_err = 0.0010

d1c = -0.166
d2c = -0.0375
d1c_err = 0.012
d2c_err = 0.0042

# d1c=-0.090
# d2c=-0.026
mpi_phys = 0.135
metas_phys = 0.69
# ms_phys = 0.098
# mc_phys = 1.095
ms_phys = 0.0956
mc_phys = 1.095

lqcd_data_path=sys.argv[1]
pydata_path=lqcd_data_path+'/precomputed_pydata_global_fit'

# 定义 JSON 文件路径
outdir = Path(lqcd_data_path) / "final_results"
outdir.mkdir(parents=True, exist_ok=True)  # 自动创建目录
file_path = f"{lqcd_data_path}/final_results/data_error_budget.json"

# 检查文件是否存在，读取现有内容，否则初始化为空字典
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
else:
    data = {}

# 处理的粒子和物理量
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
# baryons=['LAMBDA_C', 'OMEGA_CCC', 'XI_C']
charmed_baryons = [
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
    "D_S",
    "DS_STAR",
    "ETA_C",
    "JPSI",
    "CHI_C0",
    "CHI_C1",
    "H_C",
]

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
filtered_baryons = [
    baryon for baryon in charmed_baryons if baryon not in charm_valance_light
]
# baryons = ["LAMBDA_C", "SIGMA_C", "XI_C", "OMEGA_C"]
quantities = [ "Hml","mass", "Hm", "Hmsv", "Hmss", "Hms", "Hmc", "Ha", "Hag"]
light_quantities = [
    "mass",
    "Hm",
    "Hml",
    "Hmlv",
    "Hmls",
    "Hms",
    "Hmsv",
    "Hmss",
    "Ha",
    "Hag",
]
charm_valance_light_quantities= [
    "mass",
    "Hm",
    "Hml",
    "Hmlv",
    "Hmls",
    "Hms",
    "Hmsv",
    "Hmss",
    "Hmc",
    "Ha",
    "Hag",
]
stat_error_sources = ["mpi", "metas", "alttc_stat"]
sys_error_sources = ["D_S", "alttc_sys"]
light_sys_error_sources = [
    "alttc_sys",
]

for baryon in light_baryons:
    data_Hmsv = np.load(
        f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
        allow_pickle=True,
    ).item()

    value_Hmsv = data_Hmsv["fit_fcn"][0]

    l_corr_1 = -d1s * value_Hmsv * mpi_phys**2 / ms_phys
    # print(baryon,l_corr_1)
    l_corr_1_err = ((d1s_err * value_Hmsv * mpi_phys**2 / ms_phys) ** 2) ** 0.5
    s_corr_1 = -d2s * value_Hmsv * metas_phys**2 / ms_phys
    s_corr_1_err = ((d2s_err * value_Hmsv * metas_phys**2 / ms_phys) ** 2) ** 0.5

    if baryon not in data:
        data[baryon] = {}

    for quantity in light_quantities:
        if quantity not in data[baryon]:
            data[baryon][quantity] = {}
        if "Hm_corr" not in data[baryon]:
            data[baryon]["Hm_corr"] = {}

        if quantity not in ["Ha", "Hag", "Hml", "Hmsv"]:
            cntrl = np.load(
                f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                allow_pickle=True,
            ).item()["fit_fcn"][0]
            if baryon == 'DELTA':
                print('DELTA',quantity, cntrl)
        elif quantity == "Ha":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                - np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )
        elif quantity == "Hml":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                + np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )
        elif quantity == "Hmsv":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                - np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )
        elif quantity == "Hag":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                - 1.295
                * np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )

        # print(baryon, cntrl)

        if quantity == "Hml":
            cntrl += l_corr_1
        elif quantity == "Hmls":
            cntrl += l_corr_1
        elif quantity == "Hms":
            cntrl += s_corr_1
        elif quantity == "Hmss":
            cntrl += s_corr_1
        elif quantity == "Hm":
            cntrl += s_corr_1 + l_corr_1
        elif quantity == "Ha":
            cntrl -= s_corr_1 + l_corr_1
        elif quantity == "Hag":
            cntrl -= 1.295 * (s_corr_1 + l_corr_1)

        # data[baryon][quantity]["cntrl"] = f"{cntrl:.6f}"
        data[baryon][quantity]["cntrl"] = f"{cntrl:.6f}" if abs(cntrl) > 1e-7 else "0"
        if quantity == "Hm":
            data[baryon]["Hm_corr"]["cntrl"] = (
                f"{ s_corr_1 + l_corr_1:.6f}"
                if abs(s_corr_1 + l_corr_1) > 1e-7
                else "0"
            )

        chi2_mass = np.load(
            f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hm = np.load(
            f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmlv = np.load(
            f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmls = np.load(
            f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmsv = np.load(
            f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hms = np.load(
            f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]

        chi2_list = [chi2_mass, chi2_Hm, chi2_Hmlv, chi2_Hmls, chi2_Hmsv, chi2_Hms]
        chi2_list = [max(1, chi2) for chi2 in chi2_list]
        chi2_mass, chi2_Hm, chi2_Hmlv, chi2_Hmls, chi2_Hmsv, chi2_Hms = chi2_list

        # print(chi2_Hms)

        for error_source in stat_error_sources:
            # 读取数据文件
            if quantity == "Ha":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = max(
                    abs(
                        (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                    ),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_mass
                        + data3["fit_fcn_err"][0] ** 2 * chi2_Hm
                        - data2["fit_fcn_err"][0] ** 2 * chi2_mass
                        - data4["fit_fcn_err"][0] ** 2 * chi2_Hm
                    )
                    ** 0.5,
                )
            elif quantity == "Hml":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                # error = abs(
                #         (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                #         - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                #     )
                error = max(
                    abs(
                        (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                    ),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_Hmlv
                        + data3["fit_fcn_err"][0] ** 2 * chi2_Hmls
                        - data2["fit_fcn_err"][0] ** 2 * chi2_Hmlv
                        - data4["fit_fcn_err"][0] ** 2 * chi2_Hmls
                    )
                    ** 0.5,
                )
            elif quantity == "Hmss":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = max(
                    abs(
                        (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                    ),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_Hms
                        + data3["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                        - data2["fit_fcn_err"][0] ** 2 * chi2_Hms
                        - data4["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                    )
                    ** 0.5,
                )
            elif quantity == "Hag":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = max(
                    abs(
                        (data1["fit_fcn"][0] - 1.295 * data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] - 1.295 * data4["fit_fcn"][0])
                    ),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_mass
                        + (1.295 * data3["fit_fcn_err"][0]) ** 2 * chi2_Hm
                        - data2["fit_fcn_err"][0] ** 2 * chi2_mass
                        - (1.295 * data4["fit_fcn_err"][0]) ** 2 * chi2_Hm
                    )
                    ** 0.5,
                )
            else:
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                chi2_quantity = max(data1["fit_chi2"], 1)

                # 计算对比值
                error = max(
                    abs(data1["fit_fcn"][0] - data2["fit_fcn"][0]),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_quantity
                        - data2["fit_fcn_err"][0] ** 2 * chi2_quantity
                    )
                    ** 0.5,
                )

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in light_sys_error_sources:
            if quantity == "Ha":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "Hml":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                )
            elif quantity == "Hmss":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "Hag":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - 1.295 * data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - 1.295 * data4["fit_fcn"][0])
                )
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # print("Error:", error)
            # print("Error type:", type(error))
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["total stat"]:
            # 读取数据文件
            if quantity == "Ha":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_mass
                    + data3["fit_fcn_err"][0] ** 2 * chi2_Hm
                ) ** 0.5
            elif quantity == "Hml":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_Hmlv
                    + data3["fit_fcn_err"][0] ** 2 * chi2_Hmls
                ) ** 0.5
            elif quantity == "Hmss":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_Hms
                    + data3["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                ) ** 0.5
                # print(
                #     data1["fit_fcn_err"][0],
                #     data3["fit_fcn_err"][0],
                #     chi2_Hms,
                #     chi2_Hmsv,
                # )
                # print(baryon, error)
            elif quantity == "Hag":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_mass
                    + 1.295**2 * data3["fit_fcn_err"][0] ** 2 * chi2_Hm
                ) ** 0.5
            else:
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                chi2_quantity = max(data1["fit_chi2"], 1)

                # 计算对比值
                error = data1["fit_fcn_err"][0] * chi2_quantity**0.5
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["fit ansatz"]:
            if quantity == "Ha":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "Hml":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmlv_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmls_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                )
            elif quantity == "Hmss":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "Hag":
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hm_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(
                    (data1["fit_fcn"][0] - 1.295 * data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - 1.295 * data4["fit_fcn"][0])
                )
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )
        if quantity == "Hml":
            data[baryon][quantity]["correction"] = f"{l_corr_1_err:.6f}"
        if quantity == "Hmls":
            data[baryon][quantity]["correction"] = f"{l_corr_1_err:.6f}"
        if quantity == "Hms":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hmss":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hm":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Ha":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Hag":
            data[baryon][quantity][
                "correction"
            ] = f"{1.295*(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"

# # 遍历所有粒子，确保字典结构正确
for baryon in filtered_baryons:
    data_Hmsv = np.load(
        f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy", allow_pickle=True
    ).item()
    data_Hmc = np.load(
        f"{pydata_path}/{baryon}_Hmc_data_addCa_True_None.npy", allow_pickle=True
    ).item()

    value_Hmsv = data_Hmsv["fit_fcn"][0] if data_Hmsv["fit_fcn"][0] > 0.00001 else 0
    value_Hmc = data_Hmc["fit_fcn"][0] if data_Hmc["fit_fcn"][0] > 0.00001 else 0

    l_corr_1 = (
        -d1s * value_Hmsv * mpi_phys**2 / ms_phys
        - d1c * value_Hmc * mpi_phys**2 / mc_phys
    )
    l_corr_1_err = (
        (d1s_err * value_Hmsv * mpi_phys**2 / ms_phys) ** 2
        + (d1c_err * value_Hmc * mpi_phys**2 / mc_phys) ** 2
    ) ** 0.5
    s_corr_1 = (
        -d2s * value_Hmsv * metas_phys**2 / ms_phys
        - d2c * value_Hmc * metas_phys**2 / mc_phys
    )
    s_corr_1_err = (
        (d2s_err * value_Hmsv * metas_phys**2 / ms_phys) ** 2
        + (d2c_err * value_Hmc * metas_phys**2 / mc_phys) ** 2
    ) ** 0.5

    if baryon not in data:
        data[baryon] = {}

    # if "Hm_corr" not in data[baryon]:
    data[baryon]['Hmlv'] = {}
    data[baryon]['Hmls'] = {}
    data[baryon]["Hm_corr"] = {}

    for quantity in quantities:
        if quantity not in data[baryon]:
            data[baryon][quantity] = {}

        if quantity == "Hmss":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                - np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )
            if baryon == 'D_S':
                print('D_S Hmss',cntrl)
        elif quantity == "mass":
            cntrl = np.load(
                f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy",
                allow_pickle=True,
            ).item()["fit_fcn"][0]
        else:
            cntrl = np.load(
                f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                allow_pickle=True,
            ).item()["fit_fcn"][0]

        if quantity == "Hml":
            cntrl += l_corr_1
        elif quantity == "Hms":
            cntrl += s_corr_1
        elif quantity == "Hmss":
            cntrl += s_corr_1
        elif quantity == "Hm":
            cntrl += s_corr_1 + l_corr_1
        elif quantity == "Ha":
            cntrl -= s_corr_1 + l_corr_1
        elif quantity == "Hag":
            cntrl -= 1.295 * (s_corr_1 + l_corr_1)

        if abs(cntrl) < 1e-7:
            print(quantity, cntrl)
        data[baryon][quantity]["cntrl"] = f"{cntrl:.6f}" if abs(cntrl) > 1e-7 else "0"
        if quantity == "Hm":
            data[baryon]["Hm_corr"]["cntrl"] = (
                f"{ s_corr_1 + l_corr_1:.6f}"
                if abs(s_corr_1 + l_corr_1) > 1e-7
                else "0"
            )

        chi2_Hms = np.load(
            f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmsv = np.load(
            f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hms = max(chi2_Hms, 1)
        chi2_Hmsv = max(chi2_Hmsv, 1)

        for error_source in stat_error_sources:
            if quantity == "Hmss":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = max(
                    abs(
                        (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                    ),
                    0*abs(
                        (
                            data1["fit_fcn_err"][0] ** 2 * chi2_Hms
                            + data3["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                        )
                        - (
                            data2["fit_fcn_err"][0] ** 2 * chi2_Hms
                            + data4["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                        )
                    )
                    ** 0.5,
                )
                if baryon == 'D_S':
                    print(f'D_S Hmss {error_source}', error)
            elif quantity == "mass":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                # print(data1)
                chi2_quantity = max(data1["fit_chi2"], 1)

                # 计算对比值
                error = max(
                    abs(data1["fit_fcn"][0] - data2["fit_fcn"][0]),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_quantity
                        - data2["fit_fcn_err"][0] ** 2 * chi2_quantity
                    )
                    ** 0.5,
                )
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                # print(data1)
                chi2_quantity = max(data1["fit_chi2"], 1)

                # 计算对比值
                error = max(
                    abs(data1["fit_fcn"][0] - data2["fit_fcn"][0]),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_quantity
                        - data2["fit_fcn_err"][0] ** 2 * chi2_quantity
                    )
                    ** 0.5,
                )
                if baryon == 'OMEGA_CCC' and quantity == 'Hmc':
                    print('-------------',error_source, error)

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in sys_error_sources:
            if quantity == "Hmss":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "mass":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["total stat"]:
            if quantity == "Hmss":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_Hms
                    + data2["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                ) ** 0.5 * (0 if baryon == 'D_S' else 1)
            elif quantity == "mass":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = data1["fit_fcn_err"][0] * chi2_quantity**0.5
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = data1["fit_fcn_err"][0] * chi2_quantity**0.5

            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["fit ansatz"]:
            if quantity == "Hmss":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hms_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                error = abs(
                    (data1["fit_fcn"][0] - data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] - data4["fit_fcn"][0])
                )
            elif quantity == "mass":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_mass_uni_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        if quantity == "Hml":
            data[baryon][quantity]["correction"] = f"{l_corr_1_err:.6f}"
        if quantity == "Hms":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hmss":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hm":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Ha":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Hag":
            data[baryon][quantity][
                "correction"
            ] = f"{1.295*(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"

    data[baryon]['Hmlv']["cntrl"] = "0"
    data[baryon]['Hmls']["cntrl"] = data[baryon]['Hml']["cntrl"]
    data[baryon]['Hmls']["correction"] = data[baryon]['Hml']["correction"]
    for error_source in [*stat_error_sources, *sys_error_sources, 'total stat', 'fit ansatz']:
        data[baryon]['Hmlv'][error_source] = "0"
        data[baryon]['Hmls'][error_source] = data[baryon]['Hml'][error_source]

for baryon in charm_valance_light:
    data_Hmsv = np.load(
        f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy", allow_pickle=True
    ).item()
    data_Hmc = np.load(
        f"{pydata_path}/{baryon}_Hmc_data_addCa_True_None.npy", allow_pickle=True
    ).item()

    value_Hmsv = data_Hmsv["fit_fcn"][0] if data_Hmsv["fit_fcn"][0] > 0.00001 else 0
    value_Hmc = data_Hmc["fit_fcn"][0] if data_Hmc["fit_fcn"][0] > 0.00001 else 0

    l_corr_1 = (
        -d1s * value_Hmsv * mpi_phys**2 / ms_phys
        - d1c * value_Hmc * mpi_phys**2 / mc_phys
    )
    l_corr_1_err = (
        (d1s_err * value_Hmsv * mpi_phys**2 / ms_phys) ** 2
        + (d1c_err * value_Hmc * mpi_phys**2 / mc_phys) ** 2
    ) ** 0.5
    s_corr_1 = (
        -d2s * value_Hmsv * metas_phys**2 / ms_phys
        - d2c * value_Hmc * metas_phys**2 / mc_phys
    )
    s_corr_1_err = (
        (d2s_err * value_Hmsv * metas_phys**2 / ms_phys) ** 2
        + (d2c_err * value_Hmc * metas_phys**2 / mc_phys) ** 2
    ) ** 0.5

    if baryon not in data:
        data[baryon] = {}

    data[baryon]["Hm_corr"] = {}

    for quantity in charm_valance_light_quantities:
        if quantity not in data[baryon]:
            data[baryon][quantity] = {}

        if quantity == "Hms":
            cntrl = (
                np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
                + np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()["fit_fcn"][0]
            )
        else:
            cntrl = np.load(
                f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                allow_pickle=True,
            ).item()["fit_fcn"][0]

        if quantity == "Hml":
            cntrl += l_corr_1
        elif quantity == "Hmls":
            cntrl += l_corr_1
        elif quantity == "Hms":
            cntrl += s_corr_1
        elif quantity == "Hmss":
            cntrl += s_corr_1
        elif quantity == "Hm":
            cntrl += s_corr_1 + l_corr_1
        elif quantity == "Ha":
            cntrl -= s_corr_1 + l_corr_1
        elif quantity == "Hag":
            cntrl -= 1.295 * (s_corr_1 + l_corr_1)

        if abs(cntrl) < 1e-7:
            print(quantity, cntrl)
        data[baryon][quantity]["cntrl"] = f"{cntrl:.6f}" if abs(cntrl) > 1e-7 else "0"
        if quantity == "Hm":
            data[baryon]["Hm_corr"]["cntrl"] = (
                f"{ s_corr_1 + l_corr_1:.6f}"
                if abs(s_corr_1 + l_corr_1) > 1e-7
                else "0"
            )

        chi2_Hmss = np.load(
            f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmsv = np.load(
            f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
            allow_pickle=True,
        ).item()["fit_chi2"]
        chi2_Hmss = max(chi2_Hmss, 1)
        chi2_Hmsv = max(chi2_Hmsv, 1)

        for error_source in stat_error_sources:
            if quantity == "Hms":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = max(
                    abs(
                        (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                        - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                    ),
                    0*abs(
                        (
                            data1["fit_fcn_err"][0] ** 2 * chi2_Hmss
                            + data3["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                        )
                        - (
                            data2["fit_fcn_err"][0] ** 2 * chi2_Hmss
                            + data4["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                        )
                    )
                    ** 0.5,
                )
                # if baryon == 'D_S':
                #     print(f'D_S Hmss {error_source}', error)
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                # print(data1)
                chi2_quantity = max(data1["fit_chi2"], 1)

                # 计算对比值
                error = max(
                    abs(data1["fit_fcn"][0] - data2["fit_fcn"][0]),
                    abs(
                        data1["fit_fcn_err"][0] ** 2 * chi2_quantity
                        - data2["fit_fcn_err"][0] ** 2 * chi2_quantity
                    )
                    ** 0.5,
                )

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in sys_error_sources:
            if quantity == "Hms":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                error = abs(
                    (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                )
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_{error_source}.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["total stat"]:
            if quantity == "Hms":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = (
                    data1["fit_fcn_err"][0] ** 2 * chi2_Hmss
                    + data2["fit_fcn_err"][0] ** 2 * chi2_Hmsv
                ) ** 0.5 * (0 if baryon == 'D_S' else 1)
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()

                # 计算对比值
                error = data1["fit_fcn_err"][0] * chi2_quantity**0.5

            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        for error_source in ["fit ansatz"]:
            if quantity == "Hms":
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_Hmss_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                data3 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data4 = np.load(
                    f"{pydata_path}/{baryon}_Hmsv_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()

                error = abs(
                    (data1["fit_fcn"][0] + data3["fit_fcn"][0])
                    - (data2["fit_fcn"][0] + data4["fit_fcn"][0])
                )
            else:
                # 读取数据文件
                data1 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_True_None.npy",
                    allow_pickle=True,
                ).item()
                data2 = np.load(
                    f"{pydata_path}/{baryon}_{quantity}_data_addCa_False_None.npy",
                    allow_pickle=True,
                ).item()
                # 计算对比值
                error = abs(data1["fit_fcn"][0] - data2["fit_fcn"][0])

            # 存储计算结果
            # data[baryon][quantity][error_source] = f"{error:.6f}"
            data[baryon][quantity][error_source] = (
                f"{error:.6f}" if abs(error) > 1e-7 else "0"
            )

        if quantity == "Hml":
            data[baryon][quantity]["correction"] = f"{l_corr_1_err:.6f}"
        if quantity == "Hmls":
            data[baryon][quantity]["correction"] = f"{l_corr_1_err:.6f}"
        if quantity == "Hms":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hmss":
            data[baryon][quantity]["correction"] = f"{s_corr_1_err:.6f}"
        if quantity == "Hm":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Ha":
            data[baryon][quantity][
                "correction"
            ] = f"{(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"
        if quantity == "Hag":
            data[baryon][quantity][
                "correction"
            ] = f"{1.295*(s_corr_1_err**2+l_corr_1_err**2)**0.5:.6f}"

# 将数据写入 JSON 文件
with open(file_path, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4)
