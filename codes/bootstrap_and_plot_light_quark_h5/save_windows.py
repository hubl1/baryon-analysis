import json
import fcntl
import os
from collections import OrderedDict
from fasteners import InterProcessLock
from params.params import alttc_gv, fm2GeV
import gvar as gv
# fm2GeV = 0.197
alttc_arr=gv.mean(alttc_gv)
def save_window(ensemble, mass, T_strt, T_end, json_file):
    dir_name, file_name = os.path.split(json_file)
    lock_file = os.path.join(dir_name, '.'+file_name+'.lock')

    # Check if the file exists, if not create an empty JSON object
    if not os.path.exists(json_file):
        with open(json_file, "w") as file:
            json.dump({}, file)

    # Lock the file before reading
    with InterProcessLock(lock_file):
        with open(json_file, "r+") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

            if f"ensemble{ensemble}" not in data:
                data[f"ensemble{ensemble}"] = {}

            # Convert T_strt and T_end to int if they are numpy.int64
            T_strt = int(T_strt)
            T_end = int(T_end)

            data[f"ensemble{ensemble}"][f"mass{mass}"] = {
                "T_strt": T_strt,
                "T_end": T_end
            }

            # Sort the data dictionary
            # sorted_data = OrderedDict(sorted(data.items()))
            sorted_data = OrderedDict(sorted(data.items(), key=lambda x: int(x[0].replace("ensemble", ""))))
            for key in sorted_data:
                sorted_data[key] = OrderedDict(sorted(sorted_data[key].items()))

            # Move file position to the beginning before writing
            file.seek(0)
            file.truncate()

            json.dump(sorted_data, file, ensure_ascii=False, indent=4)

def save_data(ensemble, mass, T_strt, T_end, chi2, Q, json_file, fit_params):
    dir_name, file_name = os.path.split(json_file)
    lock_file = os.path.join(dir_name, '.'+file_name+'.lock')

    # Check if the file exists, if not create an empty JSON object
    if not os.path.exists(json_file):
        with open(json_file, "w") as file:
            json.dump({}, file)

    # Lock the file before reading
    with InterProcessLock(lock_file):
        with open(json_file, "r+") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

            if f"ensemble{ensemble}" not in data:
                data[f"ensemble{ensemble}"] = {}

            # Convert T_strt and T_end to int if they are numpy.int64
            T_strt = int(T_strt)
            T_end = int(T_end)
            if mass>=6: 
                data[f"ensemble{ensemble}"][f"mass{mass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mR": str(fit_params["mR"]*fm2GeV/alttc_arr[ensemble]),
                    "Zwp": str(fit_params["Zwp"]*(fm2GeV/alttc_arr[ensemble])**4),
                    "mPS": str(fit_params["mPS"]*fm2GeV/alttc_arr[ensemble]),
                    "fpi": str(fit_params["fpi"]*fm2GeV/alttc_arr[ensemble]),
                    "c1": str(fit_params["c1"]),
                    "c2": str(fit_params["c2"]),
                    "DeltaE": str(fit_params["DeltaE"]*fm2GeV/alttc_arr[ensemble]),
                    "Zwp2": str(fit_params["Zwp2"]*(fm2GeV/alttc_arr[ensemble])**4)
                }
            else:
                data[f"ensemble{ensemble}"][f"mass{mass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mR": str(fit_params["mR"]*fm2GeV/alttc_arr[ensemble]),
                    "Zwp": str(fit_params["Zwp"]*(fm2GeV/alttc_arr[ensemble])**4),
                    "mPS": str(fit_params["mPS"]*fm2GeV/alttc_arr[ensemble]),
                    "fpi": str(fit_params["fpi"]*fm2GeV/alttc_arr[ensemble])
                }

            # Sort the data dictionary
            sorted_data = OrderedDict(sorted(data.items()))
            for key in sorted_data:
                sorted_data[key] = OrderedDict(sorted(sorted_data[key].items()))

            # Move file position to the beginning before writing
            file.seek(0)
            file.truncate()

            json.dump(sorted_data, file, ensure_ascii=False, indent=4)


class DataNotFoundError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        ensemble -- input ensemble which caused the error
        mass -- input mass which caused the error
    """

    def __init__(self, ensemble, mass):
        self.ensemble = ensemble
        self.mass = mass
        self.message = f"No expected data found for ensemble {ensemble} and mass group {int(mass/3)}"
        super().__init__(self.message)


def read_window(ensemble, mass, json_file):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    if f"ensemble{ensemble}" in data and f"mass{mass}" in data[f"ensemble{ensemble}"]:
        T_strt = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_strt"]
        T_end = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_end"]
        return T_strt, T_end
    else:
        # return None, None
        raise DataNotFoundError(ensemble, mass)

def read_ini(ensemble, mass, json_file):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    if f"ensemble{ensemble}" in data and f"mass{mass}" in data[f"ensemble{ensemble}"]:
        T_strt = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_strt"]
        T_end = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_end"]
        mR = data[f"ensemble{ensemble}"][f"mass{mass}"]["mR"]
        Zwp = data[f"ensemble{ensemble}"][f"mass{mass}"]["Zwp"]
        mPS = data[f"ensemble{ensemble}"][f"mass{mass}"]["mPS"]
        fpi = data[f"ensemble{ensemble}"][f"mass{mass}"]["fpi"]
        # c2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c2"]
        # mp2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp2"]
        # return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean, 'c2': gv.gvar(c2).mean, 'mp2': gv.gvar(mp2).mean}  
        return T_strt, T_end, {'mR': gv.gvar(mR).mean/fm2GeV*alttc_arr[ensemble],'Zwp': gv.gvar(Zwp).mean/(fm2GeV/alttc_arr[ensemble])**4, 'mPS': gv.gvar(mPS).mean/fm2GeV*alttc_arr[ensemble],'fpi': gv.gvar(fpi).mean/fm2GeV*alttc_arr[ensemble]}  

    else:
        # return None, None, None
        raise DataNotFoundError(ensemble, mass)


def read_ylimit(ensemble, mass, json_file):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    if f"ensemble{ensemble}" in data and f"mass_group_{int(mass/3)}" in data[f"ensemble{ensemble}"]:
        y_lim_strt = data[f"ensemble{ensemble}"][f"mass_group_{int(mass/3)}"]["y_lim_strt"]
        y_lim_end = data[f"ensemble{ensemble}"][f"mass_group_{int(mass/3)}"]["y_lim_end"]
        return y_lim_strt, y_lim_end
    else:
        raise DataNotFoundError(ensemble, mass)

