import json
import fcntl
import os
from collections import OrderedDict
from fasteners import InterProcessLock
from params.params import alttc_gv, fm2GeV
import gvar as gv

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

def save_ini(ensemble, mass, T_strt, T_end, chi2, Q, json_file, fit_params, two_state_enabled):
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
            if not two_state_enabled:
                data[f"ensemble{ensemble}"][f"mass{mass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mp": str(fit_params["mp"]*fm2GeV/alttc_arr[ensemble]),
                    # "mp2": str(fit_params["mp2"]*fm2GeV/alttc_arr[mass]),
                    "c1": str(fit_params["c1"]),
                    # "c2": str(fit_params["c2"]),
                }
            else:
                data[f"ensemble{ensemble}"][f"mass{mass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mp": str(fit_params["mp"]*fm2GeV/alttc_arr[ensemble]),
                    "mp2": str(fit_params["mp2"]*fm2GeV/alttc_arr[ensemble]),
                    "c1": str(fit_params["c1"]),
                    "c2": str(fit_params["c2"]),
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

def read_ini(ensemble, mass, json_file, two_state_enabled):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    if f"ensemble{ensemble}" in data and f"mass{mass}" in data[f"ensemble{ensemble}"]:
        T_strt = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_strt"]
        T_end = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_end"]
        c1 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c1"]
        mp = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp"]
        if two_state_enabled:
            c2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c2"]
            mp2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp2"]
        if not two_state_enabled:
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble]}
        else: 
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble], 'c2': gv.gvar(c2).mean, 'mp2': gv.gvar(mp2).mean/fm2GeV*alttc_arr[ensemble]}

    else:
        # return None, None, None
        raise DataNotFoundError(ensemble, mass)

def read_ini_v2(ensemble, mass,cmass, json_file, two_state_enabled):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    if f"ensemble{ensemble}" in data and f"mass{cmass}" in data[f"ensemble{ensemble}"]:
        mass=cmass
    if f"ensemble{ensemble}" in data and f"mass{mass}" in data[f"ensemble{ensemble}"]:
        T_strt = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_strt"]
        T_end = data[f"ensemble{ensemble}"][f"mass{mass}"]["T_end"]
        c1 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c1"]
        mp = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp"]
        c2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c2"]
        mp2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp2"]
        if "two_state_enabled" in data[f"ensemble{ensemble}"][f"mass{mass}"]:
        # if two_state_enabled:
            if bool(data[f"ensemble{ensemble}"][f"mass{mass}"]["two_state_enabled"]):
                print('aaaa')
                two_state_enabled=True
                c2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["c2"]
                mp2 = data[f"ensemble{ensemble}"][f"mass{mass}"]["mp2"]
            else: 
                print('bbbb')
                two_state_enabled=False
        if not two_state_enabled:
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble]}, False
        else: 
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble], 'c2': gv.gvar(c2).mean, 'mp2': gv.gvar(mp2).mean/fm2GeV*alttc_arr[ensemble]}, True

    else:
        # return None, None, None
        raise DataNotFoundError(ensemble, mass)

def read_ini_pq(ensemble, lmass, smass, json_file, two_state_enabled):
    data = {}
    with open(json_file, "r") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        fcntl.flock(file, fcntl.LOCK_UN)

    ensemble_key = f"ensemble{ensemble}"
    lmass_key = f"lmass{lmass}"
    smass_key = f"smass{smass}"

    if ensemble_key in data and lmass_key in data[ensemble_key] and smass_key in data[ensemble_key][lmass_key]:
        mass_data = data[ensemble_key][lmass_key][smass_key]
        T_strt = mass_data["T_strt"]
        T_end = mass_data["T_end"]
        c1 = mass_data["c1"]
        mp = mass_data["mp"]
        if two_state_enabled:
            c2 = mass_data["c2"]
            mp2 = mass_data["mp2"]
        if not two_state_enabled:
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble]}
        else: 
            return T_strt, T_end, {'c1': gv.gvar(c1).mean, 'mp': gv.gvar(mp).mean/fm2GeV*alttc_arr[ensemble], 'c2': gv.gvar(c2).mean, 'mp2': gv.gvar(mp2).mean/fm2GeV*alttc_arr[ensemble]}

    else:
        # raise DataNotFoundError(ensemble, lmass, smass)  # Update the exception to include lmass and smass
        raise DataNotFoundError(ensemble, lmass)  # Update the exception to include lmass and smass


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

def save_ini_pq(ensemble, lmass, smass, T_strt, T_end, chi2, Q, json_file, fit_params, two_state_enabled):
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

            ensemble_key = f"ensemble{ensemble}"
            if ensemble_key not in data:
                data[ensemble_key] = {}

            # Convert T_strt and T_end to int if they are numpy.int64
            T_strt = int(T_strt)
            T_end = int(T_end)

            # Ensure lmass key exists
            if f"lmass{lmass}" not in data[ensemble_key]:
                data[ensemble_key][f"lmass{lmass}"] = {}

            # Store data under lmass and smass
            if two_state_enabled:
                data[ensemble_key][f"lmass{lmass}"][f"smass{smass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mp": str(fit_params["mp"]*fm2GeV/alttc_arr[ensemble]),
                    "mp2": str(fit_params["mp2"]*fm2GeV/alttc_arr[ensemble]),
                    "c1": str(fit_params["c1"]),
                    "c2": str(fit_params["c2"]),
                }
            else:
                data[ensemble_key][f"lmass{lmass}"][f"smass{smass}"] = {
                    "T_strt": T_strt,
                    "T_end": T_end,
                    "chi2": chi2,
                    "Q": Q,
                    "mp": str(fit_params["mp"]*fm2GeV/alttc_arr[ensemble]),
                    "c1": str(fit_params["c1"]),
                }

            # Sort the data dictionary
            sorted_data = OrderedDict(sorted(data.items(), key=lambda x: int(x[0].replace("ensemble", ""))))
            for ensemble_key in sorted_data:
                sorted_lmass = OrderedDict(sorted(sorted_data[ensemble_key].items()))
                for lmass_key in sorted_lmass:
                    sorted_lmass[lmass_key] = OrderedDict(sorted(sorted_lmass[lmass_key].items()))
                sorted_data[ensemble_key] = sorted_lmass

            # Move file position to the beginning before writing
            file.seek(0)
            file.truncate()

            json.dump(sorted_data, file, ensure_ascii=False, indent=4)

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

def save_bootstrap_baryon_mass(ensemble, mass, T_strt, T_end, mp, json_file):

    dir_name, file_name = os.path.split(json_file)
    lock_file = os.path.join(dir_name, '.'+file_name+'.lock')

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

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
                "T_end": T_end,
                "mp": str(mp),
            }

            # Sort the data dictionary
            sorted_data = OrderedDict(sorted(data.items()))
            # sorted_data = OrderedDict(sorted(data.items(), key=lambda x: int(x[0].replace("ensemble", ""))))
            for key in sorted_data:
                sorted_data[key] = OrderedDict(sorted(sorted_data[key].items()))

            # Move file position to the beginning before writing
            file.seek(0)
            file.truncate()

            json.dump(sorted_data, file, ensure_ascii=False, indent=4)

def save_bootstrap_baryon_mass_pq(ensemble, lmass, smass, T_strt, T_end, mp, json_file):
    dir_name, file_name = os.path.split(json_file)
    lock_file = os.path.join(dir_name, '.' + file_name + '.lock')

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

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

            ensemble_key = f"ensemble{ensemble}"
            if ensemble_key not in data:
                data[ensemble_key] = {}

            # Ensure lmass key exists
            if f"lmass{lmass}" not in data[ensemble_key]:
                data[ensemble_key][f"lmass{lmass}"] = {}

            # Convert T_strt and T_end to int if they are numpy.int64
            T_strt = int(T_strt)
            T_end = int(T_end)

            # Store data under lmass and smass
            data[ensemble_key][f"lmass{lmass}"][f"smass{smass}"] = {
                "T_strt": T_strt,
                "T_end": T_end,
                "mp": str(mp),
            }

            # Sort the data dictionary
            sorted_data = OrderedDict(sorted(data.items()))
            for ensemble_key in sorted_data:
                sorted_lmass = OrderedDict(sorted(sorted_data[ensemble_key].items()))
                for lmass_key in sorted_lmass:
                    sorted_lmass[lmass_key] = OrderedDict(sorted(sorted_lmass[lmass_key].items()))
                sorted_data[ensemble_key] = sorted_lmass

            # Move file position to the beginning before writing
            file.seek(0)
            file.truncate()

            json.dump(sorted_data, file, ensure_ascii=False, indent=4)

