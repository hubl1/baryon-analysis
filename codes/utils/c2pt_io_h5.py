import h5py
import numpy as np

def h5_get_pp2pt(hadron_path):
    with h5py.File(hadron_path, "r") as f:
        pp2pt_list = []
        confs_list = []

        # Iterate through configurations
        for conf in sorted(f.keys(), key=int):
            conf = str(conf)

            c2pt = None  # Initialize c2pt as None to handle first entry assignment
            n_src = 0

            # Accumulate data for each time source
            for t_src in sorted(f[conf].keys(), key=int):
            # for t_src in [sorted(f[conf].keys(), key=int)[0]]:
                if f"{conf}/{t_src}/x" in f:
                    data_x = f[f"{conf}/{t_src}/x"][:]
                    data_y = f[f"{conf}/{t_src}/y"][:]
                    data_z = f[f"{conf}/{t_src}/z"][:]
                    data = (data_x + data_y + data_z) / 3
                    # data = f[f"{conf}/{t_src}/x"][:]
                    # data = f[f"{conf}/{t_src}/y"][:]
                    # data = f[f"{conf}/{t_src}/z"][:]
                else:
                    data = f[f"{conf}/{t_src}"][:]
                    # print('aaaaa')

                # Assign data directly to c2pt for the first entry, then add for subsequent entries
                if c2pt is None:
                    c2pt = np.array(data)  # Set initial data for the first source
                else:
                    c2pt += np.array(data)  # Accumulate for other sources

                n_src += 1

            # Average c2pt by the number of sources
            pp2pt_list.append(c2pt / n_src)
            confs_list.append(int(conf))

        # Convert list to numpy array
        pp2pt = np.array(pp2pt_list)
        # confs= np.array(confs_list)
        confs= confs_list
        print(pp2pt.shape)

    return pp2pt, confs
