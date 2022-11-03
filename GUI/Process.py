import pandas as pd
import numpy as np
from config import *
from signal_processing import WavenumberCorrection as WaveC


def run(args):
    files, fast_loading, preprocessing_variables, variables = args
    data, wavenumbers = load_files(files, fast_loading)

    # if preprocessing_variables['']
    # new_data, new_wavenumbers = WaveC.correct_wavenumbers_between_samples(data, wavenumbers)
    # new_data, new_wavenumbers = WaveC.correct_wavenumbers_within_samples(data, wavenumbers)


def load_files(files, fast_loading):
    print(f"start loading data, number of files {len(files[0])}", flush=True)
    if len(files) == 1:
        files = files[0]
        # check how data is stored
        with open(files[0]) as f:
            if '#X' == f.readline().split('\t')[0]:
                header = 0
            else:
                header = None

        all_images = []
        all_wavenumbers = []
        for file in files:
            print(f"opening file {file}", flush=True)
            df = pd.read_csv(file, delimiter='\t', skipinitialspace=True, header=header, skiprows=[])
            data = df.to_numpy()

            if header is None:
                wavenumbers = data[0,2:]
                data = data[1:]
            else:
                wavenumbers = sorted(list(np.unique(data[:,2])))

            X = list(sorted(np.unique(data[:,0])))
            Y = list(sorted(np.unique(data[:,1])))

            if fast_loading:
                if header is None:
                    data = data[:,2:]
                    img = data.reshape(len(X), len(Y), len(wavenumbers))
                else:
                    data = data[:,3]
                    data = data.reshape(len(Y), len(X), len(wavenumbers))
                    data = np.rollaxis(data, 1, 0)
                    img = data[:,:,::-1]
            else:
                img = np.empty((len(X), len(Y), len(wavenumbers)), dtype=np.float64)
                if header is None:
                    for d in data:
                        i = X.index(d[0])
                        j = Y.index(d[1])
                        img[i,j,:] = d[2:]
                else:
                    for d in data:
                        i = X.index(d[0])
                        j = Y.index(d[1])
                        w = wavenumbers.index(d[2])
                        img[i,j,w] = d[3]

                    wavenumbers = np.array(wavenumbers)

            print(f"{file} loaded", flush=True)
            all_images.append(img)
            all_wavenumbers.append(wavenumbers)

        return all_images, all_wavenumbers
    else:
        files, wave_files = files[0], files[1]
        if len(wave_files) == 1:
            all_wavenumbers = [np.load(wave_files)]
            all_images = []
            for file in files:
                all_images.append(np.load(file))
        else:
            all_images = []
            all_wavenumbers = []
            for file in files:
                all_images.append(np.load(file))
                all_wavenumbers.append(np.load(file.replace('.npy','_Wavenumbers.npy')))
        print(f"data loaded", flush=True)
        return all_images, all_wavenumbers
