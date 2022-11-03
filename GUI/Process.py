import pandas as pd
import numpy as np

def run(args):
    files, preprocessing_variables, variables = args
    data, wavenumbers = load_files(files)

def load_files(files):
    print(f"start, number of files {len(files[0])}", flush=True)
    if len(files) == 1:
        files = files[0]
        # check how data is stored
        with open(files[0]) as f:
            if '#X' == f.readline().split('\t')[0]:
                header = 0
            else:
                header = None

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

            img = np.empty((len(X), len(Y), len(wavenumbers)), dtype=np.float64)

            if FAST_LOADING:
                pass

            else:
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

            # data = df.to_numpy()[:,:4]
            #
            # wavenumbers = sorted(list(np.unique(data[:,2])))
            # X = np.array(sorted(list(np.unique(data[:,0]))))
            # Y = np.array(sorted(list(np.unique(data[:,1]))))
    return None, None
