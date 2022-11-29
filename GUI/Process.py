import pandas as pd
import numpy as np
import os
import datetime

from config import *

from signal_processing import WavenumberCorrection as WaveC
from signal_processing import SaturationCorrection, CosmicrayCorrection, smoothing, splitting

def run(args):
    files, fast_loading, preprocessing_variables, save_variables, noise_removal_variables, splitting_variables, text = args

    # load files
    data, wavenumbers, filenames = load_files(files, fast_loading)
    photo_approx = None
    photo = None

    # check if preprocessing is enabled
    if preprocessing_variables:
        data, wavenumbers = preprocessing(data, wavenumbers, preprocessing_variables)
        if save_variables['save_intermediate_results']:
            save_data(data, wavenumbers, filenames, save_variables, text)

    if noise_removal_variables:
        data = remove_noise(data, wavenumbers, noise_removal_variables)
        if save_variables['save_intermediate_results']:
            save_data(data, wavenumbers, filenames, save_variables, text)

    if splitting_variables:
        if splitting_variables["approximate_photo"]:
            photo_approx = approximated_splitting(data, wavenumbers, splitting_variables)
            photo = splitting_data(data, photo_approx, wavenumbers, splitting_variables)
        else:
            photo = splitting_data(data, data, wavenumbers, splitting_variables)
        raman = data - photo

    if photo is None:
        save_data(data, wavenumbers, filenames, save_variables, text)
    else:
        filenames_raman = [os.path.splitext(f)[0]+"_raman" for f in filenames]
        filenames_photo = [os.path.splitext(f)[0]+"_photoluminescence" for f in filenames]
        save_data(raman, wavenumbers, filenames_raman, save_variables, text)
        save_data(photo, wavenumbers, filenames_photo, save_variables, text)
    print("save complete", flush=True)

def splitting_data(data, photo_approx, wavenumbers, splitting_variables):
    photo = np.empty(data.shape)
    if len(wavenumbers.shape) == 1:
        split = splitting.split(
            wavenumbers = wavenumbers,
            size = data[0].shape[-1],
            FWHM = splitting_variables["RBF_FWHM"],
            order = splitting_variables["order"],
            convergence = splitting_variables["convergence"],
            segment_width = splitting_variables["segment_width"],
            algorithm = splitting_variables["segment_fitting_algorithm"]
        )
        for i, img in enumerate(data):
            photo[i] = split(img.reshape(-1, img.shape[-1]), photo_approx.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"split for image {i+1} out of {len(data)} done", flush=True)
    else:
        for i, img in enumerate(data):
            split = splitting.split(
                wavenumbers = wavenumbers[i],
                size = data[0].shape[-1],
                FWHM = splitting_variables["RBF_FWHM"],
                order = splitting_variables["order"],
                convergence = splitting_variables["convergence"],
                segment_width = splitting_variables["segment_width"],
                algorithm = splitting_variables["segment_fitting_algorithm"]
            )
            photo[i] = split(img.reshape(-1, img.shape[-1]), photo_approx.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"split for image {i+1} out of {len(data)} done", flush=True)
    return photo

def approximated_splitting(data, wavenumbers, splitting_variables):
    photo_approx = np.empty(data.shape)
    if len(wavenumbers.shape) == 1:
        approximate_split = splitting.preliminary_split(
            wavenumbers = wavenumbers,
            convergence = splitting_variables["convergence_approximate"],
            order = splitting_variables["order_approximate"],
            FWHM = splitting_variables["RBF_FWHM_approximate"],
            size = data[0].shape[-1]
        )
        for i, img in enumerate(data):
            photo_approx[i] = approximate_split(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"approximated split for image {i+1} out of {len(data)} done", flush=True)
    else:
        for i, img in enumerate(data):
            approximate_split = splitting.preliminary_split(
                wavenumbers = wavenumbers[i],
                convergence = splitting_variables["convergence_approximate"],
                order = splitting_variables["order_approximate"],
                FWHM = splitting_variables["RBF_FWHM_approximate"],
                size = data[0].shape[-1]
            )
            photo_approx[i] = approximate_split(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"approximated split for image {i+1} out of {len(data)} done", flush=True)
    return photo_approx

def remove_noise(data, wavenumbers, noise_removal_variables):
    # check if each image has his own wavenumbers or if they are all equal.
    if len(wavenumbers.shape) == 1:
        remove_noise_cube_fft = smoothing.RemoveNoiseFFTPCA(
            algorithm = noise_removal_variables['noise_removal_algorithm'],
            percentage_noise = noise_removal_variables["noise_percenteage"],
            wavenumbers = wavenumbers,
            min_FWHM = noise_removal_variables["noise_automated_FWHM"],
            error_function = noise_removal_variables["noise_error_algorithm"],
            gradient_width = noise_removal_variables["noise_gradient_width"],
            spike_padding = noise_removal_variables["noise_spike_padding"],
            max_spike_width = noise_removal_variables["noise_max_spike_width"]
        )
        for i, img in enumerate(data):
            data[i] = remove_noise_cube_fft(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"Removing noise for image {i+1} out of {len(data)} done", flush=True)
    else:
        for i, img in enumerate(data):
            remove_noise_cube_fft = smoothing.RemoveNoiseFFTPCA(
                algorithm = noise_removal_variables['noise_removal_algorithm'],
                percentage_noise = noise_removal_variables["noise_percenteage"],
                wavenumbers = wavenumbers[i],
                min_FWHM = noise_removal_variables["noise_automated_FWHM"],
                error_function = noise_removal_variables["noise_error_algorithm"],
                gradient_width = noise_removal_variables["noise_gradient_width"],
                spike_padding = noise_removal_variables["noise_spike_padding"],
                max_spike_width = noise_removal_variables["noise_max_spike_width"]
            )
            data[i] = remove_noise_cube_fft(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            print(f"Removing noise for image {i+1} out of {len(data)} done", flush=True)

    return data

def preprocessing(data, wavenumbers, preprocessing_variables):
    # check if same stepsize is enabled.
    if 'all_images_same_stepsize' in preprocessing_variables:
        print("starting correcting for stepsize", flush=True)
        if preprocessing_variables['all_images_same_stepsize']:
            data, wavenumbers = WaveC.correct_wavenumbers_between_samples(data, wavenumbers, preprocessing_variables['stepsize'])
        else:
            data, wavenumbers = WaveC.correct_wavenumbers_within_samples(data, wavenumbers, preprocessing_variables['stepsize'])
        print("correcting for stepsize done", flush=True)

    # check if saturation is enabled.
    if 'saturation_width' in preprocessing_variables:
        remove_saturation = SaturationCorrection.remove_saturation(region_width=preprocessing_variables['saturation_width'])

        for i, img in enumerate(data):
            data[i] = remove_saturation(img)
            print(f"correcting for saturation done for image {i+1} out of {len(data)}", flush=True)

    # check if cosmic ray noise is enabled.
    if 'n_times' in preprocessing_variables:
        args = [preprocessing_variables['n_times'],
                preprocessing_variables['FWHM_smoothing'],
                preprocessing_variables['max_FWHM'],
                preprocessing_variables['region_padding'],
                preprocessing_variables['max_oc'],
                preprocessing_variables['interpolate_degree']]

        if len(wavenumbers.shape) == 1:
            cosmicray_removal = CosmicrayCorrection.remove_cosmicrays(wavenumbers, *args)
            for i, img in enumerate(data):
                data[i], _ = cosmicray_removal(img)
                print(f"correcting for cosmic ray noise done for image {i+1} out of {len(data)}", flush=True)
        else:
            for i, img in enumerate(data):
                cosmicray_removal = CosmicrayCorrection.remove_cosmicrays(wavenumbers[i], *args)
                data[i], _ = cosmicray_removal(img)
                print(f"correcting for cosmic ray noise done for image {i+1} out of {len(data)}", flush=True)

    return data, wavenumbers

def save_data(data, wavenumbers, filenames, save_variables, text):
    # save data in new folder
    timestamp = str(datetime.datetime.now()).replace(":","-")
    save_dir = save_variables["save_dir"] + '//' + timestamp + '//'
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir+"metadata.txt", "w") as f:
        f.write(text)

    if save_variables["save_as_txt"]:
        if len(wavenumbers.shape) == 1:
            w = wavenumbers
            for name, img in zip(filenames, data):
                textfile = np.empty((np.prod(img.shape[:-1])+1, len(w)+2))
                textfile[0, 2:] = w
                textfile[1:, 2:] = img.reshape(-1, len(w))
                Y, X = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
                textfile[1:, 0] = X.flatten()
                textfile[1:, 1] = Y.flatten()
                np.savetxt(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}.txt', textfile, delimiter="\t", fmt="%10.6f")
        else:
            for name, img, w in zip(filenames, data, wavenumbers):
                textfile = np.empty((np.prod(img.shape[:-1])+1, len(w)+2))
                textfile[0, 2:] = w
                textfile[1:, 2:] = img.reshape(-1, len(w))
                Y, X = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
                textfile[1:, 0] = X.flatten()
                textfile[1:, 1] = Y.flatten()
                np.savetxt(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}.txt', textfile, delimiter="\t", fmt="%10.6f")

    if save_variables["save_as_numpy"]:
        # save wavenumbers
        if len(wavenumbers.shape) == 1:
            np.save(f'{save_dir}Wavenumbers', wavenumbers)
        else:
            for name, w in zip(filenames, wavenumbers):
                np.save(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}_Wavenumbers', w)

        # save file
        for name, img in zip(filenames, data):
            np.save(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}', img)

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
        for i, file in enumerate(files):
            print(f"opening file {i+1} of {len(files)}: {file}", flush=True)
            df = pd.read_csv(file, delimiter='\t', skipinitialspace=True, header=header, skiprows=[], dtype=np.float32)
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
                img = np.empty((len(X), len(Y), len(wavenumbers)), dtype=np.float32)
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

            print(f"loaded  file {i+1} of {len(files)}: {file}", flush=True)
            all_images.append(img)
            all_wavenumbers.append(wavenumbers)
    else:
        files, wave_files = files[0], files[1]
        if len(wave_files) == 1:
            all_wavenumbers = np.load(wave_files[0])
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
    return np.array(all_images), np.array(all_wavenumbers), files
