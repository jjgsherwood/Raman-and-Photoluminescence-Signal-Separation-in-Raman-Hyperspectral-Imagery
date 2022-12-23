import pandas as pd
import numpy as np
import os
import datetime
import copy

from config import *

from signal_processing import WavenumberCorrection as WaveC
from signal_processing import SaturationCorrection, CosmicrayCorrection, smoothing, splitting
from utils import classifier, config, module, classifier


def run(args):
    files, fast_loading, preprocessing_variables, save_variables, noise_removal_variables, splitting_variables, NN_train_variables, NN_load_variables, text = args

    # load files
    data, wavenumbers, shapes, filenames = load_files(files, fast_loading)
    photo_approx = None
    photo = None
    raman_wavenumbers = None
    photo_wavenumbers = None

    data, wavenumbers = remove_duplicate_wavenumbers(data, wavenumbers)

    # check if selected parameters are possible
    try:
        checks(wavenumbers, splitting_variables, preprocessing_variables)
    except ValueError as e:
        print(e)
        return
    print("checks 1 done")

    # check for manual loading from files
    try:
        NN_train_variables['fast_loading']
    except KeyError:
        pass
    else:
        raman, raman_wavenumbers, _ = load_files(NN_train_variables['raman_files'], NN_train_variables['fast_loading'])
        photo, photo_wavenumbers, _ = load_files(NN_train_variables['photo_files'], NN_train_variables['fast_loading'])

    # check if wavenumbers are compatible
    if preprocessing_variables:
        data, wavenumbers = preprocessing(data, wavenumbers, preprocessing_variables, shapes)
        if save_variables['save_intermediate_results'] and (noise_removal_variables or splitting_variables):
            save_data(data, wavenumbers, shapes, filenames, save_variables, "intermediate results!\n\n"+text.split("/n/n")[0], name="preprocessed ")

    # check if selected parameters are possible
    try:
        checks2(data, wavenumbers, raman_wavenumbers, photo_wavenumbers, preprocessing_variables, save_variables, noise_removal_variables, splitting_variables, NN_train_variables, NN_load_variables)
    except ValueError as e:
        print(e)
        return
    print("checks 2 done")

    # save data for later use in the NN
    if NN_train_variables and not NN_train_variables['use_denoised_data']:
        raw = data

    if noise_removal_variables:
        data = remove_noise(data, wavenumbers, noise_removal_variables)
        if save_variables['save_intermediate_results'] and splitting_variables:
            save_data(data, wavenumbers, shapes, filenames, save_variables, "intermediate results!\n\n"+text.split("See selected splitting parameters below")[0], name="noise_removed ")

    # save data for later use in the NN
    if NN_train_variables and NN_train_variables['use_denoised_data']:
        raw = data

    if splitting_variables:
        if splitting_variables["approximate_photo"]:
            photo_approx = approximated_splitting(data, wavenumbers, splitting_variables)
            photo = splitting_data(data, photo_approx, wavenumbers, splitting_variables)
        else:
            photo = splitting_data(data, data, wavenumbers, splitting_variables)
        raman = data - photo

    if NN_load_variables:
        raman, photo = split_with_NN(data, wavenumbers, NN_load_variables)

    # saving data
    if photo is None:
        if noise_removal_variables:
            name = "noise_removed "
        elif preprocessing_variables:
            name = "preprocessed "
        else:
            name = "raw_"
        save_data(data, wavenumbers, shapes, filenames, save_variables, text, name=name)
    else:
        filenames_raman = [os.path.splitext(f)[0]+"_raman" for f in filenames]
        filenames_photo = [os.path.splitext(f)[0]+"_photoluminescence" for f in filenames]
        save_data(raman, wavenumbers, shapes, filenames_raman, save_variables, text, name="raman ")
        save_data(photo, wavenumbers, shapes, filenames_photo, save_variables, text, name="photoluminences ")
    print("save complete", flush=True)

    if NN_train_variables:
        print("start training neural network", flush=True)
        # try to split one image in multiple images
        if 'split_image_in_val_and_train' in NN_train_variables:
            shape = raw.shape[2:]
            for i in range(3,raw.shape[1]+1):
                try:
                    raw, photo, raman = raw.reshape(i,-1,*shape), photo.reshape(i,-1,*shape), raman.reshape(i,-1,*shape)
                except ValueError:
                    continue
                break
            NN_train_variables['validation_percentage'] = f"{min(1,int(config.VALIDATION_PER * len(raw))) / len(raw) * 100}%"
            first, last = text.split("See selected neural network training parameters below:")
            first += "See selected neural network training parameters below:\n\n"
            first += "NN_train_variables['validation_percentage']"
            first += last[1:]
            text = first
        rvc = train_NN(raw, photo, raman, NN_train_variables, save_variables, text)

def checks(wavenumbers, splitting_variables, preprocessing_variables):
    # check wavenumber stepsize
    if preprocessing_variables['all_images_same_stepsize']:
        if preprocessing_variables['stepsize'] == 'min':
            min_stepsize = min(np.partition(w[1:] - w[:-1], 2)[2] for w in wavenumbers)
        elif preprocessing_variables['stepsize'] == 'max':
            min_stepsize = max(np.partition(w[1:] - w[:-1], -3)[-3] for w in wavenumbers)

        if preprocessing_variables['stepsize'] == 'min' or preprocessing_variables['stepsize'] == 'max':
            mean_stepsize = np.mean([np.mean(w[1:] - w[:-1]) for w in wavenumbers])
            ratio = mean_stepsize / min_stepsize
            if ratio > 10 or ratio < 0.1:
                raise ValueError(f"The stepsize chosen is {preprocessing_variables['stepsize']}, however the ratio between the mean value {mean_stepsize} and the chosen step size is {ratio}.\nThis can lead to errors in processes down the line.\nAborted!")

def checks2(data, wavenumbers, raman_wavenumbers, photo_wavenumbers, preprocessing_variables, save_variables, noise_removal_variables, splitting_variables, NN_train_variables, NN_load_variables):
    # check segment width
    if  len(wavenumbers.shape) == 1:
        w = wavenumbers
    else:
        w = wavenumbers[0]
    if "segment_width" in splitting_variables and splitting_variables["segment_width"] >= (w[-1] - w[0]):
        raise ValueError("The selected segement width is wider than the signal width!")

    # check if all wave files are the same length
    if NN_train_variables or NN_load_variables:
        if len(wavenumbers.shape) == 1:
            wavenumber = wavenumbers
            wavenumbers = wavenumbers.reshape(1, *wavenumbers.shape)
        else:
            wavenumber = wavenumbers[0]
            for check_wavenumbers in wavenumbers:
                if len(wavenumber) != len(check_wavenumbers):
                    raise ValueError(f"file {check_wavenumbers} does not have the same number of wavenumbers as {wavenumbers[0]}.")
                if sum(~np.isclose(wavenumber, check_wavenumbers)):
                    raise ValueError(f"file {check_wavenumbers} does not have the same wavenumbers as {wavenumbers[0]}.\nTraining a neural nerwork with different inputs is not possible.")

    if NN_train_variables and raman_wavenumbers is not None:
        # raman_wavenumbers = np.concatenate([r.reshape(-1, raman_wavenumbers.shape[-1]) for r in raman_wavenumbers])
        # photo_wavenumbers = np.concatenate([p.reshape(-1, photo_wavenumbers.shape[-1]) for p in photo_wavenumbers])
        raman_wavenumbers = raman_wavenumbers.reshape(-1, raman_wavenumbers.shape[-1])
        photo_wavenumbers = photo_wavenumbers.reshape(-1, photo_wavenumbers.shape[-1])

        for i, check_wavenumbers in enumerate(raman_wavenumbers):
            if len(wavenumber) != len(check_wavenumbers):
                raise ValueError(f"file {NN_train_variables['raman_files'][i][0]} does not have the same number of wavenumbers as the raw data.")
            if sum(~np.isclose(wavenumber, check_wavenumbers)):
                print(wavenumber)
                print(check_wavenumbers)
                raise ValueError(f"file {NN_train_variables['raman_files'][i][0]} does not have the same wavenumbers as the raw data.\nTraining a neural nerwork with different inputs is not advised.")

        for i, check_wavenumbers in enumerate(photo_wavenumbers):
            if len(wavenumber) != len(check_wavenumbers):
                raise ValueError(f"file {NN_train_variables['photo_files'][i][0]} does not have the same number of wavenumbers as the raw data.")
            if sum(~np.isclose(wavenumber, check_wavenumbers)):
                raise ValueError(f"file {NN_train_variables['photo_files'][i][0]} does not have the same wavenumbers as the raw data.\nTraining a neural nerwork with different inputs is not advised.")

def check_and_correct_for_different_spacial_dim(data, shapes):
    # check if shapes are equally sized
    equal_sizes = True
    for shape in shapes[1:]:
        if shape != shapes[0]:
            equal_sizes = False
            break

    if not equal_sizes:
        data = np.concatenate([image.reshape(-1, image.shape[-1]) for image in data])
        print("WARNING: due to incompatible images size, the image are combined in a single line of points")
    return data.reshape(1,1,-1,data.shape[-1])

def remove_duplicate_wavenumbers(data, wavenumbers):
    for img,(d,w) in enumerate(zip(data, wavenumbers)):
        for i in sorted(np.where(w[1:] - w[:-1] == 0)[0], reverse=True):
            print(f"image {img} has a duplicate wavenumber at index {i} with value {w[i]}")
            d[:,:,i] = (d[:,:,i] + d[:,:,i+1]) / 2
            d = np.delete(d, i+1, -1)
            w = np.delete(w, i+1, -1)
            data[img] = d
            wavenumbers[img] = w
    return data, wavenumbers

def split_with_NN(data, wavenumbers, NN_load_variables):
    load_file = NN_load_variables['NN_file']
    kwargs = copy.copy(config.NN_INPUTS)
    kwargs['batch_size'] = NN_load_variables['batchsize']
    rvc = classifier.SupervisedSplitting(**kwargs)

    """
    TODO adjust number of wavenumbers to fit NN.
    """

    return rvc.predict(data, NN_load_variables["NN_file"])

def train_NN(raw, photo, raman, NN_train_variables, save_variables, text):
    kwargs = copy.copy(config.NN_INPUTS)
    kwargs['epochs'] = NN_train_variables['epochs']
    kwargs['batch_size'] = NN_train_variables['batchsize']
    kwargs['loss_func'] = module.loss_func
    kwargs['acc_func'] = module.acc_func

    # this order is very important
    data = np.stack((raw, raman, photo), 1)
    rvc = classifier.SupervisedSplitting(**kwargs)

    name = "saved_neural_network_"
    timestamp = str(datetime.datetime.now()).replace(":","-")
    save_dir = save_variables["save_dir"] + '//' + name + timestamp + '//'
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir+"metadata.txt", "w") as f:
        f.write(text)

    try:
        load_file = NN_train_variables['NN_file']
    except KeyError:
        load_file = None
    rvc.fit(data, save_dir, load_file)
    return rvc

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
            photo[i] = split(img.reshape(-1, img.shape[-1]), photo_approx[i].reshape(-1, img.shape[-1])).reshape(img.shape)
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
            photo[i] = split(img.reshape(-1, img.shape[-1]), photo_approx[i].reshape(-1, img.shape[-1])).reshape(img.shape)
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

def preprocessing(data, wavenumbers, preprocessing_variables, shapes):
    # check if same stepsize is enabled.
    if 'all_images_same_stepsize' in preprocessing_variables:
        print("starting correcting for stepsize", flush=True)
        if preprocessing_variables['all_images_same_stepsize']:
            data, wavenumbers = WaveC.correct_wavenumbers_between_samples(data, wavenumbers, preprocessing_variables['stepsize'])
        else:
            data, wavenumbers = WaveC.correct_wavenumbers_within_samples(data, wavenumbers, preprocessing_variables['stepsize'])
        print("correcting for stepsize done", flush=True)

    data = check_and_correct_for_different_spacial_dim(data, shapes)

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

def save_data(data, wavenumbers, shapes, filenames, save_variables, text, name=""):
    # save data in new folder
    timestamp = str(datetime.datetime.now()).replace(":","-")
    save_dir = save_variables["save_dir"] + '//' + name + timestamp + '//'
    os.makedirs(save_dir, exist_ok=True)

    # check if data was compressed in one image or not
    if data[0].shape[:-1] != shapes[0]:
        data_lst = []
        n = 0
        for shape in shapes:
            n_new = n + np.prod(shape)
            data_lst.append(data[:,:,n:n_new,:].reshape(*shape, -1))
            n = n_new
        data = data_lst

    with open(save_dir+"metadata.txt", "w") as f:
        f.write(text)

    if save_variables["save_as_txt"]:
        if len(wavenumbers.shape) == 1:
            w = wavenumbers
            for name, img in zip(filenames, data):
                textfile = np.empty((np.prod(img.shape[:-1]), len(w)+2))
                # textfile[0, 2:] = w
                textfile[:, 2:] = img.reshape(-1, len(w))
                Y, X = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
                textfile[:, 0] = X.flatten()
                textfile[:, 1] = Y.flatten()
                np.savetxt(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}.txt', textfile, header="\t\t"+"\t".join(map(str,w)), comments="", delimiter="\t", fmt=save_variables['save_as_txt_fmt'])
        else:
            for name, img, w in zip(filenames, data, wavenumbers):
                textfile = np.empty((np.prod(img.shape[:-1]), len(w)+2))
                # textfile[0, 2:] = w
                textfile[:, 2:] = img.reshape(-1, len(w))
                Y, X = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
                textfile[:, 0] = X.flatten()
                textfile[:, 1] = Y.flatten()
                np.savetxt(f'{save_dir}{os.path.splitext(os.path.basename(name))[0]}.txt', textfile, header="\t\t"+"\t".join(map(str,w)), comments="", delimiter="\t", fmt=save_variables['save_as_txt_fmt'])

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
            # single point load
            if not len(wavenumbers):
                wavenumbers = data[:,0]
                img = data[:,1].reshape(1,1,-1)
            elif fast_loading:
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

    shapes = [image.shape[:-1] for image in all_images]

    return all_images, all_wavenumbers, shapes, files
