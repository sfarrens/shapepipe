# -*- coding: utf-8 -*-

"""GALSIM SHAPES RUNNER

This file contains methods to measure shapes with Galsim.

:Author: Axel Guinot

"""

from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io
from shapepipe.pipeline.execute import execute

from astropy.io import fits

import re
import os

import numpy as np

import ngmix


##########################
# DEBUG : Testing galaxy #
##########################

from tqdm import tqdm

#########
# Utils #
#########

def mad(data, axis=None):
    """Median absolute deviation

    Compute the mad of an array.

    Parameters
    ----------
    data : numpy.ndarray
        Data array
    axis : int, tuple
        Axis on which computing the mad.

    """
    return np.median(np.abs(data - np.median(data, axis)), axis)*1.4826


def get_inv_gauss_2D(sigma, center=(96/2., 96/2.), shape=(96, 96)):
    """Inverse gaussian 2D

    Compute an inverse gaussian in 2D.
    Illustration :

      1 |......          .......
        |      .        .
        |       .      .
        |        .    .
        |         .  .
      0 |_________ .. ____________

    Parameters
    ----------
    sigma : float
        Sigma of the gaussian (assuming sig_x = sig_y).
    center : tupple
        Center of the gaussian (x, y).
    shape : tupple
        Output vignet shape (nx, ny).

    Returns
    -------
    numpy.ndarray
        Return the inverted gaussian

    """

    x, y = np.meshgrid(np.linspace(0, shape[0]-1, shape[0]), np.linspace(0, shape[1]-1, shape[1]))
    gauss = - np.exp(-(((x-center[0])**2. + (y-center[1])**2.))/(2. * sigma**2.)) / (sigma**2. * 2. * np.pi)
    new_gauss = (gauss - np.min(gauss))/(np.max(gauss) - np.min(gauss))

    return new_gauss


#################
# Data handling #
#################

def parse_data(map_img, img_size=96, n_img=10000):
    """ Parse data

    Parse the GREAT3 data format into an 1D array.

    Parameters
    ----------
    img_size : int
        Size of one vignet (assuming square images).
    n_img : int
        Total number of vignets.

    Returns
    -------
    final_array : numpy.ndarray
        Array containing the vignet with shape : (n_img, img_size, img_size).

    """

    map_size = map_img.shape

    final_array = np.zeros((int(n_img), int(img_size), int(img_size)))

    k = 0
    for x in range(0, int(np.sqrt(n_img)*img_size), img_size):
        for y in range(0, int(np.sqrt(n_img)*img_size), img_size):
            final_array[k] = map_img[x:x+img_size,y:y+img_size]
            k+= 1

    return final_array


def map_vignet(img_arr, dtype):
    """Map vignet

    Map vignet on one single image.

    Parameters
    ----------
    img_arr : numpy.ndarray
        Array of vinget to map
    dtype : str
        dtype of the data

    Returns
    -------
    img_map : numpy.ndarray
        Array containing all the vignets mapped on one single image
    nx : int
        Number of objects along one side (assumed square image)

    """

    n_obj = img_arr.shape[0]
    xs = img_arr[0].shape[0]
    ys = img_arr[0].shape[1]

    nx = int(np.sqrt(n_obj))
    if nx*nx != n_obj:
        nx += 1
    ny = nx

    img_map=np.ones((xs*nx,ys*ny), dtype=dtype)

    ii=0
    jj=0
    for i in range(n_obj):
        if jj>nx-1:
            jj=0
            ii+=1
        img_map[ii*xs:(ii+1)*xs,jj*ys:(jj+1)*ys]=img_arr[i]
        jj+=1

    return img_map, nx


#####################
# Metacal functions #
#####################

def psf_fitter(psf_vign, opt_dict):
    """Psf fitter

    Function used to create a gaussian fit of the PSF.

    Parameters
    ----------
    psf_vign : numpy.array
        Array containg one vignet of psf

    Returns
    -------
    psf_obs: ngmix.Observation
        Object containing all the information relative to the psf

    """

    psf_jacob = ngmix.DiagonalJacobian(scale=opt_dict['pixel_scale'], x=psf_vign.shape[0]/2., y=psf_vign.shape[1]/2.)

    psf_obs=ngmix.Observation(psf_vign, jacobian=psf_jacob)
    pfitter=ngmix.fitting.LMSimple(psf_obs,'gauss')

    shape = psf_vign.shape
    psf_pars = np.array([0., 0., 0., 0., 0.05, 1.])
    pfitter.go(psf_pars)

    psf_gmix_fit=pfitter.get_gmix()
    psf_obs.set_gmix(psf_gmix_fit)

    return psf_obs


def make_metacal(gal_vign, psf_vign, weight_vign, opt_dict):
    """Make the metacalibration

    This function call different ngmix functions to create images needed for the metacalibration.

    Parameters
    ----------
    gal_vign : numpy.array
        Array containing one vignet of galaxy
    psf_vign : numpy.array
        Array containg one vignet of psf
    opt_dict : dict
        Dictionnary containg option for ngmix (keys : ['TYPES', 'FIXNOISE', 'CHEATNOISE', 'SYMMETRIZE_PSF', 'STEP'])

    Returns
    -------
    obs_out :
        Ngmix object with all the metacal images

    """

    # psf_obs = psf_fitter(psf_vign, opt_dict)
    psf_jacob = ngmix.DiagonalJacobian(scale=opt_dict['pixel_scale'], x=psf_vign.shape[0]/2., y=psf_vign.shape[1]/2.)
    psf_obs=ngmix.Observation(psf_vign, jacobian=psf_jacob)

    gal_jacob = ngmix.DiagonalJacobian(scale=opt_dict['pixel_scale'], x=gal_vign.shape[0]/2., y=gal_vign.shape[1]/2.)

    obs = ngmix.Observation(gal_vign, psf=psf_obs, weight=weight_vign, jacobian=gal_jacob)

    obs_out = ngmix.metacal.get_all_metacal(obs,
                                            types=opt_dict['TYPES'],
                                            fixnoise=opt_dict['FIXNOISE'],
                                            cheatnoise=opt_dict['CHEATNOISE'],
                                            step=opt_dict['STEP'],
                                            psf=opt_dict['PSF_KIND'])

    return obs_out


#############
# ShapeLens #
#############

def run_shapelens(img_path, psf_path, img_size, grid, type, opt_dict):
    """Run ShapeLens

    Run ShapeLens on an image.

    Parameters
    ----------
    img_path : str
        Path to the image
    psf_path : str
        Path to the PSF image
    img_size : int
        Size of one stamp that composed the mapped image
    grid : int
        Number of objects along one side (assumed square image)
    type : str
        Metacal type in ['1m', '1p', '2m', '2p', 'noshear']
    opt_dict : dict
        Dictionnary with other options

    """

    command_line = '{} {} -p {} -s {} -g {} {}'.format(opt_dict['execute'],
                                                       img_path,
                                                       psf_path,
                                                       img_size,
                                                       grid,
                                                       opt_dict['shapelens_opt'])

    stderr, stdout = execute(command_line)

    # Re-format the output
    tmp = np.array(re.split('\n|\t', stderr))
    keys = ['id', 'x', 'y', 'e1', 'e2', 'scale', 'SNR']
    dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float']
    output_dict = {}
    for i, key, dtype in zip(range(7), keys, dtypes):
        output_dict[key] = tmp[i:-2:7].astype(dtype)

    # Save output
    output_name = opt_dict['dirs'][type] + '/shapelens' + opt_dict['file_number_string'] + '.fits'
    f = io.FITSCatalog(output_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
    f.save_as_fits(output_dict, overwrite=True)

def run_shapelens_output(img_path, psf_path, img_size, grid, type, opt_dict):
    """
    """

    output_name = opt_dict['dirs'][type] + '/shapelens' + opt_dict['file_number_string'] + '.fits'

    command_line = '{} {} {} -p {} -s {} -g {} {}'.format(opt_dict['execute'],
                                                       img_path,
                                                       output_name,
                                                       psf_path,
                                                       img_size,
                                                       grid,
                                                       opt_dict['shapelens_opt'])

    print(command_line)

    stderr, stdout = execute(command_line)

    print('##### stderr ######')
    print(stderr)
    print()
    print('###### stdout ######')
    print(stdout)
    print()


def process(img_path, psf_path, opt_dict, img_size, n_obj, dtype='float32'):
    """Process

    Main function which handle all the processing.

    Parameters
    ----------
    img_path : str
        Path to the image
    psf_path : str
        Path to the PSF image
    opt_dict : dict
        Dictionnary with other options
    img_size : int
        Size of one stamp that composed the mapped image
    n_obj : int
        Total number of objects
    dtype : str
        type of the data (Default = 'float32')

    """

    # Load images
    map_img = fits.getdata(img_path, 0)
    map_psf = fits.getdata(psf_path)
    img_header = fits.getheader(img_path, 0)

    # Parse the data
    # From mapped image to vignets
    img_arr = parse_data(map_img, img_size, n_obj)
    psf_arr = parse_data(map_psf, img_size, n_obj)

    mcal_dict = {key: np.zeros((n_obj, img_size, img_size), dtype=dtype) for key in opt_dict['TYPES'] + ['psf']}
    img_shape = np.array(img_arr[0].shape)

    # for img, psf, i in tqdm(zip(img_arr, psf_arr, range(n_obj)), total=n_obj):
    for img, psf, i in zip(img_arr, psf_arr, range(n_obj)):

        # Get noise level for the weight
        sigma_noise = mad(img)
        weight = np.ones_like(img) * 1./sigma_noise**2.

        # Create metacal images
        obs_mcal = make_metacal(img, psf, weight, opt_dict)

        new_psf = obs_mcal['noshear'].get_psf().galsim_obj
        mcal_dict['psf'][i] = new_psf.drawImage(nx=img_shape[0],ny=img_shape[1],scale=opt_dict['pixel_scale']).array
        for key in opt_dict['TYPES']:
            mcal_dict[key][i] = obs_mcal[key].image

    # Map the metacal images and save them and run shapelens
    # Save PSF first
    psf_map_name = '{}/psf{}.fits'.format(opt_dict['dirs']['psf'], opt_dict['file_number_string'])
    map_tmp, grid = map_vignet(mcal_dict['psf'], dtype)
    f = io.FITSCatalog(psf_map_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
    f.save_as_fits(map_tmp, image=True, image_header=img_header, overwrite=True)

    for key in opt_dict['TYPES']:
        print('running Shapelens on : {}'.format(key))
        map_tmp, grid = map_vignet(mcal_dict[key], dtype)
        tmp_name = '{}/{}{}.fits'.format(opt_dict['dirs'][key], key, opt_dict['file_number_string'])
        f = io.FITSCatalog(tmp_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
        f.save_as_fits(map_tmp, image=True, image_header=img_header, overwrite=True)

        # Run ShapeLens
        run_shapelens(tmp_name, psf_map_name, img_size, grid, key, opt_dict)
        # run_shapelens_output(tmp_name, psf_map_name, img_size, grid, key, opt_dict)

        if not opt_dict['keep_img']:
            os.remove(tmp_name)

    if not opt_dict['keep_img']:
        os.remove(psf_map_name)


def create_out_image(img_header, img_size, n_obj, key, opt_dict, dtype='float32'):
    """
    """
    if key == 'test':
        file_name = '{}/{}{}.fits'.format(opt_dict['dirs']['noshear'], key, opt_dict['file_number_string'])
    else:
        file_name = '{}/{}{}.fits'.format(opt_dict['dirs'][key], key, opt_dict['file_number_string'])

    img_header.tofile(file_name)

    shape = tuple(img_header['NAXIS{0}'.format(ii)] for ii in range(1, img_header['NAXIS']+1))
    with open(file_name, 'rb+') as fobj:
        fobj.seek(len(img_header.tostring()) + (np.product(shape) * np.abs(img_header['BITPIX']//8)) - 1)
        fobj.write(b'\0')

    return fits.open(file_name, mode='update', memmap=True), file_name


def process_memmap(img_path, psf_path, opt_dict, img_size, n_obj, dtype='float32'):
    """Process

    Main function which handle all the processing.

    Parameters
    ----------
    img_path : str
        Path to the image
    psf_path : str
        Path to the PSF image
    opt_dict : dict
        Dictionnary with other options
    img_size : int
        Size of one stamp that composed the mapped image
    n_obj : int
        Total number of objects
    dtype : str
        type of the data (Default = 'float32')

    """

    # Load images
    map_img = fits.getdata(img_path, ext=0, memmap=True)
    map_psf = fits.getdata(psf_path, ext=0, memmap=True)
    img_header = fits.getheader(img_path, 0)

    # Create output image
    mcal_out_img = {}
    mcal_out_path = {}
    for key in opt_dict['TYPES']+['psf']:
        mcal_out_img[key], mcal_out_path[key] = create_out_image(img_header, img_size, n_obj, key, opt_dict, dtype=dtype)

    nx = ny = int(np.sqrt(n_obj))
    for i in range(ny):
        for j in range(nx):

            img = map_img[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size]
            psf = map_psf[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size]

            # Get noise level for the weight
            sigma_noise = mad(img)
            weight = np.ones_like(img) * 1./sigma_noise**2.

            # Create metacal images
            obs_mcal = make_metacal(img, psf, weight, opt_dict)

            new_psf = obs_mcal['noshear'].get_psf().galsim_obj
            new_psf_img = new_psf.drawImage(nx=img_size, ny=img_size, scale=opt_dict['pixel_scale']).array
            mcal_out_img['psf'][0].data[j*img_size:(j+1)*img_size, i*img_size:(i+1)*img_size] = new_psf_img

            for key in opt_dict['TYPES']:
                mcal_out_img[key][0].data[j*img_size:(j+1)*img_size, i*img_size:(i+1)*img_size] = obs_mcal[key].image

    for key in opt_dict['TYPES']+['psf']:
        mcal_out_img[key].close()

    for key in opt_dict['TYPES']:
        print('running Shapelens on : {}'.format(key))

        # Run ShapeLens
        run_shapelens(mcal_out_path[key], mcal_out_path['psf'], img_size, nx, key, opt_dict)
        # run_shapelens_output(tmp_name, psf_map_name, img_size, grid, key, opt_dict)

        if not opt_dict['keep_img']:
            os.remove(mcal_out_path[key])

    if not opt_dict['keep_img']:
        os.remove(mcal_out_path['psf'])



@module_runner(version='0.0.1',
               file_pattern=['image', 'starfield'],
               file_ext=['.fits', '.fits'],
               depends=['numpy', 'ngmix', 'astropy'])
def shapelens_mcal_runner(input_file_list, run_dirs, file_number_string,
                         config, w_log):

    opt_dict = {}
    img_size = config.getint('SHAPELENS_MCAL_RUNNER', 'STAMP_SIZE')
    n_obj = config.getint('SHAPELENS_MCAL_RUNNER', 'OBJECT_NUMBER')
    opt_dict['pixel_scale'] = config.getfloat('SHAPELENS_MCAL_RUNNER', 'PIXEL_SCALE')

    format_mcal = config.getint('SHAPELENS_MCAL_RUNNER','MCAL_IMAGE_FORMAT')


    opt_dict['keep_img'] = config.getboolean('SHAPELENS_MCAL_RUNNER', 'KEEP_MCAL_IMG')

    opt_dict['TYPES'] = config.getlist('SHAPELENS_MCAL_RUNNER', 'TYPES')
    opt_dict['STEP'] = config.getfloat('SHAPELENS_MCAL_RUNNER', 'STEP')
    opt_dict['PSF_KIND'] = config.get('SHAPELENS_MCAL_RUNNER', 'PSF_MODEL')
    opt_dict['FIXNOISE'] = config.getboolean('SHAPELENS_MCAL_RUNNER', 'FIXNOISE')
    opt_dict['CHEATNOISE'] = config.getboolean('SHAPELENS_MCAL_RUNNER', 'CHEATNOISE')
    if opt_dict['FIXNOISE'] and opt_dict['CHEATNOISE']:
        raise ValueError('Either cheatnoise or fixnoise can be enabled')

    opt_dict['execute'] = config.getexpanded('SHAPELENS_MCAL_RUNNER', 'PATH')
    opt_dict['shapelens_opt'] = config.get('SHAPELENS_MCAL_RUNNER', 'OPTIONS')

    opt_dict['file_number_string'] = file_number_string
    opt_dict['dirs'] = {}
    for key in opt_dict['TYPES'] + ['psf']:
        opt_dict['dirs'][key] = run_dirs['output'] + '/{}'.format(key)
        try:
            os.mkdir(opt_dict['dirs'][key])
        except:
            print("Directory {} already exist.".format(opt_dict['dirs'][key]))

    process_memmap(*input_file_list, opt_dict, img_size, n_obj, 'float{}'.format(format_mcal))

    return None, None
