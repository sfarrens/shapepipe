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

import numpy as np

import ngmix


##########################
# DEBUG : Testing galaxy #
##########################

from tqdm import tqdm

import galsim

def get_gal():
    """
    About as simple as it gets:
      - Use a circular Gaussian profile for the galaxy.
      - Convolve it by a circular Gaussian PSF.
      - Add Gaussian noise to the image.
    """

    gal_flux = 1.e5    # total counts on the image
    gal_sigma = 4.     # arcsec
    psf_sigma = 3.     # arcsec
    pixel_scale = 1    # arcsec / pixel
    noise = 30.        # standard deviation of the counts in each pixel

    # Define the galaxy profile
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)

    # Apply shear
    #gal = gal.Shear(g1=?, g2=?)

    # Define the PSF profile
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma) # PSF flux should always = 1

    # Final profile is the convolution of these
    # Can include any number of things in the list, all of which are convolved
    # together to make the final flux profile.
    final = galsim.Convolve([gal, psf])

    # Draw the image with a particular pixel scale, given in arcsec/pixel.
    # The returned image has a member, added_flux, which is gives the total flux actually added to
    # the image.  One could use this value to check if the image is large enough for some desired
    # accuracy level.  Here, we just ignore it.
    image = final.drawImage(nx=96, ny=96, scale=pixel_scale)
    gal_image = gal.drawImage(nx=96, ny=96, scale=pixel_scale)
    psf_image = psf.drawImage(nx=96, ny=96, scale=pixel_scale)

    # Add Gaussian noise to the image with specified sigma
    image.addNoise(galsim.GaussianNoise(sigma=noise))
    #image.addNoiseSNR(galsim.GaussianNoise(sigma=noise), snr=?)


    results = image.FindAdaptiveMom()

    print('HSM reports that the image has observed shape and size:')
    print('    e1 = {:.3f}, e2 = {:.3f}, sigma = {:.3f} (pixels)'.format(results.observed_shape.e1,
                results.observed_shape.e2, results.moments_sigma))
    print('Expected values in the limit that pixel response and noise are negligible:')
    print('    e1 = {:.3f}, e2 = {:.3f}, sigma = {:.3f}'.format(0.0, 0.0,
                np.sqrt(gal_sigma**2 + psf_sigma**2)/pixel_scale))

    return image.array, psf_image.array, gal_image.array


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


def map_vignet(img_arr):
    """Map vignet

    Map vignet on one single image.

    Parameters
    ----------
    img_arr : numpy.ndarray
        Array of vinget to map

    """

    n_obj = img_arr.shape[0]
    xs = img_arr[0].shape[0]
    ys = img_arr[0].shape[1]

    nx = int(np.sqrt(n_obj))
    if nx*nx != n_obj:
        nx += 1
    ny = nx

    img_map=np.ones((xs*nx,ys*ny))

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

def psf_fitter(psf_vign):
    """Psf fitter

    Function used to create a gaussian fit of the PSF.

    Parameters
    ----------
    psf_vign : numpy.array
        Array containg one vignet of psf

    """

    psf_obs=ngmix.Observation(psf_vign)
    pfitter=ngmix.fitting.LMSimple(psf_obs,'gauss')

    shape = psf_vign.shape
    psf_pars = np.array([0., 0., 0., 0., 4., 1.])
    pfitter.go(psf_pars)

    psf_gmix_fit=pfitter.get_gmix()
    psf_obs.set_gmix(psf_gmix_fit)

    return psf_obs


def make_metacal(gal_vign, psf_vign, weight_vign, option_dict):
    """Make the metacalibration

    This function call different ngmix functions to create images needed for the metacalibration.

    Parameters
    ----------
    gal_vign : numpy.array
        Array containing one vignet of galaxy
    psf_vign : numpy.array
        Array containg one vignet of psf
    option_dict : dict
        Dictionnary containg option for ngmix (keys : ['TYPES', 'FIXNOISE', 'CHEATNOISE', 'SYMMETRIZE_PSF', 'STEP'])

    """

    psf_obs = psf_fitter(psf_vign)

    obs = ngmix.Observation(gal_vign, psf=psf_obs, weight=weight_vign)

    obs_out = ngmix.metacal.get_all_metacal(obs,
                                            types=option_dict['TYPES'],
                                            fixnoise=option_dict['FIXNOISE'],
                                            cheatnoise=option_dict['CHEATNOISE'],
                                            step=option_dict['STEP'],
                                            psf=option_dict['PSF_KIND'])

    return obs_out


#############
# ShapeLens #
#############

def run_shapelens(img_path, psf_path, img_size, grid, opt_dict):
    """
    """

    pass

    # if

    # command_line = '{} {} -p {} -s {} -g {} {}'.format(opt_dict['execute'],
    #                                                    img_path,
    #                                                    psf_path,
    #                                                    img_size,
    #                                                    grid,
    #                                                    opt_dict['shapelens_opt'])
    #
    # stderr, stdout = execute(command_line)
    #
    # print(stderr)
    # print(stdout)



def process(img_path, psf_path, opt_dict, img_size, n_obj, dtype='float32'):
    """
    """

    # Load images
    map_img = fits.getdata(img_path, 0)
    map_psf = fits.getdata(psf_path)
    img_header = fits.getheader(img_path, 0)

    # Parse the data
    # From mapped image to vignets
    img_arr = parse_data(map_img, img_size, 100)
    psf_arr = parse_data(map_psf, img_size, 100)

    mcal_dict = {key: np.zeros((n_obj, img_size, img_size), dtype=dtype) for key in opt_dict['TYPES'] + ['psf']}
    img_shape = np.array(img_arr[0].shape)

    for img, psf, i in tqdm(zip(img_arr, psf_arr, range(n_obj)), total=n_obj):

        # Get noise level for the weight
        sigma_noise = mad(img)
        weight = np.ones_like(img) * 1./sigma_noise**2.

        # Create metacal images
        obs_mcal = make_metacal(img, psf, weight, opt_dict)

        new_psf = obs_mcal['noshear'].get_psf().galsim_obj
        mcal_dict['psf'][i] = new_psf.drawImage(nx=img_shape[0],ny=img_shape[1],scale=1).array
        for key in opt_dict['TYPES']:
            mcal_dict[key][i] = obs_mcal[key].image

    # Map the metacal images and save them and run shapelens
    # Save PSF first
    psf_map_name = '{}/psf{}.fits'.format(opt_dict['dirs']['psf'], opt_dict['file_number_string'])
    map_tmp, grid = map_vignet(mcal_dict['psf'])
    f = io.FITSCatalog(psf_map_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
    f.save_as_fits(map_tmp, image=True, image_header=img_header, overwrite=True)

    for key in opt_dict['TYPES']:
        map_tmp, grid = map_vignet(mcal_dict[key])
        tmp_name = '{}/{}{}.fits'.format(opt_dict['dirs'][key], key, opt_dict['file_number_string'])
        f = io.FITSCatalog(tmp_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
        f.save_as_fits(map_tmp, image=True, image_header=img_header, overwrite=True)

        # Run ShapeLens
        run_shapelens(tmp_name, psf_map_name, img_size, grid, opt_dict)


    return mcal_dict




@module_runner(input_module=['sextractor_runner', 'psfexinterp_runner',
                             'vignetmaker_runner'],
               version='0.0.1',
               file_pattern=['tile_sexcat', 'image', 'exp_background',
                             'galaxy_psf', 'weight', 'flag'],
               file_ext=['.fits', '.sqlite', '.sqlite', '.sqlite', '.sqlite',
                         '.sqlite'],
               depends=['numpy', 'ngmix', 'galsim'])
def galsim_shapes_runner(input_file_list, run_dirs, file_number_string,
                         config, w_log):

    output_name = (run_dirs['output'] + '/' + 'galsim' +
                   file_number_string + '.fits')

    # f_wcs_path = config.getexpanded('NGMIX_RUNNER', 'LOG_WCS')

    metacal_res = process(*input_file_list, w_log)
    res_dict = compile_results(metacal_res)
    save_results(res_dict, output_name)

    return None, None
