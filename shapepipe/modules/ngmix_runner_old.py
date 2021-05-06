# -*- coding: utf-8 -*-

""" NGMIX RUNNER

This file contains methods to run ngmix for shape measurement.

:Author: Axel Guinot

"""

from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io
from sqlitedict import SqliteDict

import re
import os

import numpy as np
from numpy.random import uniform as urand

from modopt.math.stats import sigma_mad

import ngmix
from ngmix.observation import Observation, ObsList, MultiBandObsList
#from ngmix.fitting import LMSimple

from astropy.io import fits

import galsim

##
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse



def MegaCamFlip(vign, ccd_nb):
    """ MegaCam Flip
    MegaCam has CCD that are upside down.
    This function flip the CCDs accordingly.

    Parameters
    ----------
    vign : numpy.ndarray
        Array containing the postage stamp to flip.
    ccd_nb : int
        Id of the ccd containing the postage stamp.

    Return
    ------
    vign : numpy.ndarray
        The postage stamp flip accordingly.

    """

    if ccd_nb < 18 or ccd_nb in [36, 37]:
        # swap x axis so origin is on top-right
        return np.rot90(vign, k=2)
    else:
        # swap y axis so origin is on bottom-left
        return vign


def get_prior(rng, g_sigma = 0.3, galsim=True):
    """ Get prior

    Return prior for the different parameters

    Return
    ------
    prior : ngmix.priors
        Priors for the different parameters.

    """

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
    g_sigma = g_sigma
    g_prior = ngmix.priors.GPriorBA(sigma=g_sigma, rng=rng)

    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which
    # would be zero)
    # and the sigma of the gaussians
    # units same as jacobian, probably arcsec
    row, col = 0.0, 0.0
    row_sigma, col_sigma = 0.186, 0.186  # pixel size of DES
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

    # similar for flux.  Make sure the bounds make sense for
    # your images
    Fminval = -1.e2
    Fmaxval = 1.e9
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    if galsim:
        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)
        Tminval = -1.0  # arcsec squared
        Tmaxval = 1.e2
        r50_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

        # now make a joint prior.  This one takes priors
        # for each parameter separately
        prior = ngmix.joint_prior.PriorGalsimSimpleSep(cen_prior,
                                                 g_prior,
                                                 r50_prior,
                                                 F_prior)
    else:
        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)
        Tminval = -1.0  # arcsec squared
        Tmaxval = 1.e2
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

        # now make a joint prior.  This one takes priors
        # for each parameter separately
        prior = ngmix.joint_prior.PriorSimpleSep(cen_prior,
                                                 g_prior,
                                                 T_prior,
                                                 F_prior)

    return prior


def get_guess(img, pixel_scale=0.187,
              guess_flux_unit='img',
              guess_size_type='T', guess_size_unit='sky',
              guess_centroid=True, guess_centroid_unit='sky'):
    """Get guess

    Get the guess vector for the ngmix shape measurement
    [center_x, center_y, g1, g2, size_T, flux]
    No guess are given for the ellipticity (0., 0.)

    Parameters
    ----------
    img : numpy.ndarray
        Array containing the image
    pixel_scale : float
        Approximation of the pixel scale
    guess_flux_unit : string
        If 'img' return the flux in pixel unit
        if 'sky' return the flux in arcsec^-2
    guess_size_type : string
        if 'T' return the size in quadrupole moments definition (2 * sigma**2)
        if 'sigma' return moments sigma
    guess_size_unit : string
        If 'img' return the size in pixel unit
        if 'sky' return the size in arcsec
    guess_centroid : bool
        If True, will return a guess on the object centroid
        if False, will return the image center
    guess_centroid_unit : string
        If 'img' return the centroid in pixel unit
        if 'sky' return the centroid in arcsec

    Returns
    -------
    guess : numpy.ndarray
        Return the guess array : [center_x, center_y, g1, g2, size_T, flux]
    """

    galsim_img = galsim.Image(img, scale=pixel_scale)

    hsm_shape = galsim.hsm.FindAdaptiveMom(galsim_img, strict=False)

    error_msg = hsm_shape.error_message

    if error_msg != '':
        raise galsim.hsm.GalSimHSMError('Error in adaptive moments :\n{}'.format(error_msg))

    if guess_flux_unit == 'img':
        guess_flux = hsm_shape.moments_amp
    elif guess_flux_unit == 'sky':
        guess_flux = hsm_shape.moments_amp/pixel_scale**2.
    else:
        raise ValueError("guess_flux_unit must be in ['img', 'sky'], got : {}".format(guess_flux_unit))

    if guess_size_unit == 'img':
        size_unit = 1.
    elif guess_size_unit == 'sky':
        size_unit = pixel_scale
    else:
        raise ValueError("guess_size_unit must be in ['img', 'sky'], got : {}".format(guess_size_unit))

    if guess_size_type == 'sigma':
        guess_size = hsm_shape.moments_sigma*size_unit
    elif guess_size_type == 'T':
        guess_size = 2.*(hsm_shape.moments_sigma*size_unit)**2.

    if guess_centroid_unit == 'img':
        centroid_unit = 1.
    elif guess_centroid_unit == 'sky':
        centroid_unit = pixel_scale
    else:
        raise ValueError("guess_centroid_unit must be in ['img', 'sky'], got : {}".format(guess_centroid_unit))

    if guess_centroid:
        guess_centroid = (hsm_shape.moments_centroid-galsim_img.center) * centroid_unit
    else:
        guess_centroid = galsim_img.center * centroid_unit

    guess = np.array([guess_centroid.x,
                      guess_centroid.y,
                      0., 0.,
                      guess_size,
                      guess_flux])
    #guess = np.array([guess_centroid.x,
    #                  guess_centroid.y,
    #                  0., 0.,
    #                  1.,
    #                  guess_flux])

    #print('Guess')
    #print(hsm_shape.observed_shape.g1, hsm_shape.observed_shape.g2)

    return guess


def make_galsimfit(obs, model, guess0, rng, prior=None, lm_pars=None, ntry=5):
    """
    """

    guess = np.copy(guess0)
    fres = {}
    for it in range(ntry):
        #guess[0:5] += rng.uniform(low=-0.1, high=0.1)
        #guess[5:] *= (1. + rng.uniform(low=-0.1, high=0.1))
        fres['flags'] = 1
        try:
            #fitter = ngmix.fitting.galsim_fitters.GalsimFitter(model,
            #                                prior=prior,
            #                                fit_pars=lm_pars)
            fitter = ngmix.fitting.Fitter(model,
                                          prior=prior,
                                          fit_pars=lm_pars)
            fres = fitter.go(obs, guess)

            # Old runner
            #fitter = ngmix.GalsimSimple(obs,
            #                            model,
            #                            prior=prior,
            #                            lm_pars=lm_pars)
            #fitter.go(guess)
            #fres = fitter.get_result()
        except:
            continue

        if fres['flags'] == 0:
            break

    if fres['flags'] != 0:
        raise ngmix.gexceptions.BootGalFailure("Failes to fit galaxy with galsimfit")

    fres['ntry'] = it + 1

    return fres


def get_jacob(wcs, ra, dec):
    """ Get jacobian

    Return the jacobian of the wcs at the required position.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS object for wich we want the jacobian.
    ra : float
        Ra position of the center of the vignet (in Deg).
    dec : float
        Dec position of the center of the vignet (in Deg).

    Returns
    -------
    galsim_jacob : galsim.wcs.BaseWCS.jacobian
        Jacobian of the WCS at the required position.

    """

    g_wcs = galsim.fitswcs.AstropyWCS(wcs=wcs)
    world_pos = galsim.CelestialCoord(ra=ra*galsim.angle.degrees,
                                      dec=dec*galsim.angle.degrees)
    galsim_jacob = g_wcs.jacobian(world_pos=world_pos)

    return galsim_jacob


def get_noise(gal, weight, guess, thresh=1.2, pixel_scale=0.187):
    """ Get Noise

    Compute the sigma of the noise from an object postage stamp.
    Use a guess on the object size, ellipticity and flux to create a window
    function.

    Parameters
    ----------
    gal : numpy.ndarray
        Galaxy image.
    weight : numpy.ndarray
        Weight image.
    guess : list
        Gaussian parameters fot the window function: [x0, y0, g1, g2, T, flux]
    thresh : float
        Threshold to cut the window function. Cut = thresh * sig_noise
    pixel_scale = float
        Pixel scale of the galaxy image

    Return
    ------
    sig_noise : float
        Sigma of the noise on the galaxy image.

    """

    img_shape = gal.shape

    m_weight = weight != 0

    sig_tmp = sigma_mad(gal[m_weight])

    gauss_win = galsim.Gaussian(sigma=np.sqrt(guess[4]/2.), flux=guess[5])
    gauss_win = gauss_win.shear(g1=guess[2], g2=guess[3])
    gauss_win = gauss_win.drawImage(nx=img_shape[0], ny=img_shape[1], scale=pixel_scale).array

    m_weight = weight[gauss_win < thresh*sig_tmp] != 0

    sig_noise = sigma_mad(gal[gauss_win < thresh*sig_tmp][m_weight])

    return sig_noise


#def make_fake_gals(psfs, jacob_list, bkg_list, g1=0., g2=0.):
#    """
#    """

#    #g1, g2 = np.random.normal(size=2)*0.3
#    gauss = galsim.Gaussian(half_light_radius=0.001).withFlux(5000).shear(g1=g1, g2=g2)
#    all_img = []
#    all_psf = []
#    #all_jacob = []
#    for psf, bkg, jacob in zip(psfs, bkg_list, jacob_list):
#        pixel = jacob.toWorld(galsim.Pixel(scale=1))
#        pixel_inv = galsim.Deconvolve(pixel)

#        #wcs = galsim.PixelScale(0.187)
#        #jacob = wcs.jacobian()
#        gal_psf_int = galsim.InterpolatedImage(galsim.Image(psf), wcs=jacob)
#        gal_psf = galsim.Convolve((gal_psf_int, pixel_inv))
#        #gal_psf = galsim.Gaussian(fwhm=0.64).withFlux(1)

#        obj = galsim.Convolve([gauss, gal_psf])
#        img = obj.drawImage(nx=51, ny=51, wcs=jacob)

#        img_psf = gal_psf.drawImage(nx=51, ny=51, wcs=jacob)

#        img.addNoise(galsim.PoissonNoise(rng=galsim.BaseDeviate(), sky_level=bkg))
#        all_img.append(img.array)

#        all_psf.append(img_psf.array)

#        #all_jacob.append(jacob)
#    return all_img, all_psf#, all_jacob



def make_gal(flux=10000, g1=0., g2=0., r_50=0.5,
             psf_fwhm=0.6, bkg=300,
             dx=0, dy=0,
             dudx=0.187, dudy=0., dvdx=0., dvdy=-0.187,
             rng=None):
    """
    """

    gal = galsim.Gaussian(half_light_radius=r_50, flux=flux)
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.Gaussian(fwhm=psf_fwhm)

    wcs=galsim.JacobianWCS(dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)

    obj = galsim.Convolve((gal, psf))
    offset = galsim.PositionD(dx, dy)
    obj_img = obj.drawImage(nx=51, ny=51, wcs=wcs, offset=offset,
                            method='auto',)# rng=rng)

    psf_img = psf.drawImage(nx=51, ny=51, wcs=wcs)

    noise = galsim.PoissonNoise(rng=rng, sky_level=bkg)

    obj_img.addNoise(noise)

    return obj_img.array, psf_img.array, wcs


def make_star(flux=10000, g1=0., g2=0., r_50=0.5,
              psf_fwhm=0.6, bkg=300,
              dx=0, dy=0,
              dudx=0.187, dudy=0., dvdx=0., dvdy=-0.187,
              rng=None):
    """
    """

    #star = galsim.Gaussian(fwhm=0.00001).withFlux(flux)

    psf = galsim.Gaussian(fwhm=psf_fwhm).withFlux(flux)

    wcs=galsim.JacobianWCS(dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)

    #obj = galsim.Convolve((star, psf))
    obj = psf
    offset = galsim.PositionD(dx, dy)
    obj_img = obj.drawImage(nx=51, ny=51, wcs=wcs, offset=offset,
                            method='auto')#, rng=rng)

    psf_img = psf.drawImage(nx=51, ny=51, wcs=wcs)

    noise = galsim.PoissonNoise(rng=rng, sky_level=bkg)

    obj_img.addNoise(noise)

    return obj_img.array, psf_img.array, wcs


def make_fake_gals(np_rng,
                   n_epoch=1, n_obj=1,
                   psf_fwhm=0.6,
                   pixel_scale=0.187, sigma_wcs=0.01,
                   flux_range=[10000, 10000],
                   r_50_range = [0.3, 0.5],
                   bkg_range=[250, 250],
                   offset_range=[-0.5, 0.5],
                   verbose=False):
    """
    """

    galsim_rng = galsim.BaseDeviate(seed=1234)
   
    gals = []
    psfs = []
    flags = []
    weights = []
    psfs_sigma = []
    jacob_list = []
    offset_list = []
    bkg_list = []
    flux_gal = int(np_rng.uniform(low=flux_range[0], high=flux_range[1]))
    r_50_gal = np_rng.uniform(low=r_50_range[0], high=r_50_range[1])
    
    #g1_shear = 0.02
    #g2_shear = 0.
    #g1_int, g2_int = np_rng.normal(size=2) * 0.3
    #g1_gal = g1_shear + g1_int
    #g2_gal = g2_shear + g2_int
    #while np.abs(g1_gal + g2_gal*1j) > 1:
    #    g1_int, g2_int = np_rng.normal(size=2) * 0.3
    #    g1_gal = g1_shear + g1_int
    #    g2_gal = g2_shear + g2_int
    g1_gal = 0.02
    g2_gal = 0.1
    #bkg = int(np_rng.uniform(low=bkg_range[0], high=bkg_range[1]))
    for n_e in range(n_epoch):
        if verbose:
            print('Epoch {}\n'.format(n_e))
        #dx = np_rng.uniform(low=offset_range[0],
        #                         high=offset_range[1]) #* pixel_scale
        #dy = np_rng.uniform(low=offset_range[0],
        #                         high=offset_range[1]) #* pixel_scale
        dx = dy = 0
        bkg = int(np_rng.uniform(low=bkg_range[0], high=bkg_range[1]))
        #g1_wcs, g2_wcs = np_rng.normal(size=2) * sigma_wcs
        g1_wcs = g2_wcs = 0
        wcs = galsim.ShearWCS(scale=pixel_scale,
                              shear=galsim.Shear(g1=g1_wcs, g2=g2_wcs)).jacobian()
        if verbose:
            print('wcs : ', wcs)
            print('Background : ', bkg)
            print('Flux : ', flux_gal)
            print('dx :', dx, 'dy :', dy)

        gal_tmp , psf_tmp, wcs_tmp = make_gal(flux=flux_gal,
                                                g1=g1_gal,
                                                g2=g2_gal,
                                                r_50 = r_50_gal,
                                                psf_fwhm=psf_fwhm,
                                                bkg=bkg,
                                                dx=dx,
                                                dy=dy,
                                                dudx=wcs.dudx,
                                                dudy=wcs.dudy,
                                                dvdx=wcs.dvdx,
                                                dvdy=wcs.dvdy,
                                                rng=galsim_rng)
        gals.append(gal_tmp)
        psfs.append(psf_tmp)
        flags.append(np.zeros_like(gal_tmp))
        weights.append(np.ones_like(gal_tmp))
        psfs_sigma.append(0.6/2.355/0.187)
        jacob_list.append(wcs_tmp)
        offset_list.append((dx, dy))
        bkg_list.append(bkg)

    print('True g1,g2: {:.4f} {:.4f}'.format(g1_gal, g2_gal))

    return gals, psfs, flags, weights, jacob_list, psfs_sigma, offset_list, bkg_list, flux_gal, r_50_gal, np.array([g1_gal, g2_gal])


# def do_ngmix_metacal(gals, psfs, psfs_sigma, weights, flags, jacob_list, offset_list,
#                      bkg_list, prior, id_tmp, psf_hsm_shapes, tile_flux, tile_r50, rng):
#     """ Do ngmix metacal

#     Do the metacalibration on a multi-epoch object and return the join shape
#     measurement with ngmix

#     Parameters
#     ---------
#     gals : list
#         List of the galaxy vignets.
#     psfs : list
#         List of the PSF vignets.
#     psfs_sigma : list
#         List of the sigma PSFs.
#     weights : list
#         List of the weight vignets.
#     flags : list
#         List of the flag vignets.
#     jacob_list : list
#         List of the jacobians.
#     prior : ngmix.priors
#         Priors for the fitting parameters.

#     Returns
#     -------
#     metacal_res : dict
#         Dictionary containing the results of ngmix metacal.

#     """

#     n_epoch = len(gals)
#     n_epoch = 3

#     #gals, psfs = make_fake_gals(psfs, jacob_list, bkg_list)
#     gals, psfs, flags, weights, jacob_list, psfs_sigma, offset_list, bkg_list, tile_flux, tile_r50 = make_fake_gals(rng, n_epoch)
#     #gals = np.copy(psfs)

#     pixel_scale = 0.187

#     if n_epoch == 0:
#         raise ValueError("0 epoch to process")

#     # Make observation
#     gal_obs_list = ObsList()
#     T_guess_psf = []
#     psf_res_gT = {'g_PSFo': np.array([0., 0.]),
#                   'g_err_PSFo': np.array([0., 0.]),
#                   'T_PSFo': 0.,
#                   'T_err_PSFo': 0.}
#     gal_guess = []
#     gal_img = []
#     gal_guess_flag = True
#     wsum = 0.
#     for n_e in range(n_epoch):
#         psf_jacob = ngmix.Jacobian(x=(psfs[0].shape[0]-1)/2. + 0.5,
#                                    y=(psfs[0].shape[1]-1)/2. + 0.5,
#                                    dudx=jacob_list[n_e].dudx,
#                                    dudy=jacob_list[n_e].dudy,
#                                    dvdx=jacob_list[n_e].dvdx,
#                                    dvdy=jacob_list[n_e].dvdy)
        
#         # PSF noise
#         psf_noise = np.sqrt(np.sum(psfs[n_e]**2)) / 50000
#         #print('PSF noise : {}'.format(psf_noise))
#         psf_weight = np.ones_like(psfs[n_e]) / psf_noise**2

#         #psfs[n_e] += (np.random.normal(size=psfs[n_e].shape) * psf_noise)

#         #psf_model = galsim.Gaussian(sigma=psf_hsm_shapes[n_e]['SIGMA_PSF_HSM']*0.187).withFlux(1).shear(g1=psf_hsm_shapes[n_e]['E1_PSF_HSM'], g2=psf_hsm_shapes[n_e]['E2_PSF_HSM'])
#         #psf_im = psf_model.drawImage(nx=51, ny=51, scale=0.187).array

#         #psf_im = _make_tapering(psfs[n_e]/np.sum(psfs[n_e]), 0.6)
#         #psf_obs = Observation(psf_im, jacobian=psf_jacob, weight=psf_weight)
#         psf_obs = Observation(psfs[n_e]/np.sum(psfs[n_e]), jacobian=psf_jacob, weight=psf_weight)

#         psf_T = 2. * psfs_sigma[n_e]**2.
#         #psf_T = psfs_sigma[n_e]*1.17741*pixel_scale

#         #w = np.copy(weights[n_e])
#         #w[np.where(flags[n_e] != 0)] = 0.
#         #w[w != 0] = 1

#         psf_guess = np.array([0., 0., 0., 0., psf_T, 1.])
#         try:
#             psf_res = make_galsimfit(psf_obs, 'gauss', psf_guess, rng, None)

#             # Set PSF GMix object
#             pars_psf = psf_res['pars']
#             #pars_psf[4] = (pars_psf[4]/1.17741)**2. * 2.
#             psf_gmix = ngmix.GMixModel(pars_psf, 'gauss')
#             #psf_obs.set_gmix(psf_gmix)
#         except:
#             continue

#         # Original PSF fit
#         #print('PSF fit')
#         #print(psf_res['pars'][2:4])
#         sig_noise = bkg_list[n_e]
#         w = np.ones_like(gals[n_e]) * 1/sig_noise
#         w_tmp = np.sum(w)
#         psf_res_gT['g_PSFo'] += psf_res['g']*w_tmp
#         psf_res_gT['g_err_PSFo'] += np.array([psf_res['pars_err'][2], psf_res['pars_err'][3]])*w_tmp
#         psf_res_gT['T_PSFo'] += psf_res['T']*w_tmp
#         psf_res_gT['T_err_PSFo'] += psf_res['T_err']*w_tmp
#         wsum += w_tmp

#         # Noise handling
#         #if gal_guess_flag:
#         #    sig_noise = get_noise(gals[n_e], w, gal_guess_tmp, pixel_scale=pixel_scale)
#         #else:
#         #    sig_noise = sigma_mad(gals[n_e])
#         sig_noise = bkg_list[n_e]

#         noise_img = rng.normal(size=gals[n_e].shape)*np.sqrt(sig_noise)
#         noise_img_gal = rng.normal(size=gals[n_e].shape)*np.sqrt(sig_noise)

#         gal_masked = np.copy(gals[n_e])
#         if (len(np.where(flags[n_e] != 0)[0]) != 0):
#             gal_masked[flags[n_e] != 0] = noise_img_gal[flags[n_e] != 0]

#         #w *= 1/sig_noise
#         #w *= 0
#         #w += 1
#         w = np.ones_like(gal_masked) * 1/sig_noise

#         # Gal guess
#         try:
#             gal_guess_tmp = get_guess(gal_masked, pixel_scale=pixel_scale, guess_size_type='T', guess_centroid_unit='img')
#         except:
#             gal_guess_flag = False
#             gal_guess_tmp = np.array([0., 0., 0., 0., 1, 100])

#         # Recenter jacobian if necessary
#         #gal_jacob = ngmix.Jacobian(x=(gals[0].shape[0]-1)/2.,# + gal_guess_tmp[0],
#         #                           y=(gals[0].shape[1]-1)/2.,# + gal_guess_tmp[1],
#         #                           dudx=jacob_list[n_e].dudx,
#         #                           dudy=jacob_list[n_e].dudy,
#         #                           dvdx=jacob_list[n_e].dvdx,
#         #                           dvdy=jacob_list[n_e].dvdy)
#         gal_jacob = ngmix.Jacobian(x=(gals[0].shape[0]-1)/2. + offset_list[n_e][0] + 0.5,# + offset_list[n_e][0],
#                                    y=(gals[0].shape[1]-1)/2. + offset_list[n_e][1] + 0.5,# + offset_list[n_e][1],
#                                    dudx=jacob_list[n_e].dudx,
#                                    dudy=jacob_list[n_e].dudy,
#                                    dvdx=jacob_list[n_e].dvdx,
#                                    dvdy=jacob_list[n_e].dvdy)
#         #print('Epoch {}'.format(n_e))
#         #print('Moment shift : {:.3f} {:.3f}'.format(*(gal_guess_tmp[0:2])))
#         #print('"Real" shift : {:.3f} {:.3f}'.format(offset_list[n_e][0]+0.5, offset_list[n_e][1]+0.5))
#         #print('SEP shift : {:.3f} {:.3f}'.format(*get_sep_offset(gals[n_e], bkg_list[n_e])))

#         #gal_obs = Observation(gals[n_e], weight=w, jacobian=gal_jacob,
#         #                      psf=psf_obs)
#         # offset = galsim.PositionD(*(gal_guess_tmp[:2]*-1))
#         #offset = galsim.PositionD(0., 0.)
#         #gal_int = galsim.InterpolatedImage(galsim.Image(gal_masked, wcs=jacob_list[n_e]), x_interpolant='lanczos15')
#         # img_shape = gal_masked.shape
#         # gal_img_tmp = gal_int.drawImage(nx=img_shape[0], ny=img_shape[1],
#         #                                 offset=offset, wcs=jacob_list[n_e],
#         #                                 method="no_pixel").array
#         gal_img_tmp = gal_masked
#         gal_img.append(gal_img_tmp)
#         gal_obs = Observation(gal_img_tmp, weight=w, jacobian=gal_jacob,
#                               psf=psf_obs, noise=noise_img,
#                               bmask=flags[n_e].astype(np.int32),
#                               ormask=flags[n_e].astype(np.int32))

#         if gal_guess_flag:
#             gal_guess_tmp[:2] = 0
#             #gal_guess_tmp[4] *= 1.17741
#             #gal_guess_tmp[4] = 2*gal_guess_tmp[4]**2.
#             #gal_guess_tmp[2:4] = 1e-3
#             #gal_guess_tmp[4] = tile_r50
#             gal_guess_tmp[5] = tile_flux
#             gal_guess.append(gal_guess_tmp)

#         gal_obs_list.append(gal_obs)
#         T_guess_psf.append(psf_T)
#         gal_guess_flag = True

#     #print(jacob_list[n_e].dudx, jacob_list[n_e].dudy, jacob_list[n_e].dvdx, jacob_list[n_e].dvdy)

#     if wsum == 0:
#         raise ZeroDivisionError('Sum of weights = 0, division by zero')

#     # Normalize PSF fit output
#     for key in psf_res_gT.keys():
#         psf_res_gT[key] /= wsum

#     Tguess = np.mean(T_guess_psf)
    
#     #print()
#     #print("ME PSF fit")
#     #print(psf_res_gT)

#     # Gal guess handling
#     fail_get_guess = False
#     if len(gal_guess) == 0:
#         fail_get_guess = True
#         gal_pars = [0., 0., 0., 0., Tguess, 100]
#     else:
#         #gal_pars = [0., 0., 0., 0., Tguess, gal_guess[-1]]
#         gal_pars = np.mean(gal_guess, 0)
#         #if gal_pars[-2] < Tguess:
#         #gal_pars[-2] = Tguess
#             #gal_pars[-1] = 5000


#     # boot = ngmix.bootstrap.MaxMetacalBootstrapper(gal_obs_list)

#     psf_model = 'gauss'
#     gal_model = 'gauss'

#     #psf_reconv = {'model': 'gauss', 'pars': {'fwhm': np.mean(T_guess_psf)/1.17741*2.355*1.1}}
#     #psf_reconv = {'model': 'moffat', 'pars': {'fwhm': np.mean(T_guess_psf)/1.17741*2.355*1.1, 'beta': 4.5}}
#     psf_reconv = 'gauss'

#     # metacal specific parameters
#     metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'],
#                     'step': 0.01,
#                     'psf': psf_reconv,
#                     'fixnoise': True,
#                     'use_noise_image': True,
#                     'rng': rng}
#     #metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'],
#     #                'step': 0.01,
#     #                'fixnoise': True,
#     #                'cheatnoise': False,
#     #                'symmetrize_psf': True,
#     #                'symmetrize_tapering': True,
#     #                'tapering_alpha': 0.8,
#     #                'use_noise_image': True}

#     # maximum likelihood fitter parameters
#     # parameters for the Levenberg-Marquardt fitter in scipy
#     # lm_pars = {'maxfev': 2000,
#     #            'xtol': 5.0e-5,
#     #            'ftol': 5.0e-5}
#     # max_pars = {
#     #     # use scipy.leastsq for the fitting
#     #     'method': 'lm',

#     #     # parameters for leastsq
#     #     'lm_pars': lm_pars}

#     # # psf_pars = {'maxiter': 5000,
#     # #             'tol': 5.0e-6}
#     # psf_pars = {'maxfev': 5000,
#     #            'xtol': 5.0e-6,
#     #            'ftol': 5.0e-6}

#     # Tguess = np.mean(T_guess_psf)*0.186**2  # size guess in arcsec

#     #print('Tguess : {}'.format(Tguess))
#     #print('Tguess dilate : {}'.format(Tguess*1.02))
#     #print('Gal guess : {}'.format(gal_pars))

#     # Tguess = 4.0*0.186**2
#     # ntry = 2       # retry the fit twice

#     obs_dict_mcal = ngmix.metacal.get_all_metacal(gal_obs_list, **metacal_pars)
#     res = {'mcal_flags': 0}
#     obs_dict_mcal['noshear'] = gal_obs_list

#     gal_pars = np.array([0., 0., 0., 0., (tile_r50/1.17741)**2 * 2, tile_flux])
#     #gal_pars = np.array([0., 0., 0., 0., 1., tile_flux])

#     ntry = 5

#     for key in sorted(obs_dict_mcal):

#         fres = make_galsimfit(obs_dict_mcal[key],
#                               gal_model, gal_pars,
#                               rng,
#                               prior=prior)

#         res['mcal_flags'] |= fres['flags']
#         tres = {}

#         for name in fres.keys():
#             tres[name] = fres[name]
#         tres['flags'] = fres['flags']

#         wsum = 0.0
#         Tpsf_sum = 0.0
#         gpsf_sum = np.zeros(2)
#         npsf = 0
#         for n_e, obs in enumerate(obs_dict_mcal[key]):

#             if hasattr(obs, 'psf_nopix'):
#                 try:
#                     psf_res = make_galsimfit(obs.psf_nopix,
#                                              psf_model,
#                                              np.array([0., 0., 0., 0., T_guess_psf[n_e], 1.]),
#                                              rng,
#                                              prior=None,
#                                              ntry=ntry)
#                 except:
#                     continue
#                 g1, g2 = psf_res['g']
#                 T = psf_res['T']
#             else:
#                 try:
#                     psf_res = make_galsimfit(obs.psf,
#                                              psf_model,
#                                              np.array([0., 0., 0., 0., T_guess_psf[n_e], 1.]),
#                                              rng,
#                                              prior=None,
#                                              ntry=ntry)
#                 except:
#                     continue
#                 g1, g2 = psf_res['g']
#                 T = psf_res['T']

#             # TODO we sometimes use other weights
#             twsum = obs.weight.sum()

#             wsum += twsum
#             gpsf_sum[0] += g1*twsum
#             gpsf_sum[1] += g2*twsum
#             Tpsf_sum += T*twsum
#             npsf += 1

#         tres['gpsf'] = gpsf_sum/wsum
#         tres['Tpsf'] = Tpsf_sum/wsum

#         res[key] = tres

    
#     print()
#     print('Gal guess')
#     print(gal_guess_tmp)
#     print('Final res')
#     print(res['noshear']['pars'])
#     print('Final res err')
#     print(np.sqrt(res['noshear']['pars_err']))
#     print('T psf: {}'.format(res['noshear']['Tpsf']))
#     print()
#     res_mcal = make_galsimfit(obs_dict_mcal['noshear'][0], gal_model, gal_pars, rng, prior=prior)
#     print('Galsim fit mcal')
#     print('g1 : {}\ng2 : {}'.format(res_mcal['pars'][2], res_mcal['pars'][3]))
#     res_no_mcal = make_galsimfit(gal_obs_list[0], gal_model, gal_pars, rng, prior=prior)
#     print('Galsim fit no mcal')
#     print('g1 : {}\ng2 : {}'.format(res_no_mcal['pars'][2], res_no_mcal['pars'][3]))
#     print('HSM mcal')
#     try:
#         s = galsim.hsm.FindAdaptiveMom(galsim.Image(obs_dict_mcal['noshear'][0].image, wcs=obs_dict_mcal['noshear'][0].jacobian.get_galsim_wcs()))
#         print('g1 : {}\ng2 : {}'.format(s.observed_shape.g1, s.observed_shape.g2))
#     except:
#         print('Fails')
#     print('HSM no mcal')
#     try:
#         s = galsim.hsm.FindAdaptiveMom(galsim.Image(gal_obs_list[0].image, wcs=gal_obs_list[0].jacobian.get_galsim_wcs()))
#         print('g1 : {}\ng2 : {}'.format(s.observed_shape.g1, s.observed_shape.g2))
#     except:
#         print('Fails')
#     print("True flux : {}".format(tile_flux))
#     print("Meas flux : {}".format(res['noshear']['flux']))
#     print("True r50 : {}".format(tile_r50))
#     print("Meas r50 : {}".format(np.sqrt(res['noshear']['pars'][4]/2)*1.17741))

#     ####
#     fig_dir = '/automnt/n17data/guinot/simu_W3/plot_ngmix/'
#     ####
#     plt.figure()
#     plt.imshow(obs_dict_mcal['noshear'][n_e].psf.image, cmap='gist_stern')
#     plt.colorbar()
#     plt.savefig(fig_dir + '/psf_metacal_{}_{}.png'.format(id_tmp, n_e))
#     s = galsim.hsm.FindAdaptiveMom(galsim.Image(obs_dict_mcal['noshear'][n_e].psf.image, scale=0.187))
#     #print(len(obs_dict_mcal[key]))
#     #print(T_guess_psf[n_e]/1.17741*2.355)
#     #print(s.observed_shape.g1, s.observed_shape.g2, s.moments_sigma*2.355*0.187)
#     #print(g1, g2, T/1.17741*2.355)
#     ####
#     plt.figure()
#     plt.imshow(gal_obs_list[n_e].psf.image, cmap='gist_stern')
#     plt.colorbar()
#     plt.savefig(fig_dir + '/psf_ori_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     psf_profile = np.sum(obs_dict_mcal['noshear'][n_e].psf.image, 0)
#     plt.figure()
#     plt.semilogy(psf_profile)
#     plt.savefig(fig_dir + '/psf_profile_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     plt.figure()
#     plt.imshow(flags[n_e])
#     plt.colorbar()
#     plt.savefig(fig_dir + '/flags_ori_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     plt.figure()
#     plt.imshow(w)
#     plt.colorbar()
#     plt.savefig(fig_dir + '/weight_ori_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     plt.figure()
#     plt.imshow(obs_dict_mcal['noshear'][n_e].image)
#     plt.colorbar()
#     plt.savefig(fig_dir + '/gal_metacal_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     plt.figure()
#     plt.imshow(obs_dict_mcal['1p'][n_e].image)
#     plt.colorbar()
#     plt.savefig(fig_dir + '/gal_1p_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     plt.figure()
#     plt.imshow(gal_img[0]-gal_img[-1])
#     plt.colorbar()
#     plt.savefig(fig_dir + '/gal_ori_stack_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     for n_e in range(n_epoch):
#         fig, ax = plt.subplots()
#         ax.imshow(gals[n_e])
#         s = galsim.Shear(g1=res['noshear']['pars'][2], g2=res['noshear']['pars'][3])
#         e = Ellipse(xy=(25.5, 25.5),
#                 width=6.,
#                 height=6*s.q,
#                 angle=s.beta.deg)
#         e.set_facecolor('none')
#         e.set_edgecolor('red')
#         ax.add_artist(e)
#         ax.plot(25.5+offset_list[n_e][0], 25.5+offset_list[n_e][1], 'k+')
#         ax.plot(25.5, 25.5, 'r+')
#         #plt.colorbar()
#         plt.savefig(fig_dir + '/gal_ori_{}_{}.png'.format(id_tmp, n_e))
#     ####
#     fig, ax = plt.subplots()
#     ax.imshow(gal_masked)
#     s = galsim.Shear(g1=res['noshear']['pars'][2], g2=res['noshear']['pars'][3])
#     e = Ellipse(xy=(25.5+res['noshear']['pars'][0], 25.5+res['noshear']['pars'][1]),
#                 width=6.,
#                 height=6*s.q,
#                 angle=s.beta.deg)
#     e.set_facecolor('none')
#     e.set_edgecolor('red')
#     ax.add_artist(e)
#     #plt.colorbar()
#     plt.savefig(fig_dir + '/gal_mask_{}_{}.png'.format(id_tmp, n_e))
    


#     # result dictionary, keyed by the types in metacal_pars above
#     metacal_res = res
#     metacal_res.update(psf_res_gT)
#     metacal_res['moments_fail'] = fail_get_guess

#     return metacal_res


def do_ngmix_metacal(gals, psfs, psfs_sigma, weights, flags, jacob_list, offset_list,
                     bkg_list, prior, id_tmp, psf_hsm_shapes, tile_flux, tile_r50, rng):
    """ Do ngmix metacal

    Do the metacalibration on a multi-epoch object and return the join shape
    measurement with ngmix

    Parameters
    ---------
    gals : list
        List of the galaxy vignets.
    psfs : list
        List of the PSF vignets.
    psfs_sigma : list
        List of the sigma PSFs.
    weights : list
        List of the weight vignets.
    flags : list
        List of the flag vignets.
    jacob_list : list
        List of the jacobians.
    prior : ngmix.priors
        Priors for the fitting parameters.

    Returns
    -------
    metacal_res : dict
        Dictionary containing the results of ngmix metacal.

    """

    #n_epoch = len(gals)
    n_epoch = 3

    #gals, psfs = make_fake_gals(psfs, jacob_list, bkg_list)
    gals, psfs, flags, weights, jacob_list, psfs_sigma, offset_list, bkg_list, tile_flux, tile_r50, true_g = make_fake_gals(rng, n_epoch)
    #gals = np.copy(psfs)

    #n_epoch = 1

    pixel_scale = 0.187

    if n_epoch == 0:
        raise ValueError("0 epoch to process")

    # Make observation
    gal_obs_list = ObsList()
    T_guess_psf = []
    psf_res_gT = {'g_PSFo': np.array([0., 0.]),
                  'T_PSFo': 0.}
    gal_guess = []
    gal_img = []
    gal_guess_flag = True
    wsum = 0.
    for n_e in range(n_epoch):
        psf_jacob = ngmix.Jacobian(x=(psfs[0].shape[0]-1)/2.,# + 0.5,
                                   y=(psfs[0].shape[1]-1)/2.,# + 0.5,
                                   dudx=jacob_list[n_e].dudx,
                                   dudy=jacob_list[n_e].dudy,
                                   dvdx=jacob_list[n_e].dvdx,
                                   dvdy=jacob_list[n_e].dvdy)
        
        # PSF noise
        psf_noise = np.sqrt(np.sum(psfs[n_e]**2)) / 50000
        #print('PSF noise : {}'.format(psf_noise))
        psf_weight = np.ones_like(psfs[n_e]) / psf_noise**2

        #psf_obs = Observation(psf_im, jacobian=psf_jacob, weight=psf_weight)
        psf_obs = Observation(psfs[n_e]/np.sum(psfs[n_e]), jacobian=psf_jacob, weight=psf_weight)

        psf_T = 2. * (psfs_sigma[n_e]*pixel_scale)**2.
        #psf_T = psfs_sigma[n_e]*1.17741*pixel_scale

        #psf_guess = np.array([0., 0., 0., 0., psf_T, 1.])
        #try:
        #    psf_res = make_galsimfit(psf_obs, 'gauss', psf_guess, rng, None)

        #    # Set PSF GMix object
        #    pars_psf = psf_res['pars']
        #    #pars_psf[4] = (pars_psf[4]/1.17741)**2. * 2.
        #    psf_gmix = ngmix.GMixModel(pars_psf, 'gauss')
        #    #psf_obs.set_gmix(psf_gmix)
        #except:
        #    continue

        

        # Original PSF fit
        #sig_noise = bkg_list[n_e]
        #w = np.ones_like(gals[n_e]) * 1/sig_noise
        #w_tmp = np.sum(w)
        #psf_res_gT['g_PSFo'] += psf_res['g']*w_tmp
        #psf_res_gT['g_err_PSFo'] += np.array([psf_res['pars_err'][2], psf_res['pars_err'][3]])*w_tmp
        #psf_res_gT['T_PSFo'] += psf_res['T']*w_tmp
        #psf_res_gT['T_err_PSFo'] += psf_res['T_err']*w_tmp
        #wsum += w_tmp

        # Noise handling
        sig_noise = bkg_list[n_e]
        #print("Sigma^2 noise : {}".format(sig_noise))

        #noise_img = rng.normal(size=gals[n_e].shape)*np.sqrt(sig_noise)
        #noise_img_gal = rng.normal(size=gals[n_e].shape)*np.sqrt(sig_noise)
        noise_img = rng.poisson(lam=sig_noise, size=gals[n_e].shape)-sig_noise
        noise_img_gal = rng.poisson(lam=sig_noise, size=gals[n_e].shape)-sig_noise

        # Weight
        weight_img = np.ones_like(gals[n_e]) * 1/sig_noise
        #print(weight_img[0,0])

        # Fit original PSF
        try:
            ntry = 5
            psf_ngauss = 5
            # psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss)
            #psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng,
            #                                               ngauss=psf_ngauss)
            psf_fitter = ngmix.em.EMFitter()
            psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng,
                                                        ngauss=psf_ngauss)
            psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, 
                                                 guesser=psf_guesser,
                                                 ntry=ntry)
            psf_res = psf_runner.go(psf_obs)
            psf_g1, psf_g2, psf_T = psf_res.get_gmix().get_g1g2T()
            
            wtmp = np.sum(weight_img)
            psf_res_gT['g_PSFo'] += np.array([psf_g1, psf_g2]) * wtmp
            psf_res_gT['T_PSFo'] += psf_T * wtmp
            wsum += wtmp
        except:
            continue

        # Handle masks
        gal_masked = np.copy(gals[n_e])
        if (len(np.where(flags[n_e] != 0)[0]) != 0):
            gal_masked[flags[n_e] != 0] = noise_img_gal[flags[n_e] != 0]
            weight_img[flags[n_e] != 0] = 0

        gal_jacob = ngmix.Jacobian(x=(gals[0].shape[0]-1)/2. + offset_list[n_e][0],# + 0.5,# + offset_list[n_e][0],
                                   y=(gals[0].shape[1]-1)/2. + offset_list[n_e][1],# + 0.5,# + offset_list[n_e][1],
                                   dudx=jacob_list[n_e].dudx,
                                   dudy=jacob_list[n_e].dudy,
                                   dvdx=jacob_list[n_e].dvdx,
                                   dvdy=jacob_list[n_e].dvdy)

        gal_img_tmp = gal_masked
        #gal_img.append(gal_img_tmp)
        #print(weight_img[0,0])
        gal_obs = Observation(gal_img_tmp, weight=weight_img, jacobian=gal_jacob,
                              psf=psf_obs, noise=noise_img,
                              bmask=flags[n_e].astype(np.int32),
                              ormask=flags[n_e].astype(np.int32))

        gal_obs_list.append(gal_obs)
        T_guess_psf.append(psf_T)

    if wsum == 0:
        raise ZeroDivisionError('Sum of weights = 0, division by zero')

    # Normalize PSF fit output
    for key in psf_res_gT.keys():
        psf_res_gT[key] /= wsum

    Tguess = np.mean(T_guess_psf)

    #print("T guess : {}".format(Tguess))
    #print("Ngmix T guess : {}".format(psf_res_gT['T_PSFo']))

    psf_model = 'gauss'
    gal_model = 'gauss'

    psf_reconv = 'gauss'
    #psf_reconv = galsim.Gaussian(sigma=np.sqrt(Tguess/2)*(1+0.02)).withFlux(1)

    # metacal specific parameters
    #metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'],
    #                'step': 0.01,
    #                'psf': psf_reconv,
    #                'fixnoise': True,
    #                'use_noise_image': True,
    #                'rng': rng}

    # obs_dict_mcal = ngmix.metacal.get_all_metacal(gal_obs_list, **metacal_pars)
    # res = {'mcal_flags': 0}
    # obs_dict_mcal['noshear'] = gal_obs_list

    #gal_pars = np.array([0., 0., 0., 0., (tile_r50/1.17741)**2 * 2, tile_flux])
    #gal_pars = np.array([0., 0., 0., 0., 1., tile_flux])

    ntry = 5

    prior = get_prior(rng=rng, galsim=True)

    #fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior)
    fitter = ngmix.fitting.GalsimFitter(model=gal_model, prior=prior)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng,
                                                     T=Tguess,
                                                     prior=prior)
    #guesser = ngmix.guessers.R50FluxGuesser(
    #    rng=rng,
    #    r50=tile_r50,
    #    flux=tile_flux,
    #    prior=prior,
    #)
    #fitter = ngmix.fitting.GalsimFitter(model=gal_model, prior=prior)
    
    psf_ngauss = 1
    psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss)
    psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng,
                                                   ngauss=psf_ngauss)
    #psf_fitter = ngmix.em.EMFitter()
    #psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng,
    #                                            ngauss=psf_ngauss)

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, 
                                         guesser=psf_guesser,
                                         ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter,
                                  guesser=guesser,
                                  ntry=ntry)

    boot_no_mcal = ngmix.bootstrap.Bootstrapper(runner=runner,
                                                psf_runner=psf_runner)
    boot = ngmix.metacal.MetacalBootstrapper(runner=runner, 
                                             psf_runner=psf_runner,
                                             psf=psf_reconv,
                                             rng=rng,
                                             step=0.01,
                                             types=['noshear', '1p', '1m', '2p', '2m'],
                                             use_noise_image=True)

    res, obs_dict_mcal = boot.go(gal_obs_list)
    res_no_mcal = boot_no_mcal.go(gal_obs_list)
    res['no_mcal'] = res_no_mcal

    #for key in res.keys():
        #print(obs_dict_mcal[key][0].psf.meta['result'])
    #    res[key]['Tpsf'] = obs_dict_mcal[key][0].psf.meta['result']['T']


    for key in sorted(obs_dict_mcal):

    #     fres = make_galsimfit(obs_dict_mcal[key],
    #                           gal_model, gal_pars,
    #                           rng,
    #                           prior=prior)

        fres = res[key]

        #res['mcal_flags'] |= fres['flags']
        tres = {}

        for name in fres.keys():
            tres[name] = fres[name]
        tres['flags'] = fres['flags']

        wsum = 0.0
        Tpsf_sum = 0.0
        gpsf_sum = np.zeros(2)
        npsf = 0
        for n_e, obs in enumerate(obs_dict_mcal[key]):

            try:
                psf_res = obs.psf.meta['result']
            except:
                continue
            g1, g2 = psf_res['g']
            T = psf_res['T']

            # TODO we sometimes use other weights
            twsum = obs.weight.sum()

            wsum += twsum
            gpsf_sum[0] += g1*twsum
            gpsf_sum[1] += g2*twsum
            Tpsf_sum += T*twsum
            npsf += 1

        tres['gpsf'] = gpsf_sum/wsum
        tres['Tpsf'] = Tpsf_sum/wsum

        res[key] = tres

    #res['true_g'] = true_g

    
    print()
    print('Final res')
    try:
        print(res['noshear']['pars'], res['noshear']['s2n'])
    except:
        print(res['noshear']['pars'], res['noshear']['s2n_r'])
    print('Final res no mcal')
    try:
        print(res_no_mcal['pars'], res_no_mcal['s2n'])
    except:
        print(res_no_mcal['pars'], res_no_mcal['s2n_r'])
    print('Final res err')
    print(np.sqrt(res['noshear']['pars_err']))
    #print('T psf: {}'.format(res['noshear']['Tpsf']))
    print()
    #res_mcal = make_galsimfit(obs_dict_mcal['noshear'][0], gal_model, gal_pars, rng, prior=prior)
    #print('Galsim fit mcal')
    #print('g1 : {}\ng2 : {}'.format(res_mcal['pars'][2], res_mcal['pars'][3]))
    #res_no_mcal = make_galsimfit(gal_obs_list[0], gal_model, gal_pars, rng, prior=prior)
    #print('Galsim fit no mcal')
    #print('g1 : {}\ng2 : {}'.format(res_no_mcal['pars'][2], res_no_mcal['pars'][3]))
    #print('HSM mcal')
    #try:
    #    s = galsim.hsm.FindAdaptiveMom(galsim.Image(obs_dict_mcal['noshear'][0].image, wcs=obs_dict_mcal['noshear'][0].jacobian.get_galsim_wcs()))
    #    print('g1 : {}\ng2 : {}'.format(s.observed_shape.g1, s.observed_shape.g2))
    #except:
    #    print('Fails')
    print('HSM no mcal')
    try:
        s = galsim.hsm.FindAdaptiveMom(galsim.Image(gal_obs_list[0].image, wcs=gal_obs_list[0].jacobian.get_galsim_wcs()))
        print('g1 : {}\ng2 : {}'.format(s.observed_shape.g1, s.observed_shape.g2))
    except:
        print('Fails')
    print("True flux : {}".format(tile_flux))
    print("Meas flux : {}".format(res['noshear']['flux']))
    print("True mag : {}".format(-2.5 * np.log10(tile_flux) + 30))
    print("Meas mag : {}".format(-2.5 * np.log10(res['noshear']['flux']) + 30))
    print("True r50 : {}".format(tile_r50))
    #print("Meas r50 : {}".format(np.sqrt(np.abs(res['noshear']['pars'][4])/2)*1.17741))
    print("Meas r50 : {}".format(res['noshear']['pars'][4]))
    
    ####
    fig_dir = '/automnt/n17data/guinot/simu_W3/plot_ngmix/'
    
    ####
    plt.figure()
    plt.imshow(obs_dict_mcal['noshear'][n_e].psf.image, cmap='gist_stern')
    plt.colorbar()
    plt.savefig(fig_dir + '/psf_metacal_{}_{}.png'.format(id_tmp, n_e))
    s = galsim.hsm.FindAdaptiveMom(galsim.Image(obs_dict_mcal['noshear'][n_e].psf.image, scale=0.187))
    #print(len(obs_dict_mcal[key]))
    #print(T_guess_psf[n_e]/1.17741*2.355)
    #print(s.observed_shape.g1, s.observed_shape.g2, s.moments_sigma*2.355*0.187)
    #print(g1, g2, T/1.17741*2.355)
    ####
    #plt.figure()
    #plt.imshow(gal_obs_list[n_e].psf.image, cmap='gist_stern')
    #plt.colorbar()
    #plt.savefig(fig_dir + '/psf_ori_{}_{}.png'.format(id_tmp, n_e))
    ####
    #psf_profile = np.sum(obs_dict_mcal['noshear'][n_e].psf.image, 0)
    #plt.figure()
    #plt.semilogy(psf_profile)
    #plt.savefig(fig_dir + '/psf_profile_{}_{}.png'.format(id_tmp, n_e))
    ####
    #plt.figure()
    #plt.imshow(flags[n_e])
    #plt.colorbar()
    #plt.savefig(fig_dir + '/flags_ori_{}_{}.png'.format(id_tmp, n_e))
    ####
    #plt.figure()
    #plt.imshow(w)
    #plt.colorbar()
    #plt.savefig(fig_dir + '/weight_ori_{}_{}.png'.format(id_tmp, n_e))
    
    ####
    plt.figure()
    plt.imshow(obs_dict_mcal['noshear'][n_e].image)
    plt.colorbar()
    plt.suptitle('Metacal noshear gal {} epoch {}'.format(id_tmp, n_e))
    plt.savefig(fig_dir + '/gal_metacal_{}_{}.png'.format(id_tmp, n_e))
    ####
    plt.figure()
    plt.imshow(obs_dict_mcal['1p'][n_e].image)
    plt.colorbar()
    plt.savefig(fig_dir + '/gal_1p_{}_{}.png'.format(id_tmp, n_e))
    ####
    #plt.figure()
    #plt.imshow(gal_img[0]-gal_img[-1])
    #plt.colorbar()
    #plt.savefig(fig_dir + '/gal_ori_stack_{}_{}.png'.format(id_tmp, n_e))
    ####
    for n_e in range(n_epoch):
        fig, ax = plt.subplots()
        im = ax.imshow(gals[n_e])
        fig.colorbar(im)
        s = galsim.Shear(g1=res['noshear']['pars'][2], g2=res['noshear']['pars'][3])
        e = Ellipse(xy=((gals[0].shape[0]-1)/2. + offset_list[n_e][0] + res['noshear']['pars'][0], 
                        (gals[0].shape[1]-1)/2. + offset_list[n_e][1] + res['noshear']['pars'][1]),
                    width=6.,
                    height=6*s.q,
                    angle=s.beta.deg)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
        ax.plot((gals[0].shape[0]-1)/2.+offset_list[n_e][0], (gals[0].shape[1]-1)/2.+offset_list[n_e][1], 'k+')
        #ax.plot(25., 25., 'r+')
        fig.suptitle('Original gal {} epoch {}'.format(id_tmp, n_e))
        plt.savefig(fig_dir + '/gal_ori_{}_{}.png'.format(id_tmp, n_e))
    ####
    fig, ax = plt.subplots()
    ax.imshow(gal_masked, cmap='gist_stern')
    s = galsim.Shear(g1=res['noshear']['pars'][2], g2=res['noshear']['pars'][3])
    e = Ellipse(xy=(25.5+res['noshear']['pars'][0], 25.5+res['noshear']['pars'][1]),
                width=6.,
                height=6*s.q,
                angle=s.beta.deg)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)
    #plt.colorbar()
    plt.savefig(fig_dir + '/gal_mask_{}_{}.png'.format(id_tmp, n_e))
    


    # result dictionary, keyed by the types in metacal_pars above
    metacal_res = res
    metacal_res.update(psf_res_gT)

    return metacal_res


# def compile_results(results):
#     """ Compile results

#     Prepare the results of ngmix before saving.

#     Parameters
#     ----------
#     results : dict
#         Dictionary containing the results of ngmix metacal.

#     Returns
#     -------
#     output_dict : dict
#         Dictionary containing ready to be saved.

#     """

#     names = ['1m', '1p', '2m', '2p', 'noshear']
#     names2 = ['id', 'n_epoch_model', 'g1', 'g1_err', 'g2', 'g2_err', 'T',
#               'T_err', 'Tpsf', 's2n', 'flags', 'mcal_flags']
#     output_dict = {k: {kk: [] for kk in names2} for k in names}
#     for i in range(len(results)):
#         for name in names:
#             output_dict[name]['id'].append(results[i]['obj_id'])
#             output_dict[name]['n_epoch_model'].append(results[i]['n_epoch_model'])
#             output_dict[name]['g1'].append(results[i][name]['g'][0])
#             output_dict[name]['g1_err'].append(results[i][name]['pars_err'][2])
#             output_dict[name]['g2'].append(results[i][name]['g'][1])
#             output_dict[name]['g2_err'].append(results[i][name]['pars_err'][3])
#             output_dict[name]['T'].append(results[i][name]['T'])
#             output_dict[name]['T_err'].append(results[i][name]['T_err'])
#             output_dict[name]['Tpsf'].append(results[i][name]['Tpsf'])
#             output_dict[name]['s2n'].append(results[i][name]['s2n'])
#             output_dict[name]['flags'].append(results[i][name]['flags'])
#             output_dict[name]['mcal_flags'].append(results[i]['mcal_flags'])

#     return output_dict


def compile_results(results, ZP):
    """ Compile results

    Prepare the results of ngmix before saving.

    Parameters
    ----------
    results : dict
        Dictionary containing the results of ngmix metacal.
    ZP : float
        Magnitude zero point.

    Returns
    -------
    output_dict : dict
        Dictionary containing ready to be saved.

    """

    names = ['1m', '1p', '2m', '2p', 'noshear']
    names2 = ['id', 'n_epoch_model',
              'g1_psfo_ngmix', 'g2_psfo_ngmix', 'T_psfo_ngmix',
              'g1', 'g1_err', 'g2', 'g2_err',
              'T', 'T_err', 'Tpsf', 'g1_psf', 'g2_psf',
              'flux', 'flux_err', 's2n',
              'mag', 'mag_err',
              'flags']
    output_dict = {k: {kk: [] for kk in names2} for k in names}
    for i in range(len(results)):
        for name in names:

            mag = -2.5 * np.log10(results[i][name]['flux']) + ZP
            mag_err = np.abs(-2.5 * results[i][name]['flux_err'] / (results[i][name]['flux'] * np.log(10)))

            output_dict[name]['id'].append(results[i]['obj_id'])
            output_dict[name]['n_epoch_model'].append(results[i]['n_epoch_model'])
            #©
            #output_dict[name]['ntry_fit'].append(results[i][name]['ntry'])
            output_dict[name]['g1_psfo_ngmix'].append(results[i]['g_PSFo'][0])
            output_dict[name]['g2_psfo_ngmix'].append(results[i]['g_PSFo'][1])
            #output_dict[name]['g1_err_psfo_ngmix'].append(results[i]['g_err_PSFo'][0])
            #output_dict[name]['g2_err_psfo_ngmix'].append(results[i]['g_err_PSFo'][1])
            output_dict[name]['T_psfo_ngmix'].append(results[i]['T_PSFo'])
            #output_dict[name]['T_err_psfo_ngmix'].append(results[i]['T_err_PSFo'])
            output_dict[name]['g1'].append(results[i][name]['g'][0])
            output_dict[name]['g1_err'].append(results[i][name]['pars_err'][2])
            output_dict[name]['g2'].append(results[i][name]['g'][1])
            output_dict[name]['g2_err'].append(results[i][name]['pars_err'][3])
            output_dict[name]['T'].append(results[i][name]['T'])
            output_dict[name]['T_err'].append(results[i][name]['T_err'])
            output_dict[name]['Tpsf'].append(results[i][name]['Tpsf'])
            output_dict[name]['g1_psf'].append(results[i][name]['gpsf'][0])
            output_dict[name]['g2_psf'].append(results[i][name]['gpsf'][1])
            output_dict[name]['flux'].append(results[i][name]['flux'])
            output_dict[name]['flux_err'].append(results[i][name]['flux_err'])
            output_dict[name]['mag'].append(mag)
            output_dict[name]['mag_err'].append(mag_err)

            #output_dict[name]['true_g1'].append(results[i]['true_g'][0])
            #output_dict[name]['true_g2'].append(results[i]['true_g'][1])

            try:
                output_dict[name]['s2n'].append(results[i][name]['s2n'])
            except:
                output_dict[name]['s2n'].append(results[i][name]['s2n_r'])
            output_dict[name]['flags'].append(results[i][name]['flags'])
            #output_dict[name]['mcal_flags'].append(results[i]['mcal_flags'])

    return output_dict


def save_results(output_dict, output_name):
    """ Save results

    Save the results into a fits file.

    Parameters
    ----------
    output_dict : dict
        Dictionary containing the results.
    output_name : str
        Name of the output file.

    """

    f = io.FITSCatalog(output_name,
                       open_mode=io.BaseCatalog.OpenMode.ReadWrite)

    for key in output_dict.keys():
        f.save_as_fits(output_dict[key], ext_name=key.upper())


def process(tile_cat_path, gal_vignet_path, bkg_vignet_path,
            psf_vignet_path, weight_vignet_path, flag_vignet_path,
            f_wcs_path, w_log, id_obj_min=-1, id_obj_max=-1,
            rng=np.random.RandomState(1234)):
    """ Process

    Process function.

    Parameters
    ----------
    tile_cat_path: str
        Path to the tile SExtractor catalog.
    gal_vignet_path: str
        Path to the galaxy vignets catalog.
    bkg_vignet_path: str
        Path to the background vignets catalog.
    psf_vignet_path: str
        Path to the PSF vignets catalog.
    weight_vignet_path: str
        Path to the weight vignets catalog.
    flag_vignet_path: str
        Path to the flag vignets catalog.
    f_wcs_path: str
        Path to the log file containing the WCS for each CCDs.
    w_log: log file object
        log file
    id_obj_min: int, optional, default=-1
        minimum object ID to be processed if > 0
    id_obj_max: int, optional, default=-1
        maximum object ID to be processed if > 0

    Returns
    -------
    final_res: dict
        Dictionary containing the ngmix metacal results.

    """

    tile_cat = io.FITSCatalog(tile_cat_path, SEx_catalog=True)
    tile_cat.open()
    obj_id = np.copy(tile_cat.get_data()['NUMBER'])
    tile_vign = np.copy(tile_cat.get_data()['VIGNET'])
    # tile_flag = np.copy(tile_cat.get_data()['FLAGS'])
    # tile_imaflag = np.copy(tile_cat.get_data()['IMAFLAGS_ISO'])
    tile_ra = np.copy(tile_cat.get_data()['XWIN_WORLD'])
    tile_dec = np.copy(tile_cat.get_data()['YWIN_WORLD'])
    tile_flux = np.copy(tile_cat.get_data()['FLUX_AUTO'])
    tile_fwhm = np.copy(tile_cat.get_data()['FWHM_IMAGE'])
    tile_cat.close()
    # sm_cat = io.FITSCatalog(sm_cat_path, SEx_catalog=True)
    # sm_cat.open()
    # sm = np.copy(sm_cat.get_data()['SPREAD_MODEL'])
    # sm_err = np.copy(sm_cat.get_data()['SPREADERR_MODEL'])
    # sm_cat.close()
    f_wcs_file = SqliteDict(f_wcs_path)
    gal_vign_cat = SqliteDict(gal_vignet_path)
    bkg_vign_cat = SqliteDict(bkg_vignet_path)
    psf_vign_cat = SqliteDict(psf_vignet_path)
    weight_vign_cat = SqliteDict(weight_vignet_path)
    flag_vign_cat = SqliteDict(flag_vignet_path)

    final_res = []
    prior = get_prior(rng)

    count = 0
    id_first = -1
    id_last = -1

    tmp_g1 = []
    tmp_g2 = []
    R11 = []
    R22 = []
    tmp_g1_err = []
    tmp_g2_err = []
    tmp_g1_true = []
    tmp_g2_true = []
    tmp_g1_true_err = []
    tmp_g2_true_err = []
    psf_hsm_shapes = []
    #w = 1

    #for i_tile, id_tmp in tqdm(enumerate(obj_id), total=len(obj_id[:2000])):
    for i_tile, id_tmp in enumerate(obj_id):

        if id_obj_min > 0 and id_tmp < id_obj_min:
            continue
        if id_obj_max > 0 and id_tmp > id_obj_max:
            continue

        if id_first == -1:
            id_first = id_tmp
        id_last = id_tmp

        w_log.info(id_tmp)
        #####
        
        print(os.path.split(tile_cat_path)[1], '    ', id_tmp)
        print('############\n############')
        print(id_tmp)
        print("Tile flux : {}".format(tile_flux[i_tile]))
        """
        #vign_tmp_tile = np.copy(tile_vign[i_tile])
        #vign_tmp_tile[vign_tmp_tile == -1e30] = 0
        #plt.figure()
        #plt.imshow(vign_tmp_tile)
        #plt.colorbar()
        #plt.savefig(fig_dir + '/gal_tile_{}.png'.format(id_tmp))
        """

        ####
        count = count + 1

        gal_vign = []
        psf_vign = []
        sigma_psf = []
        weight_vign = []
        flag_vign = []
        jacob_list = []
        bkg_list = []
        offset_list = []
        if (psf_vign_cat[str(id_tmp)] == 'empty') or (gal_vign_cat[str(id_tmp)] == 'empty'):
            continue
        psf_expccd_name = list(psf_vign_cat[str(id_tmp)].keys())
        for expccd_name_tmp in psf_expccd_name:
            exp_name, ccd_n = re.split('-', expccd_name_tmp)

            gal_vign_tmp = gal_vign_cat[str(id_tmp)][expccd_name_tmp]['VIGNET']
            if len(np.where(gal_vign_tmp.ravel() == 0)[0]) != 0:
                continue

            bkg_vign_tmp = bkg_vign_cat[str(id_tmp)][expccd_name_tmp]['VIGNET']
            gal_vign_sub_bkg = gal_vign_tmp - bkg_vign_tmp

            tile_vign_tmp = MegaCamFlip(np.copy(tile_vign[i_tile]), int(ccd_n))

            flag_vign_tmp = flag_vign_cat[str(id_tmp)][expccd_name_tmp]['VIGNET']
            flag_vign_tmp[np.where(tile_vign_tmp == -1e30)] = 2**10
            v_flag_tmp = flag_vign_tmp.ravel()
            if len(np.where(v_flag_tmp != 0)[0])/(51*51) > 1/3.:
                continue

            weight_vign_tmp = weight_vign_cat[str(id_tmp)][expccd_name_tmp]['VIGNET']

            wcs_tmp = f_wcs_file[exp_name][int(ccd_n)]['WCS']
            jacob_tmp = get_jacob(wcs_tmp,
                                  tile_ra[i_tile],
                                  tile_dec[i_tile])

            # Get offset
            true_pos = wcs_tmp.all_world2pix(tile_ra[i_tile], tile_dec[i_tile], 1)
            rounded_pos = np.round(true_pos).astype(int)
            dx, dy = true_pos-rounded_pos
            offset_list.append([dx, dy])

            # Get Fscale
            header_tmp = fits.Header.fromstring(f_wcs_file[exp_name][int(ccd_n)]['header'])
            Fscale = header_tmp['FSCALE']
            #Fscale = 1

            gal_vign_scaled = gal_vign_sub_bkg*Fscale
            weight_vign_scaled = weight_vign_tmp #* 1/Fscale**2.
            bkg_list.append(np.mean(bkg_vign_tmp)*Fscale)

            gal_vign.append(gal_vign_scaled)
            psf_vign.append(psf_vign_cat[str(id_tmp)][expccd_name_tmp]['VIGNET'])
            sigma_psf.append(psf_vign_cat[str(id_tmp)][expccd_name_tmp]['SHAPES']['SIGMA_PSF_HSM'])
            weight_vign.append(weight_vign_scaled)
            flag_vign.append(flag_vign_tmp)
            jacob_list.append(jacob_tmp)

            psf_hsm_shapes.append(psf_vign_cat[str(id_tmp)][expccd_name_tmp]['SHAPES'])

        if len(gal_vign) == 0:
            continue
        #try:
        res = do_ngmix_metacal(gal_vign,
                                   psf_vign,
                                   sigma_psf,
                                   weight_vign,
                                   flag_vign,
                                   jacob_list,
                                   offset_list,
                                   bkg_list,
                                   prior,
                                   id_tmp,
                                   psf_hsm_shapes,
                                   tile_flux[i_tile],
                                   tile_fwhm[i_tile]*0.187/2.355*1.17741,
                                   rng)
        #except Exception as ee:
        #    w_log.info('ngmix failed for object ID={}.\nMessage: {}'.format(id_tmp, ee))
        #    continue
        
        if (res['noshear']['T']/res['noshear']['Tpsf']>0.5) & (res['noshear']['pars'][-1]/res['noshear']['pars_err'][-1]>10):
            tmp_g1.append(res['noshear']['pars'][2])
            tmp_g2.append(res['noshear']['pars'][3])
            R11.append((res['1p']['pars'][2] - res['1m']['pars'][2])/0.02)
            R22.append((res['2p']['pars'][3] - res['2m']['pars'][3])/0.02)
            tmp_g1_err.append(res['noshear']['pars_err'][2])
            tmp_g2_err.append(res['noshear']['pars_err'][3])
            tmp_g1_true.append(res['no_mcal']['pars'][2])
            tmp_g2_true.append(res['no_mcal']['pars'][3])
            tmp_g1_true_err.append(res['no_mcal']['pars_err'][2])
            tmp_g2_true_err.append(res['no_mcal']['pars_err'][3])
            w = 1/(2*0.34**2 + np.array(tmp_g1_err) + np.array(tmp_g2_err))
            w_true = 1/(2*0.34**2 + np.array(tmp_g1_true_err) + np.array(tmp_g2_true_err))
        print('\n\n############\nN objects : {}\n                WAM          AM          WA        Rxx        std\nCURRENT G1 : {:.5}   {:.5}   {:.5}   {:.5}    {:.3}\nCURRENT G2 : {:.5}   {:.5}   {:.5}   {:.5}    {:.3}\n############\n\n'.format(len(tmp_g1), np.average(tmp_g1, weights=w)/np.mean(R11), np.mean(tmp_g1)/np.mean(R11), np.average(tmp_g1_true, weights=w_true), np.mean(R11), np.std(tmp_g1), np.average(tmp_g2, weights=w)/np.mean(R22), np.mean(tmp_g2)/np.mean(R22), np.average(tmp_g2_true, weights=w_true), np.mean(R22), np.std(tmp_g2)))
        
        res['obj_id'] = id_tmp
        res['n_epoch_model'] = len(gal_vign)
        final_res.append(res)

    w_log.info('ngmix loop over objects finished, processed {} '
               'objects, id first/last={}/{}'.format(count,
                                                     id_first,
                                                     id_last))

    f_wcs_file.close()
    gal_vign_cat.close()
    bkg_vign_cat.close()
    flag_vign_cat.close()
    weight_vign_cat.close()
    psf_vign_cat.close()

    return final_res


@module_runner(input_module=['sextractor_runner', 'psfex_interp_runner_me',
                             'vignetmaker_runner2'],
               version='0.0.1',
               file_pattern=['tile_sexcat', 'image', 'exp_background',
                             'galaxy_psf', 'weight', 'flag'],
               file_ext=['.fits', '.sqlite', '.sqlite', '.sqlite', '.sqlite',
                         '.sqlite'],
               depends=['numpy', 'ngmix', 'galsim', 'sqlitedict', 'astropy'])
def ngmix_runner_old(input_file_list, run_dirs, file_number_string,
                 config, w_log):

    # Init randoms
    seed = int(''.join(re.findall(r'\d+', file_number_string)))
    rng = np.random.RandomState(seed)

    output_name = (run_dirs['output'] + '/' + 'ngmix' +
                   file_number_string + '.fits')

    ZP = config.getfloat('NGMIX_RUNNER_OLD', 'MAG_ZP')

    f_wcs_path = config.getexpanded('NGMIX_RUNNER_OLD', 'LOG_WCS')

    id_obj_min = config.getint('NGMIX_RUNNER_OLD', 'ID_OBJ_MIN')
    id_obj_max = config.getint('NGMIX_RUNNER_OLD', 'ID_OBJ_MAX')

    metacal_res = process(*input_file_list, f_wcs_path, w_log,
                          id_obj_min=id_obj_min, id_obj_max=id_obj_max,
                          rng=rng)
    res_dict = compile_results(metacal_res, ZP)
    save_results(res_dict, output_name)

    return None, None
