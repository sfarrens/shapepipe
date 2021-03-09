# -*- coding: utf-8 -*-

"""GALAXY MAKER

This file contains methods to simulate a galaxies.

:Author: Axel Guinot

"""


import numpy as np

import galsim


def sersic_second_moments(n,hlr,q,beta):
    """Calculate the second-moment tensor of a sheared Sersic radial profile.
    
    Code from: https://github.com/esheldon/WeakLensingDeblending/blob/master/descwl/model.py

    Args:
        n(int): Sersic index of radial profile. Only n = 1 and n = 4 are supported.
        hlr(float): Radius of 50% isophote before shearing, in arcseconds.
        q(float): Ratio b/a of Sersic isophotes after shearing.
        beta(float): Position angle of sheared isophotes in degrees, measured anti-clockwise
            from the positive x-axis.
    Returns:
        numpy.ndarray: Array of shape (2,2) with values of the second-moments tensor
            matrix, in units of square arcseconds.
    Raises:
        RuntimeError: Invalid Sersic index n.
    """
    # Lookup the value of cn = 0.5*(r0/hlr)**2 Gamma(4*n)/Gamma(2*n)
    if n == 1:
        cn = 1.06502
    elif n == 4:
        cn = 10.8396
    else:
        raise RuntimeError('Invalid Sersic index n.')
    e_mag = (1.-q)/(1.+q)
    e_mag_sq = e_mag**2
    e1 = e_mag*np.cos(2*beta*np.pi/180)
    e2 = e_mag*np.sin(2*beta*np.pi/180)
    Q11 = 1 + e_mag_sq + 2*e1
    Q22 = 1 + e_mag_sq - 2*e1
    Q12 = 2*e2
    return np.array(((Q11,Q12),(Q12,Q22)))*cn*hlr**2/(1-e_mag_sq)**2


def sheared_second_moments(Q,g1,g2):
    """Apply shear to a second moments matrix.

    Code from: https://github.com/esheldon/WeakLensingDeblending/blob/master/descwl/model.py

    The shear M = ((1-g1,-g2),(-g2,1+g1)) is applied to each Q by calculating
    Q' = (M**-1).Q.(M**-1)^t where M**-1 = ((1+g1,g2),(g2,1-g1))/\|M\|.
    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).
    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric.
        g1(float): Shear ellipticity component g1 (+) with \|g\| = (a-b)/(a+b).
        g2(float): Shear ellipticity component g2 (x) with \|g\| = (a-b)/(a+b).
    Returns:
        numpy.ndarray: Array with the same shape as the input Q with the shear
            (g1,g2) applied to each 2x2 second-moments submatrix.
    """
    detM = 1 - g1**2 - g2**2
    Minv = np.array(((1+g1,g2),(g2,1-g1)))/detM
    return np.einsum('ia,...ab,jb',Minv,Q,Minv)


def moments_size_and_shape(Q):
    """Calculate size and shape parameters from a second-moment tensor.

    Code from: https://github.com/esheldon/WeakLensingDeblending/blob/master/descwl/model.py

    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).
    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric (only the [0,1] component is used).
    Returns:
        tuple: Tuple (sigma_m,sigma_p,a,b,beta,e1,e2) of :class:`numpy.ndarray` objects
            with shape (...). Refer to :ref:`analysis-results` for details on how
            each of these vectors is defined.
    """
    trQ = np.trace(Q,axis1=-2,axis2=-1)
    detQ = np.linalg.det(Q)
    asymQx = Q[...,0,0] - Q[...,1,1]
    asymQy = 2*Q[...,0,1]
    e_denom = trQ + 2*np.sqrt(detQ)
    e1 = asymQx/e_denom
    e2 = asymQy/e_denom
    return e1,e2


class GalaxyMaker(object):
    """ Galaxy Maker

    This class contruct a bulge+disk galaxy model from input parameters.
    The main method to call here is "make_gal" which hadle the process.

    Parameters
    ----------
    param_dict: dict
        Dictionary with config values.
        Not used ATM.

    """

    def __init__(self, param_dict):

        self.param_dict = param_dict

    def make_bulge(self, flux, hlr, q, beta):
        """ Make bulge

        Construct the bulge of the galaxy model.
        The bulge use a De Vaucouleur light profile (sersic n=4).

        Paramters
        ---------
        flux: float
            Flux of the bulge.
        hlr: float
            Half-light-radius of the major axis of the bulge.
            (hard coded with MICE simu format)
        q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a.
        beta: float
            Angle orientation between the major axis and the horizon. Defined 
            anti-clock wise. In degrees.

        Returns
        -------
        bulge: galsim.GSObject
            Galsim object with the bulge model.
        g1, g2: float, float
            Intrinsic ellipticity of the bulge. Defined as galsim reduced shear.

        """

        a = hlr
        b = hlr*q

        bulge = galsim.DeVaucouleurs(half_light_radius=np.sqrt(a*b), flux=flux).shear(q=q, beta=beta*galsim.degrees)

        # s = galsim.Shear(q=q, beta=beta*galsim.degrees)
        bulge_moments = sersic_second_moments(4, np.sqrt(a*b), b/a, beta)

        return bulge, bulge_moments

    def make_disk(self, flux, hlr, q, beta):
        """ Make disk

        Construct the disk of the galaxy model.
        The disk use an Exponential light profile (sersic n=1).

        Paramters
        ---------
        flux: float
            Flux of the disk.
        hlr: float
            Half-light-radius of the major axis of the disk.
            (hard coded with MICE simu format)
        q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a.
        beta: float
            Angle orientation between the major axis and the horizon. Defined 
            anti-clock wise. In degrees.

        Returns
        -------
        disk: galsim.GSObject
            Galsim object with the disk model.
        g1, g2: float, float
            Intrinsic ellipticity of the disk. Defined as galsim reduced shear.
            
        """

        a = hlr
        b = hlr*q

        disk = galsim.Exponential(half_light_radius=np.sqrt(a*b), flux=flux).shear(q=q, beta=beta*galsim.degrees)

        disk_moments = sersic_second_moments(1, np.sqrt(a*b), b/a, beta)

        return disk, disk_moments

    def make_gal(self, flux, bulge_total_ratio,
                 disk_hlr, disk_q, disk_beta,
                 bulge_hlr, bulge_q, bulge_beta,
                 shear_gamma1, shear_gamma2, shear_kappa):
        """ Make galaxy

        This method handle the creation of the galaxy profile form a bulge and a
        disk. The created profile is not convolved by the PSF. The profile have
        an intrinsic ellipticity and a shear.

        Parameters
        ----------
        flux: float
            Total flux of the profile
        bulge_total_ratio: float
            Ration between the flux of the bulge and the total flux. [0, 1]
        disk_hlr: float
            Half-light-radius of the major axis of the disk.
            (hard coded with MICE simu format)
        disk_q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a
            for the disk.
        disk_beta: float
            Angle orientation between the major axis and the horizon for the
            disk. Defined 
            anti-clock wise. In degrees.
        bulge_hlr: float
            Half-light-radius of the major axis of the bulge.
            (hard coded with MICE simu format)
        bulge_q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a 
            for the bulge.
        bulge_beta: float
            Angle orientation between the major axis and the horizon for the
            bulge. Defined 
            anti-clock wise. In degrees.
        shear_gamma1, shear_gamma2: float, float
            Shear gamma values.
        kappa: float
            Shear kappa.

        Returns
        -------
        gal: galsim.GSObject
            Galsim object with the galaxy total model.
        intrinsic_g1, intrinsic_g2: float, float
            Intrinsic ellipticity of the full model. Defined as galsim reduced 
            shear.
            The total ellipticity is defined as: 
            g_bulge * flux_bulge/total_flux + g_disk * flux_disk/total_flux
        g1, g2: float, float
            Reduced shear.
        mu: float
            Shear magnification.

        """

        g1 = - shear_gamma1 / (1 - shear_kappa)
        g2 = shear_gamma2 / (1 - shear_kappa)
        mu = 1 / ((1 - shear_kappa)**2. - (shear_gamma1**2 + shear_gamma2**2))

        bulge_flux = flux * bulge_total_ratio
        disk_flux = flux - bulge_flux

        bulge, bulge_moments = self.make_bulge(bulge_flux, bulge_hlr,
                                               bulge_q, bulge_beta)
        if disk_hlr != 0:
            disk, disk_moments = self.make_disk(disk_flux, disk_hlr, 
                                                disk_q, disk_beta)
            gal = bulge + disk
        else:
            disk_moments = 0
            gal = bulge

        total_moments = bulge_total_ratio*bulge_moments + \
                        (1-bulge_total_ratio)*disk_moments

        gal = gal.lens(g1=g1, g2=g2, mu=mu)

        # total_sheared_moments = sheared_second_moments(total_moments, g1, g2)

        intrinsic_e1, intrinsic_e2 = moments_size_and_shape(total_moments)
        s = galsim.Shear(e1=intrinsic_e1, e2=intrinsic_e2)
        intrinsic_g1 = s.g1
        intrinsic_g2 = s.g2

        total_moments_sheared = sheared_second_moments(total_moments, g1, g2)

        return gal, intrinsic_g1, intrinsic_g2, total_moments_sheared, g1, g2, mu