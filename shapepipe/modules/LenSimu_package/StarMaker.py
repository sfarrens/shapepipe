# -*- coding: utf-8 -*-

"""GALAXY MAKER

This file contains methods to simulate a galaxies.

:Author: Axel Guinot

"""


import numpy as np

import galsim


from sqlitedict import SqliteDict


class seeing_distribution(object):
    """ Seeing distribution

    Provide a seeing following CFIS distribution. Seeing generated from
    scipy.stats.rv_histogram(np.histogram(obs_seeing)). Object already
    initialized and saved into a numpy file.

    Parameters
    ----------
    path_to_file: str
        Path to the numpy file containing the scipy object.
    seed: int
        Seed for the random generation. If None rely on default one.

    """

    def __init__(self, path_to_file, seed=None):

        self._file_path = path_to_file
        self._load_distribution()

        self._random_seed = None
        if seed != None:
            self._random_seed = np.random.RandomState(seed)

    def _load_distribution(self):
        """ Load distribution

        Load the distribution from numpy file.

        """

        self._distrib = np.load(self._file_path, allow_pickle=True).item()

    def get(self, size=None):
        """ Get

        Return a seeing value from the distribution.

        Parameters
        ----------
        size: int
            Number of seeing value required.

        Returns
        -------
        seeing: float (numpy.ndarray)
            Return the seeing value or a numpy.ndarray if size != None.

        """

        return self._distrib.rvs(size=size, random_state=self._random_seed)


class StarMaker(object):
    """ Star Maker
    """

    def __init__(self, param_dict, seed=None):

        self.param_dict = param_dict

        self._init_random(seed)
        self._init_PSF_param(seed)

    def _init_random(self, seed):
        """
        """

        np.random.seed(seed)
        self._ud = galsim.UniformDeviate(seed)

    def _init_PSF_param(self, seed):
        """
        """

        # Files path
        e1_optic_path = self.param_dict['PSF']['PSF_FILE_DIR'] + '/e1_psf.npy'
        e2_optic_path = self.param_dict['PSF']['PSF_FILE_DIR'] + '/e2_psf.npy'
        seeing_distribution_path = self.param_dict['PSF']['PSF_FILE_DIR'] + '/seeing_distribution.npy'

        # Init seeing
        self.seeing_func = seeing_distribution(seeing_distribution_path, seed)
        self.atmo_fwhm = self.seeing_func.get(1)

        # Init optic ellipticities
        self._e1_optic_array = np.load(e1_optic_path)
        self._e2_optic_array = np.load(e2_optic_path)

        # Compute pix size on mean shapes plot
        self._mean_shape_plot = {}
        self._mean_shape_plot['size_img_x'] = (self.param_dict['data_sec'][1] - self.param_dict['data_sec'][0]) + 1
        self._mean_shape_plot['size_img_y'] = (self.param_dict['data_sec'][3] - self.param_dict['data_sec'][2]) + 1
        self._mean_shape_plot['n_pix_x'], self._mean_shape_plot['n_pix_y'] = self._e1_optic_array[0].shape
        self._mean_shape_plot['pix_size_x'] = self._mean_shape_plot['size_img_x'] / self._mean_shape_plot['n_pix_x']
        self._mean_shape_plot['pix_size_y'] = self._mean_shape_plot['size_img_y'] / self._mean_shape_plot['n_pix_y']


    def optical_PSF(self):
        """
        """

        optical = galsim.Airy(lam=self.param_dict['filter_lambda'], 
                              diam=self.param_dict['telescope_diameter'])

        return optical

    def atmospheric_PSF(self, atmo_fwhm):
        """
        """

        atmo = galsim.Kolmogorov(fwhm=atmo_fwhm)

        return atmo

    def make_star(self, flux,
                  ccd_number, x, y):
        """
        """

        # Create model
        optical = self.optical_PSF()
        atmo = self.atmospheric_PSF(self.atmo_fwhm)

        star = galsim.Convolve([optical, atmo])

        star = star.withFlux(flux)

        # Get ellipticity
        pos_x_msp = int((x-self.param_dict['TELESCOPE']['DATA_SEC'][0]) / self._mean_shape_plot['pix_size_x'])
        pos_y_msp = int((y-self.param_dict['TELESCOPE']['DATA_SEC'][2]) / self._mean_shape_plot['pix_size_y'])
        g1 = self._e1_optic_array[ccd_number, pos_x_msp, pos_y_msp]
        g2 = self._e2_optic_array[ccd_number, pos_x_msp, pos_y_msp]

        # Final star
        star = star.shear(g1=g1, g2=g2)

        return star, g1, g2, self.atmo_fwhm