# -*- coding: utf-8 -*-

"""TILEOBJ_AS_EXP_SCRIPT

This script contains a class to handle processing for the combine_mexp_script
module: Combine information on objects and PSF from multiple
exposure-single-CCD catalogues.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

:Date: February 2019

:Version: 1.0

"""


import os
import re

import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
from astropy.table import Table

import shapepipe.pipeline.file_io as io


class CombineMExpError(Exception):
   """ CombineMExpError

   Generic error that is raised by this script.

   """

   pass


class combine_mexp():

    """Class for combining multiple exposure-single-CCD catalogue and PSF information.

    Parameters
    ----------
    input_file_name: list of 2 string
        path to input files, [image, catalogue]
    config: config class
        config file content
    output_dir: string
        output directory
    image_number: string
        image number for input tile according to numbering scheme from config file
    w_log: log file class
        log file

    Returns
    -------
    None
    """

    def __init__(self, input_file_names, output_dir, image_number, config, w_log):

        self._img_tile_path = input_file_names[0]
        self._output_dir = output_dir
        self._image_number = image_number
        self._config = config
        self._w_log = w_log

        self._vignet = True
        self._do_check_consistency = True
        self._method = 'PSFsize'

    def process(self):

        # List of exposures contributing to this tile
        exp_list_uniq = self.get_exposure_list()

        # List of exposure-single-CCD catalogue and PSF file names that were created by the
        # tileobj_as_exp modules
        exp_CCD_cat_list, psf_list = self.get_exp_CCD_cat_and_PSF_list(len(exp_list_uniq))

        n_ok = self.combine_information(exp_CCD_cat_list, psf_list)
        self._w_log.info('Processed {} pairs of exposure-single-CCD and galaxy PSF files'.format(n_ok))

        return None, None

    def combine_information(self, exp_CCD_cat_list, psf_list):
        """Combine galaxy and PSF information.

        Parameters
        ----------
        exp_CCD_list: list of strings
            exposure-single-CCD catalogue base names
        psf_list: list of strings
            PSF vignet file base names

        Returns
        -------
        None
        """

        outcat_pattern = 'mobj' 

        input_dir_exp_cat = self._config.get('COMBINE_MEXP_RUNNER',
                                             'INPUT_DIR_EXP_CAT')
        exp_cat_ext = self._config.get('COMBINE_MEXP_RUNNER',
                                       'INPUT_EXP_CAT_EXT')

        input_dir_psf = self._config.get('COMBINE_MEXP_RUNNER',
                                         'INPUT_DIR_PSF')
        psf_ext = self._config.get('COMBINE_MEXP_RUNNER', 'INPUT_PSF_EXT')

        tile_data = None

        n_ok = 0
        for exp_base_name, psf_base_name in zip(exp_CCD_cat_list, psf_list):

            exp_cat = self.open_or_continue(input_dir_exp_cat, exp_base_name,
                                            exp_cat_ext)
            psf = self.open_or_continue(input_dir_psf, psf_base_name, psf_ext)

            if not exp_cat or not psf:
                continue

            exp_cat_data = exp_cat.get_data()
            exp_cat_cols = exp_cat.get_col_names()
            n_data = len(exp_cat_data)

            psf_data = psf.get_data()
            psf_cols = psf.get_col_names()

            if n_data != len(psf_data):
                raise CombineMExpError('Length of objects ({}) and PSF ({}) '
                                       'have to be the same'.format(n_data,
                                        len(psf_data)))

            combined_data = self.combine_exp_psf(exp_cat_data, exp_cat_cols, psf_data, psf_cols)

            self.select_galaxies(combined_data)

            exp_cat.close()
            psf.close()

            # TODO (here?) selecting objects according to size

            # Saving to file maybe only for testing. Note that tile_num here is string
            out_path = '{}/{}{}-0.fits'.format(self._output_dir, outcat_pattern, self._image_number)
            print('Writing tile data to file \'{}\''.format(out_path))
            output = io.FITSCatalog(out_path, open_mode=sc.BaseCatalog.OpenMode.ReadWrite)
            output.save_as_fits(tile_data)

            n_ok = n_ok + 1

        return n_ok

    def select_galaxies(self, combined_data):

        # Create empty data array for selected galaxies
        dat_gal = {}
        for c in col_names:
            dat_gal[c] = []

        # Add new column to store number of exposures for which object
        # passes galaxy selection
        dat_gal['nexp'] = []

        # Find all objects with same ID and their frequency
        IDs = dat['ID']
        IDs_unique, nexp = np.unique(IDs, return_counts=True)

        n_is_gal = 0
        for u, n in zip(IDs_unique, nexp):
            gal = dat[IDs == u]

            if self.is_consistent(gal, n):
                raise CombineMExpError('Length of galaxy data {} != number of '
                                       'exposures {}'.format(len(gal), nexp))

            is_gal, params_out = self.select_one(gal, n, method, params_in)
            if is_gal:
                n_is_gal += 1
                for c in col_names:
                    dat_gal[c].append(gal[0][c])
                dat_gal['nexp'].append(n)

        print(n_is_gal)
        if verbose:
            print('{}/{} galaxies  selected'.format(len(dat_gal['nexp']), len(dat)))


    def is_consistency(self, gal, nexp):
        """Check consistency of multi-exposure galaxy data

        Parameters
        ----------
        gal: FITS_rec
            galaxy and PSF data from multiple exposures
        nexp: int
            number of exposures

        Returns
        -------
        result: bool
            True if consistent
        """

        if not self._do_check_consistency:
            return True

        if len(gal) != nexp:
            return False
        else:
            return True


    def select_one(self, gal, nexp, method, params):
        """Select galaxy according to multi-exposure information.

        Parameters
        ----------
        gal: list of FITS_rec
            galaxy and PSF data from multiple exposures
        nexp: int
            number of exposures
        method: string
            selection method name
        params: dict
            method parameters

        Returns
        -------
        is_gal, params_out: bool
            True if object is selected as galaxy
        """

        is_gal    = False
        params_out = {}

        if self._method == 'PSF_size':

            fwhm = gal['FWHM_IMAGE'][0]
            n_larger = 0
            for g in gal:
                if g['HSM_FLAG'] == 0 and fwhm > 2.355 * g['SIGMA_PSF_HSM']:
                    n_larger += 1
            params_out['n_larger'] = n_larger
            if n_larger == nexp:
                is_gal = True

        else:

            raise CombineMExpError('Invalid selection method \'{}\''.format(method))


    def combine_exp_psf(self, exp_cat_data, exp_cat_cols, psf_data, psf_cols):
        """Return data array with combined information from exposure-
           single-CCD and PSF.

        Parameters
        ----------
        exp_cat_data: FITS_rec
            exposure-single-CCD data
       exp_cols: array of string
            exposure-single-CCD column names
       psf_data: FITS_rec
           PSF data
       psf_cols: array of string
           PSF column names

        Returns
        -------
        combined_data: array
            combined data for this tile
        """

        exp_cat_data_plus = {}

        # Copy object data
        for c in exp_cat.get_col_names():
            exp_cat_data_plus[c] = exp_cat_data[c]

        # Add PSF information to object catalogue
        for c in psf.get_col_names():
            if self._vignet or c != 'VIGNET':
                # Add vignet only if argument 'vignet' is True
                exp_cat_data_plus[c] = psf_data[c]

        if combined_data is None:
            combined_data = exp_cat_data_plus
        else:
            for c in exp_cat_data_plus:
                combined_data[c] = np.concatenate((exp_cat_data_plus[c], combined_data[c]))

        return combined_data

    def open_or_continue(self, input_dir, base_name, ext):
        """Return file content, or continue if file does not exist.

        Parameters
        ----------
        input_dir: string
            input directory
        base_name: string
            file base name
        ext: string
            file extension

        Returns
        -------
        cat: io.FITSCatalogue
            file content
        """

        hdu_no = 2
        cat_name = '{}/{}{}'.format(input_dir, base_name, ext)

        cat = io.FITSCatalog(cat_name, hdu_no=hdu_no)
        try:
            cat.open()
        except:
            msg = 'File \'{}\' not found, continuing...'.format(cat_name)
            self._w_log.info(msg)
            cat = None

        return cat

    def get_exposure_list(self):
        """Return list of exposure file used for the tile in process, from tiles FITS header
           MKDEBUG: Note: this function is identical to the find_exposures class routine

        Parameters
        ----------

        Returns
        -------
        exp_list_uniq: list of string
            list of exposure basenames
        """

        try:
            hdu = fits.open(self._img_tile_path)
            hist = hdu[0].header['HISTORY']

        except:
            self._w_log.info('Error while reading tile catalogue FITS file {}, continuing...'.format(self._img_tile_path))

        tile_num = self._image_number

        exp_list = []

        # Get exposure file names
        for h in hist:
            temp = h.split(' ')

            pattern = r'(.*)\.{1}.*'
            m = re.search(pattern, temp[3])
            if not m:
                raise CombineMExpError('re match \'{}\' failed for filename \'{}\''.format(pattern, temp[3]))

            exp_name = m.group(1)

            exp_list.append(exp_name)

        exp_list_uniq = list(set(exp_list))

        self._w_log.info('Found {} exposures used in tile #{}'.format(len(exp_list_uniq), tile_num))
        self._w_log.info('{} duplicates were removed'.format(len(exp_list) - len(exp_list_uniq)))

        return exp_list_uniq

    def get_exp_CCD_cat_and_PSF_list(self, n_exp):
        """Return lists of exposure-single-CCD catalogue file base names and
           PSF vignet file base names.
           MKDEBUG note: This function is identical to the tileob_as_exp
           class routine.

        Parameters
        ----------
        n_exp: int
            number of exposures

        Returns
        -------
        exp_CCD_list: list of strings
            exposure-single-CCD catalogue base names
        psf_list: list of strings
            PSF vignet file base names
        """

        n_hdu = int(self._config.get('COMBINE_MEXP_RUNNER', 'N_HDU'))
        exp_basename = 'cat.exp.img'
        psf_basename = 'galaxy_psf'

        exp_CCD_list = []
        psf_list     = []
        for i in range(n_exp):
            exp_CCD_base = '{}{}_{}'.format(exp_basename, self._image_number, i)
            psf_base = '{}{}_{}'.format(psf_basename, self._image_number, i)

            for k_img in range(n_hdu):
                exp_CCD_name = '{}_{:02d}'.format(exp_CCD_base, k_img)
                exp_CCD_list.append(exp_CCD_name)

                psf_name = '{}_{:02d}'.format(psf_base, k_img)
                psf_list.append(psf_name)

        return exp_CCD_list, psf_list
