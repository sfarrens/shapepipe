
from tqdm import tqdm
import os
import re

import numpy as np
import numpy.lib.recfunctions as rfn

import galsim

from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u

import sip_tpv as stp

from shapepipe.pipeline import file_io as io
from astropy.io import fits

from shapepipe.modules.LenSimu_package.CCDMaker import CCDMaker


class ExposureMaker(object):
    """
    """

    def __init__(self, header_list, gal_catalog, star_catalog,
                 output_dir='/Users/aguinot/Documents/pipeline/simu_MICE/output_full_tile',
                 psf_file_dir='/Users/aguinot/Documents/pipeline/simu_MICE/input_param'):

        param_dict={'TELESCOPE': {'N_CCD': 40,
                                    'BACKGROUND_VALUE_KEY': 'IMMODE',
                                    'PIXEL_SCALE_KEY': 'PIXSCAL1',
                                    'CCD_SIZE_X_KEY': 'NAXIS1',
                                    'CCD_SIZE_Y_KEY': 'NAXIS2',
                                    'MAG_ZP_KEY': 'PHOTZP',
                                    'DATA_SEC': [33, 2080, 1, 4612],
                                    'SATURATE_KEY': 'SATURATE',
                                    'FILE_NAME_KEY': 'FILENAME',
                                    'FOV_DIAMATER': 1.5},
                    'PSF': {'SAVE_PSF': False},
                    'GALAXY': {'CONVERT_MAG_SDSS': True},
                    'filter_lambda': 650,
                    'telescope_diameter': 3.5,
                    'FILE' : {'COMPRESS_IMAGE': True,
                              'MAKE_WEIGHT': True,
                              'MAKE_FLAG': True}}
        param_dict['FILE']['OUTPUT_DIR'] = output_dir
        param_dict['PSF']['PSF_FILE_DIR'] = psf_file_dir

        self.header_list = header_list
        self.param_dict = param_dict

        # self.gal_catalog = gal_catalog
        # self.star_catalog = star_catalog

        self._init_catalog(gal_catalog, star_catalog)

        self._init_output()

    def _init_catalog(self, gal_catalog, star_catalog):
        """
        """

        gal_catalog_ap = coord.SkyCoord(ra=gal_catalog['ra']*u.degree,
                                             dec=gal_catalog['dec']*u.degree)
        star_catalog_ap = coord.SkyCoord(ra=star_catalog['ra']*u.degree,
                                             dec=star_catalog['dec']*u.degree)

        field_center = coord.SkyCoord(ra=[self.header_list[0]['CRVAL1']*u.degree],
                                      dec=[self.header_list[0]['CRVAL2']*u.degree])
        
        # Pre-select objects
        m_gal = gal_catalog_ap.search_around_sky(field_center, 
                    seplimit=self.param_dict['TELESCOPE']['FOV_DIAMATER']/2.*u.degree)[1]
        m_star = star_catalog_ap.search_around_sky(field_center, 
                    seplimit=self.param_dict['TELESCOPE']['FOV_DIAMATER']/2.*u.degree)[1]

        # self.gal_catalog = np.array([gal_catalog[key][m_gal] for key in gal_catalog.keys()])
        k = 0
        for key in gal_catalog.keys():
            if k == 0:
                self.gal_catalog = gal_catalog[key][m_gal].astype([(key, gal_catalog[key].dtype.type)])
            else:
                tmp_array = gal_catalog[key][m_gal].astype([(key, gal_catalog[key].dtype.type)])
                self.gal_catalog = rfn.merge_arrays((self.gal_catalog, tmp_array), flatten=True)
            k += 1
        self.star_catalog = star_catalog[m_star]
        self.gal_catalog_ap = gal_catalog_ap[m_gal]
        self.star_catalog_ap = star_catalog_ap[m_star]

    def _init_output(self):
        """
        """

        self.output_image_path = self.param_dict['FILE']['OUTPUT_DIR'] + '/images'
        self.output_catalog_path = self.param_dict['FILE']['OUTPUT_DIR'] + '/catalogs'

        if not os.path.exists(self.output_image_path):
            os.mkdir(self.output_image_path)
        if not os.path.exists(self.output_catalog_path):
            os.mkdir(self.output_catalog_path)


    def get_ccd_catalog(self, ccd_number):
        """
        """

        header = self.header_list[ccd_number].copy()
        stp.pv_to_sip(header)

        w = WCS(header)

        mask_gal = w.footprint_contains(self.gal_catalog_ap)
        mask_star = w.footprint_contains(self.star_catalog_ap)

        return self.gal_catalog[mask_gal], self.star_catalog[mask_star]

    def running_func(self, ccd_number, seed, seed_psf):
        """
        """

        gal_catalog, star_catalog = self.get_ccd_catalog(ccd_number)

        header = self.header_list[ccd_number].copy()
        stp.pv_to_sip(header)

        ccd_obj = CCDMaker(ccd_number,
                           header,
                           self.param_dict,
                           seed,
                           seed_psf)
        
        ccd_img, ccd_catalog, ccd_psf_catalog = ccd_obj.go(gal_catalog, star_catalog)

        return [ccd_img, header], ccd_catalog, ccd_psf_catalog

    def runner(self, seed_ori=1234):
        """
        """

        n_ccd = self.param_dict['TELESCOPE']['N_CCD']

        single_exposure = []
        single_exposure_cat = []
        single_exposure_psf_cat = []

        # seed_ori = 1234

        for ccd_number in range(0, n_ccd):

            seed = seed_ori + 10000*ccd_number

            ccd_tmp, cat_tmp, psf_cat_tmp = self.running_func(ccd_number, seed, seed_ori)
            single_exposure.append(ccd_tmp)
            single_exposure_cat.append(cat_tmp)
            single_exposure_psf_cat.append(psf_cat_tmp)

        return single_exposure, single_exposure_cat, single_exposure_psf_cat

    def write_output(self, ccd_images, ccd_cats, psf_cats):
        """
        """

        ori_name = re.findall('\d+', self.header_list[0][self.param_dict['TELESCOPE']['FILE_NAME_KEY']])[0]

        # Write images
        primary_hdu = fits.PrimaryHDU()
        hdu_list = fits.HDUList([primary_hdu])
        for i in range(self.param_dict['TELESCOPE']['N_CCD']):
            saturate = self.header_list[i][self.param_dict['TELESCOPE']['SATURATE_KEY']]
            img_tmp = np.copy(ccd_images[i][0].array)
            img_tmp[img_tmp > saturate] = saturate
            # hdu_list.append(fits.CompImageHDU(img_tmp, header=ccd_images[i][1], name='CCD_{}'.format(i)))
            hdu_list.append(fits.CompImageHDU(img_tmp, header=self.header_list[i], name='CCD_{}'.format(i)))
        hdu_list.writeto(self.output_image_path + '/simu_image-{}.fits'.format(ori_name))

        # Write weights
        if self.param_dict['FILE']['MAKE_WEIGHT']:
            primary_hdu = fits.PrimaryHDU()
            hdu_list = fits.HDUList([primary_hdu])
            for i in range(self.param_dict['TELESCOPE']['N_CCD']):
                img_tmp = np.ones_like(ccd_images[0][0].array)
                hdu_list.append(fits.CompImageHDU(img_tmp, name='CCD_{}'.format(i)))
            hdu_list.writeto(self.output_image_path + '/simu_weight-{}.fits'.format(ori_name))

        # Write flags
        if self.param_dict['FILE']['MAKE_FLAG']:
            primary_hdu = fits.PrimaryHDU()
            hdu_list = fits.HDUList([primary_hdu])
            for i in range(self.param_dict['TELESCOPE']['N_CCD']):
                img_tmp = np.zeros_like(ccd_images[0][0].array)
                hdu_list.append(fits.CompImageHDU(img_tmp, name='CCD_{}'.format(i)))
            hdu_list.writeto(self.output_image_path + '/simu_flag-{}.fits'.format(ori_name))

        # Write catalogs
        out_cat = io.FITSCatalog(self.output_catalog_path + '/simu_cat-{}.fits'.format(ori_name),
                                 open_mode=io.BaseCatalog.OpenMode.ReadWrite)
        for i in range(self.param_dict['TELESCOPE']['N_CCD']):
            out_cat.save_as_fits(ccd_cats[i], ext_name='CCD_{}'.format(i))
        
        # Write PSF catalogs
        if self.param_dict['PSF']['SAVE_PSF']:
            out_psf_cat = io.FITSCatalog(self.output_catalog_path + '/simu_psf_cat-{}.fits'.format(ori_name),
                                         open_mode=io.BaseCatalog.OpenMode.ReadWrite)
            for i in range(self.param_dict['TELESCOPE']['N_CCD']):
                out_psf_cat.save_as_fits(psf_cats[i], ext_name='CCD_{}'.format(i))

    def go(self, seed=1234):
        """
        """

        ccd_img, ccd_cat, psf_cat = self.runner(seed)
        self.write_output(ccd_img, ccd_cat, psf_cat)
