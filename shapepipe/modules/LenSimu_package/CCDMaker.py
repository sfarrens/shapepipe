

import numpy as np

import galsim

from astropy.wcs import WCS

from shapepipe.modules.LenSimu_package.StarMaker import StarMaker
from shapepipe.modules.LenSimu_package.GalaxyMaker import GalaxyMaker


class CCDMaker(object):
    """
    """

    def __init__(self, ccd_number, header, 
                param_dict,
                seed):

        self.param_dict = param_dict
        self.seed = seed

        self.ccd_number = ccd_number
        self.header = header

        self._init_randoms(seed)
        self._init_parameters()
        self.init_full_image()

    def _init_randoms(self, seed):
        """
        """

        # Init random
        np.random.seed(seed)
        self._ud = galsim.BaseDeviate(seed)

    def _init_parameters(self):
        """
        """

        # Init CCD image values
        self.param_dict['ccd_size_x'] = self.header[self.param_dict['TELESCOPE']['CCD_SIZE_X_KEY']]
        self.param_dict['ccd_size_y'] = self.header[self.param_dict['TELESCOPE']['CCD_SIZE_Y_KEY']]
        self.param_dict['data_sec'] = self.param_dict['TELESCOPE']['DATA_SEC']
        self.ccd_wcs = galsim.AstropyWCS(wcs=WCS(self.header))

    def init_full_image(self):
        """
        """

        self.ccd_image = galsim.ImageI(self.param_dict['ccd_size_x'],
                                       self.param_dict['ccd_size_y'])
        
        self.ccd_image.setOrigin(1,1)
        self.ccd_image.wcs = self.ccd_wcs

    def go(self, gal_catalog, star_catalog):
        """
        """

        self.add_objects(gal_catalog, star_catalog)
        self.finalize_full_image()

        return self.ccd_image, self.final_catalog, self.psf_catalog

    def add_objects(self, gal_catalog, star_catalog):
        """
        """

        _StarMaker = StarMaker(self.param_dict, self.seed)

        final_catalog = {'id': [],
                         'mice_gal_id': [], 'mice_halo_id': [],
                         'flag_central': [],
                         'ra': [], 'dec': [], 'z': [],
                         'flux': [], 'mag': [],
                         'bulge_total_ratio': [],
                         'disk_hlr': [], 'disk_q': [], 'disk_beta': [],
                         'bulge_hlr': [], 'bulge_q': [], 'bulge_beta': [],
                         'intrinsic_g1': [], 'intrinsic_g2': [],
                         'shear_gamma1': [], 'shear_gamma2': [], 'shear_kappa': [],
                         'shear_g1': [], 'shear_g2': [], 'shear_mu': [],
                         'Qxx': [], 'Qyy': [], 'Qxy': [],
                         'psf_g1': [], 'psf_g2': [], 'psf_fwhm': [],
                         'type': []}
        if self.param_dict['PSF']['SAVE_PSF']:
            psf_catalog = {'id': [],
                           'psf_g1': [], 'psf_g2': [], 'psf_fwhm': [],
                           'psf_vignet': []}

        id_n = 0
        for gal_cat in gal_catalog:
            img_pos = self.ccd_wcs.toImage(galsim.CelestialCoord(gal_cat['ra']*galsim.degrees,
                                                                 gal_cat['dec']*galsim.degrees))

            if (img_pos.x < self.param_dict['TELESCOPE']['DATA_SEC'][0]) | (img_pos.x > self.param_dict['TELESCOPE']['DATA_SEC'][1]) | (img_pos.y < self.param_dict['TELESCOPE']['DATA_SEC'][2]) | (img_pos.y > self.param_dict['TELESCOPE']['DATA_SEC'][3]):
                continue

            # Make galaxy model
            if self.param_dict['GALAXY']['CONVERT_MAG_SDSS']:
                try:
                    mag = self._convert_mag(gal_cat['sdss_r_mag'], gal_cat['sdss_r_mag'])
                except:
                    raise ValueError("Either 'sdss_r_mag' or 'sdss_r_mag' is not defined.")
            else:
                mag = gal_cat(gal_cat['mag'])
            flux = 10**((mag - self.header[self.param_dict['TELESCOPE']['MAG_ZP_KEY']])/(-2.5))
            gal, intrinsic_g1, intrinsic_g2, moments, shear_g1, shear_g2, shear_mu = GalaxyMaker(self.param_dict).make_gal(flux, 
                                         gal_cat['bulge_total_ratio'],
                                         gal_cat['disk_hlr'],
                                         gal_cat['disk_q'],
                                         gal_cat['disk_beta'],
                                         gal_cat['bulge_hlr'],
                                         gal_cat['bulge_q'],
                                         gal_cat['bulge_beta'],
                                         gal_cat['shear_gamma1'],
                                         gal_cat['shear_gamma2'],
                                         gal_cat['shear_kappa'])
            
            # Make PSF
            psf, psf_g1, psf_g2, psf_fwhm = _StarMaker.make_star(1, self.ccd_number, img_pos.x, img_pos.y)

            # Final obj
            obj = galsim.Convolve([gal, psf])

            stamp = self.draw_stamp(obj, img_pos)

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog['id'].append(id_n)
            final_catalog['mice_gal_id'].append(gal_cat['mice_gal_id'])
            final_catalog['mice_halo_id'].append(gal_cat['mice_halo_id'])
            final_catalog['flag_central'].append(gal_cat['flag_central'])
            final_catalog['ra'].append(gal_cat['ra'])
            final_catalog['dec'].append(gal_cat['dec'])
            final_catalog['z'].append(gal_cat['z'])
            final_catalog['flux'].append(flux)
            final_catalog['mag'].append(mag)
            final_catalog['bulge_total_ratio'].append(gal_cat['bulge_total_ratio'])
            final_catalog['disk_hlr'].append(gal_cat['disk_hlr'])
            final_catalog['disk_q'].append(gal_cat['disk_q'])
            final_catalog['disk_beta'].append(gal_cat['disk_beta'])
            final_catalog['bulge_hlr'].append(gal_cat['bulge_hlr'])
            final_catalog['bulge_q'].append(gal_cat['bulge_q'])
            final_catalog['bulge_beta'].append(gal_cat['bulge_beta'])
            final_catalog['intrinsic_g1'].append(intrinsic_g1)
            final_catalog['intrinsic_g2'].append(intrinsic_g2)
            final_catalog['shear_gamma1'].append(gal_cat['shear_gamma1'])
            final_catalog['shear_gamma2'].append(gal_cat['shear_gamma2'])
            final_catalog['shear_kappa'].append(gal_cat['shear_kappa'])
            final_catalog['shear_g1'].append(shear_g1)
            final_catalog['shear_g2'].append(shear_g2)
            final_catalog['shear_mu'].append(shear_mu)
            final_catalog['Qxx'].append(moments[...,0,0])
            final_catalog['Qyy'].append(moments[...,1,1])
            final_catalog['Qxy'].append(moments[...,0,1])
            final_catalog['psf_g1'].append(psf_g1)
            final_catalog['psf_g2'].append(psf_g2)
            final_catalog['psf_fwhm'].append(psf_fwhm)
            final_catalog['type'].append(1)

            if self.param_dict['PSF']['SAVE_PSF']:
                psf_vign = self.draw_psf(psf, img_pos)
                psf_catalog['id'].append(id_n)
                psf_catalog['psf_g1'].append(psf_g1)
                psf_catalog['psf_g2'].append(psf_g2)
                psf_catalog['psf_fwhm'].append(psf_fwhm)
                psf_catalog['psf_vignet'].append(psf_vign)
            
            id_n += 1


        for star_cat in star_catalog:
            img_pos = self.ccd_wcs.toImage(galsim.CelestialCoord(star_cat['ra']*galsim.degrees,
                                                                 star_cat['dec']*galsim.degrees))
            
            if (img_pos.x < self.param_dict['TELESCOPE']['DATA_SEC'][0]) | (img_pos.x > self.param_dict['TELESCOPE']['DATA_SEC'][1]) | (img_pos.y < self.param_dict['TELESCOPE']['DATA_SEC'][2]) | (img_pos.y > self.param_dict['TELESCOPE']['DATA_SEC'][3]):
                continue

            # Make star model
            # WIP

            mag = star_cat['mag']
            flux = 10**((mag - self.header[self.param_dict['TELESCOPE']['MAG_ZP_KEY']])/(-2.5))
            
            # Make PSF
            star, psf_g1, psf_g2, psf_fwhm = _StarMaker.make_star(flux, self.ccd_number, img_pos.x, img_pos.y)
            psf, psf_g1, psf_g2, psf_fwhm = _StarMaker.make_star(1, self.ccd_number, img_pos.x, img_pos.y)

            # Final obj
            obj = star

            stamp = self.draw_stamp(obj, img_pos)

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog['id'].append(id_n)
            final_catalog['mice_gal_id'].append(-10)
            final_catalog['mice_halo_id'].append(-10)
            final_catalog['flag_central'].append(-10)
            final_catalog['ra'].append(star_cat['ra'])
            final_catalog['dec'].append(star_cat['dec'])
            final_catalog['z'].append(-10)
            final_catalog['flux'].append(flux)
            final_catalog['mag'].append(mag)
            final_catalog['bulge_total_ratio'].append(-10)
            final_catalog['disk_hlr'].append(-10)
            final_catalog['disk_q'].append(-10)
            final_catalog['disk_beta'].append(-10)
            final_catalog['bulge_hlr'].append(-10)
            final_catalog['bulge_q'].append(-10)
            final_catalog['bulge_beta'].append(-10)
            final_catalog['intrinsic_g1'].append(-10)
            final_catalog['intrinsic_g2'].append(-10)
            final_catalog['shear_gamma1'].append(-10)
            final_catalog['shear_gamma2'].append(-10)
            final_catalog['shear_kappa'].append(-10)
            final_catalog['shear_g1'].append(-10)
            final_catalog['shear_g2'].append(-10)
            final_catalog['shear_mu'].append(-10)
            final_catalog['Qxx'].append(-10)
            final_catalog['Qyy'].append(-10)
            final_catalog['Qxy'].append(-10)
            final_catalog['psf_g1'].append(psf_g1)
            final_catalog['psf_g2'].append(psf_g2)
            final_catalog['psf_fwhm'].append(psf_fwhm)
            final_catalog['type'].append(0)

            if self.param_dict['PSF']['SAVE_PSF']:
                psf_vign = self.draw_psf(psf, img_pos)
                psf_catalog['id'].append(id_n)
                psf_catalog['psf_g1'].append(psf_g1)
                psf_catalog['psf_g2'].append(psf_g2)
                psf_catalog['psf_fwhm'].append(psf_fwhm)
                psf_catalog['psf_vignet'].append(psf_vign)
            
            id_n += 1

        self.final_catalog = final_catalog
        if self.param_dict['PSF']['SAVE_PSF']:
            self.psf_catalog = psf_catalog
        else:
            self.psf_catalog = None

    def finalize_full_image(self):
        """
        """

        sky_background = self.header[self.param_dict['TELESCOPE']['BACKGROUND_VALUE_KEY']]

        # sky_background = 1

        noise = galsim.PoissonNoise(sky_level=sky_background, rng=self._ud)

        self.ccd_image.addNoise(noise)

        self.ccd_image += sky_background

    def draw_stamp(self, galsim_obj, img_pos):
        """
        """

        # Handling of position (cf. galsim demo11.py)
        # Guess: this is to have the stamp center at integer value,
        # ix_nominal, to have a good integration in final big image. 
        # We account for intra-pixel shift as an offset, dx.
        x_nominal = img_pos.x + 0.5
        y_nominal = img_pos.y + 0.5
        ix_nominal = int(np.floor(x_nominal + 0.5))
        iy_nominal = int(np.floor(y_nominal + 0.5))
        dx = x_nominal - ix_nominal
        dy = y_nominal - iy_nominal
        offset = galsim.PositionD(dx, dy)

        if galsim_obj.flux <= 1e7:
            stamp = galsim_obj.drawImage(wcs=self.ccd_wcs.local(img_pos), 
                                         offset=offset,
                                         method='phot',
                                         rng=self._ud,
                                         nx=151,
                                         ny=151)
        else:
            stamp = galsim_obj.drawImage(wcs=self.ccd_wcs.local(img_pos), 
                                         offset=offset,
                                         method='fft',
                                         nx=1001,
                                         ny=1001)
            stamp.quantize()
            img_tmp = np.copy(stamp.array)
            img_tmp[img_tmp < 0] = 0
            poisson_noise = np.random.poisson(lam=img_tmp) - img_tmp
            stamp += poisson_noise

        stamp.setCenter(ix_nominal, iy_nominal)

        return stamp

    def draw_psf(self, galsim_obj, img_pos):
        """
        """

        stamp = galsim_obj.drawImage(wcs=self.ccd_wcs.local(img_pos),
                                     nx=51,
                                     ny=51)

        return stamp.array



    def _convert_mag(self, sdss_r, sdss_g):
        """
        """

        mc_r = sdss_r - 0.087*(sdss_g - sdss_r)

        return mc_r
