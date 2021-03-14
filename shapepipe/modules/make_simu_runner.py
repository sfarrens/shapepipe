# -*- coding: utf-8 -*-

"""SIMU RUNNER

This file is the pipeline runner for the mask simulations.

:Author: Axel Guinot

"""

from shapepipe.modules.module_decorator import module_runner
from shapepipe.modules.LenSimu_package.ExposureMaker import ExposureMaker

from astropy.io import fits

from sqlitedict import SqliteDict

import sip_tpv as stp

import re


@module_runner(version='1.0',
               file_pattern=['image'],
               file_ext=['.fits'],
               depends=['numpy', 'astropy', 'galsim', 'sip_tpv'],
               numbering_scheme='-0000000')
def make_simu_runner(input_file_list, run_dirs, file_number_string,
                    config, w_log):
    """
    """

    path_simu_gal_cat = config.getexpanded('MAKE_SIMU_RUNNER', 'INPUT_GAL_CAT')
    path_simu_star_cat = config.getexpanded('MAKE_SIMU_RUNNER', 'INPUT_STAR_CAT')

    psf_file_dir = config.getexpanded('MAKE_SIMU_RUNNER', 'PSF_FILE_DIR')

    only_header = config.getboolean('MAKE_SIMU_RUNNER', 'ONLY_HEADER')

    # gal_cat = fits.getdata(path_simu_gal_cat, 1)
    gal_cat = SqliteDict(path_simu_gal_cat)
    star_cat = fits.getdata(path_simu_star_cat, 1)

    seed = int(re.findall('\d+', file_number_string)[0])

    if only_header:
        header_list = []
        f = open(input_file_list[0], 'r', encoding='latin1')
        split_f = re.split('\nEND', f.read())
        n_ext = len(split_f)
        for i in range(1, n_ext-1):
            h_tmp = fits.Header.fromstring(re.split('\nEND', split_f[i])[0], sep='\n')
            stp.pv_to_sip(h_tmp)
            header_list.append(h_tmp)
        f.close()
    else:
        header_list = []
        for i in range(1,41):
            h_tmp = fits.getheader(input_file_list[0], i)
            # stp.pv_to_sip(h_tmp)
            header_list.append(h_tmp)

    exp_runner = ExposureMaker(header_list, gal_cat, star_cat,
                               run_dirs['output'], psf_file_dir)
    exp_runner.go(seed)
    gal_cat.close()

    return None, None
