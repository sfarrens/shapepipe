# -*- coding: utf-8 -*-

"""PSFEX_INTERP RUNNER

This file is the pipeline runner for the PSFExInterpolation package.

:Author: Axel Guinot

"""

from shapepipe.modules.module_decorator import module_runner
from shapepipe.modules.PSFExInterpolation_package import interpolation_script

import re


def get_psfex_run_dir(output_dir):
    """
    """

    # return '/' + '/'.join(re.split('/', output_dir)[1:-2]) + '/psfex_runner/output'
    return '/s03data2/guinot/pipeline_output/shapepipe_run_2019-08-04_11-59-50/psfex_runner/output/'


@module_runner(input_module=['piff_runner', 'setools_runner'], version='1.0',
               file_pattern=['star_selection', 'galaxy_selection'],
               file_ext=['.piff', '.fits'],
               depends=['numpy', 'astropy', 'galsim', 'sqlitedict'])
def piff_interp_runner(input_file_list, run_dirs, file_number_string,
                           config, w_log):

    mode = config.get('PIFF_INTERP_RUNNER', 'MODE')

    pos_params = config.getlist('PIFF_INTERP_RUNNER', 'POSITION_PARAMS')
    get_shapes = config.getboolean('PIFF_INTERP_RUNNER', 'GET_SHAPES')
    star_thresh = config.getint('PIFF_INTERP_RUNNER', 'STAR_THRESH')
    chi2_thresh = None

    if mode == 'CLASSIC':
        psfcat_path, galcat_path = input_file_list

        inst = interpolation_script.PSFExInterpolator(psfcat_path, galcat_path,
                                                      run_dirs['output'], file_number_string, w_log,
                                                      pos_params, get_shapes, star_thresh, chi2_thresh,
                                                      True)
        inst.process()

    elif mode == 'MULTI-EPOCH':
        # dot_psf_dir = config.getexpanded('PSFEX_INTERP_RUNNER_ME', 'ME_DOT_PSF_DIR')
        dot_psf_dir = get_psfex_run_dir(run_dirs['output'])
        dot_psf_pattern = config.get('PIFF_INTERP_RUNNER', 'ME_DOT_PSF_PATTERN')
        f_wcs_path = config.getexpanded('PIFF_INTERP_RUNNER', 'ME_LOG_WCS')

        galcat_path = input_file_list[0]

        inst = interpolation_script.PSFExInterpolator(None, galcat_path,
                                                      run_dirs['output'], file_number_string, w_log,
                                                      pos_params, get_shapes, star_thresh, chi2_thresh,
                                                      True)

        inst.process_me(dot_psf_dir, dot_psf_pattern, f_wcs_path)

    elif mode == 'VALIDATION':
        psfcat_path, galcat_path = input_file_list

        inst = interpolation_script.PSFExInterpolator(psfcat_path, galcat_path,
                                                      run_dirs['output'], file_number_string, w_log,
                                                      pos_params, get_shapes, star_thresh, chi2_thresh,
                                                      True)

        inst.process_validation(None)

    else:
        ValueError('MODE has to be in : [C, ME]')

    return None, None
