# -*- coding: utf-8 -*-

"""SPLIT STK VIS RUNNER - MODIFIED VERSION OF SPLIT EXP RUNNER to handle splitting of VIS stack files

This module split the different CCD's hdu of a stacked exposure of VIS into separate
files.

:Author:  Joel Gehin

"""


import numpy as np
import sip_tpv as stp
from astropy.wcs import WCS
from astropy.io import fits

from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io


def create_hdus(exp_path, output_dir, output_name, output_suffix, n_hdu = 3,
                transf_coord=False, transf_int=False, save_header=False):
    """ Create HDUs

    Split a stacked exposures CCDs of VIS containing image stack, weight stack and flag stack into 3 separate files.

    exp_path : str
        Path to the stacked exp.
    output_dir : str
        Path to the output directory.
    output_sufix : str # to be modified to adapt to multi type fits file
        Suffix for the output file.
    n_hdu : int
        Number of HDUs (i.e. : number of CCDs).

    """
    for i in range(1, n_hdu+1):

        h = fits.getheader(exp_path, i)
        ext_name = h['EXTNAME']
        d = fits.getdata(exp_path, i)
        
        output_suffix = 'undefined'
        if 'SCI' in ext_name:
            output_suffix = 'image'
            try:
                stp.pv_to_sip(h)
            except:
                w_log.info('Coord. transform pv to sip failed on hdu {}'.format(i))
        if 'RMS' in ext_name:
            output_suffix = 'weight'
        if 'FLG' in ext_name:
            output_suffix = 'flag'
            d = d.astype(np.int16)
        
        
        j = int((i-1)//3)+1
        file_name = (output_dir + '/' + output_suffix + output_name +
                     '-' + str(j-1) + '.fits')
        new_file = io.FITSCatalog(file_name,
                                  open_mode=io.BaseCatalog.OpenMode.ReadWrite)
        new_file.save_as_fits(data=d, image=True, image_header=h)


@module_runner(version='1.0', file_pattern=['stk'],
               file_ext=['.fits'],
               depends=['numpy', 'astropy', 'sip_tpv'])
def split_stk_vis_runner(input_file_list, run_dirs, file_number_string,
                     config, w_log):

    file_suffix = config.getlist("SPLIT_STK_VIS_RUNNER", "OUTPUT_SUFFIX")
    n_hdu = config.getint("SPLIT_STK_VIS_RUNNER", "N_HDU")

    for exp_path in input_file_list:

        create_hdus(exp_path, run_dirs['output'], file_number_string,
                    n_hdu)

    return None, None
