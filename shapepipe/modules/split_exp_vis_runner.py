# -*- coding: utf-8 -*-

"""SPLIT EXP VIS RUNNER - MODIFIED VERSION OF SPLIT EXP RUNNER (to handle splitting of VIS EXP source files)

This module splits the different CCD's hdu of a single exposure into separate
files.

In EUCLID each source file mixes HDU of 3 types: image (SCI), weight(RMS) and flag(FLG).

:Author: Axel Guinot - Modified by Joel Gehin for EUCLID

"""


import numpy as np
import sip_tpv as stp
from astropy.wcs import WCS
from astropy.io import fits

from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io


def create_hdus(exp_path, output_dir, output_name, n_hdu=108):
    """ Create HDUs

    Split a single exposures CCDs into separate files.

    exp_path : str
        Path to the single exp.
    output_dir : str
        Path to the output directory.
    output_sufix : str # to be modified to adapt to multi type fits file
        Suffix for the output file.
    n_hdu : int
        Number of HDUs (i.e. : number of CCDs).
    transf_coord : bool
        If True will transform the WCS (pv to sip).
    transf_int : bool
        If True will set datas to int.
    save_header : bool
        If True will save WCS information

    """
    
    header_file = np.zeros(n_hdu//3, dtype='O')
    for i in range(1, n_hdu+1):
       
        h = fits.getheader(exp_path, i)
        ext_name = h['EXTNAME']
        d = fits.getdata(exp_path, i)
        
        # target indices for image weight and flag should be a contiguous series of integer, deduced from origin i indices
        j = int((i-1)//3)+1
        
        output_suffix = 'undefined'
        if 'SCI' in ext_name:
            output_suffix = 'image'
            try:
                stp.pv_to_sip(h)
            except:
                w_log.info('Coord. transform pv to sip failed on hdu {}'.format(i))
            w = WCS(h)
            header_file[j-1] = w
        if 'RMS' in ext_name:
            output_suffix = 'weight'
        if 'FLG' in ext_name:
            output_suffix = 'flag'
            d = d.astype(np.int16)
        
        file_name = (output_dir + '/' + output_suffix + output_name +
                     '-' + str(j-1) + '.fits')
        new_file = io.FITSCatalog(file_name,
                                  open_mode=io.BaseCatalog.OpenMode.ReadWrite)
        new_file.save_as_fits(data=d, image=True, image_header=h)

    file_name = output_dir + '/' + 'headers' + output_name + '.npy'
    np.save(file_name, header_file)

@module_runner(version='1.0', file_pattern=['EUC_VIS_SWL-DET'],
               file_ext=['.fits'],
               depends=['numpy', 'astropy', 'sip_tpv'])
def split_exp_vis_runner(input_file_list, run_dirs, file_number_string,
                     config, w_log):

    file_suffix = config.getlist("SPLIT_EXP_VIS_RUNNER", "OUTPUT_SUFFIX")
    n_hdu = config.getint("SPLIT_EXP_VIS_RUNNER", "N_HDU")

    for exp_path in input_file_list :

        create_hdus(exp_path, run_dirs['output'], file_number_string, n_hdu)

    return None, None
