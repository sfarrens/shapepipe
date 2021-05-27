# -*- coding: utf-8 -*-

"""PSFEX RUNNER

This module run PSFEx.

:Author: Axel Guinot

"""

import re
import os
from shapepipe.pipeline.execute import execute
from shapepipe.modules.module_decorator import module_runner

from astropy.io import fits
from shapepipe.pipeline import file_io as io


@module_runner(input_module=['split_exp_runner', 'mask_runner', 'setools_runner'], version='1.0',
               file_pattern=['image', 'weight', 'pipeline_flag', 'star_selection'], file_ext=['.fits', '.fits', '.fits', '.fits'],
               executes='piffify')
def piff_runner(input_file_list, run_dirs, file_number_string,
                config, w_log):

    exec_path = config.getexpanded("PIFF_RUNNER", "EXEC_PATH")
    piff_config = config.getexpanded("PIFF_RUNNER", "PIFF_CONFIG_FILE")
    tmp_dir = config.getexpanded("PIFF_RUNNER", "PIFF_TMP_DIR")
    clear_tmp = config.getboolean("PIFF_RUNNER", "PIFF_CLEAR_TMP")
    outcat_name = 'piff_cat{}.piff'.format(file_number_string)
    

    
    # Prep data
    tmp_file_name = 'image' + file_number_string + '.fits.fz'
    full_tmp_path = tmp_dir + '/' + tmp_file_name
    img_file = fits.open(input_file_list[0])
    weight_file = fits.open(input_file_list[1])
    flag_file = fits.open(input_file_list[2])

    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())
    img_hdu = fits.CompImageHDU(data=img_file[0].data, header=img_file[0].header)
    weight_hdu = fits.CompImageHDU(data=weight_file[0].data, header=weight_file[0].header)
    flag_hdu = fits.CompImageHDU(data=flag_file[0].data, header=flag_file[0].header)
    hdu_list.append(img_hdu)
    hdu_list.append(weight_hdu)
    hdu_list.append(flag_hdu)
    hdu_list.writeto(full_tmp_path)

    # Run Piff
    cmd_line = "{} {} input.dir={} input.image_file_name={} input.cat_file_name={} output.dir={} output.file_name={}".format(
               exec_path, piff_config, 
               tmp_dir, tmp_file_name, input_file_list[3], 
               run_dirs['output'], outcat_name)

    print(cmd_line)

    stderr, stdout = execute(cmd_line)

    # Clear tmp files
    if clear_tmp:
        execute("rm {}".format(full_tmp_path))

    return stdout, stderr
