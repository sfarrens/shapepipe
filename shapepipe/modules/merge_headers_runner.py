# -*- coding: utf-8 -*-

"""CREATE LOG EXP HEADER

This module merge the "header" files output of the split_exp_runner.py module.
It create a binnary file that contain the wcs of each CCDs for each exposures.

:Author: Axel Guinot

"""


import os
import re

import numpy as np
from sqlitedict import SqliteDict

from shapepipe.modules.module_decorator import module_runner
from shapepipe.modules.merge_headers_package.merge_headers_script import merge_headers

@module_runner(input_module='split_exp_runner', version='1.0',
               file_pattern=['headers'],
               file_ext=['.npy'], depends=['numpy', 'sqlitedict'],
               run_method='serial')
def merge_headers_runner(input_file_list, run_dirs, file_number_string,
                         config, module_config_sec, w_log):

    output_dir = run_dirs['output']
    if config.has_option('MERGE_HEADERS_RUNNER', 'OUTPUT_PATH'):
        output_dir = config.getexpanded('MERGE_HEADERS_RUNNER', 'OUTPUT_PATH')
    w_log.info('output_dir = {}'.format(output_dir))
    
    merge_headers(output_dir, input_file_list)
    
    return None, None



