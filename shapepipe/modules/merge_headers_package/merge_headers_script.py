"""
CREATE LOG EXP HEADER

This module merge the "header" files output of the split_exp_runner.py module.
It create a binnary file that contain the wcs of each CCDs for each exposures.

:Author: Axel Guinot
"""

import os

import numpy as np
from sqlitedict import SqliteDict


def merge_headers(output_dir, input_file_list):

   final_file = SqliteDict(output_dir + '/log_exp_headers.sqlite')
   for file_path in input_file_list:
       file_path_scalar = file_path[0]
       file_name = os.path.split(file_path_scalar)[1]
       file_base_name = os.path.splitext(file_name)[0]
       pattern = 'headers-'
       m = re.split(pattern, file_base_name)
       if len(m) < 2:
           raise IndexError('Regex \'{}\' not found in base name \'{}\''
                            ''.format(pattern, file_base_name))
       key = m[1]
       final_file[key] = np.load(file_path_scalar, allow_pickle=True)

   final_file.commit()
   final_file.close()

   return None, None
