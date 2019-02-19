# -*- coding: utf-8 -*-

"""SEXTRACTOR RUNNER

This module run SExtractor.

:Author: Axel Guinot

"""

import re
from shapepipe.pipeline.execute import execute
from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io

import numpy as np


def make_post_process(cat_path, f_wcs_path, pos_params):
    """Make post process

    This function will add one hdu by epoch to the SExtractor catalog. Only works for tiles.
    The columns will be : NUMBER same as SExtractor NUMBER
                          EXP_NAME name of the single exposure for this epoch
                          CCD_N extansion where the object is

    Parameters
    ----------
    cat_path : str
        Path to the outputed SExtractor catalog
    f_wcs_path : str
        Path to the log file containing wcs for all single exp CCDs
    pos_params : list
        World coordinates to use to match the objects.

    """

    cat = io.FITSCatalog(cat_path, SEx_catalog=True, open_mode=io.BaseCatalog.OpenMode.ReadWrite)
    cat.open()

    f_wcs = np.load(f_wcs_path).item()
    n_hdu = len(f_wcs[list(f_wcs.keys())[0]])

    hist = []
    for i in cat.get_data(1)[0][0]:
        if re.split('HISTORY', i)[0] == '': 
            hist.append(i) 

    exp_list = []
    pattern = r'([0-9]*)p\.(.*)'
    for i in hist:
        m = re.search(pattern, i)
        exp_list.append(m.group(1))
    
    obj_id = np.copy(cat.get_data()['NUMBER'])

    n_epoch = np.zeros(len(obj_id), dtype='int32')
    for i, exp in enumerate(exp_list):
        pos_tmp = np.ones(len(obj_id), dtype='int32') * -1
        for j in range(n_hdu):
            w = f_wcs[exp][j]
            pix_tmp = w.all_world2pix(cat.get_data()[pos_params[0]], cat.get_data()[pos_params[1]], 0)
            ind =  ((pix_tmp[0]>0) & (pix_tmp[0]<2112) & (pix_tmp[1]>0) & (pix_tmp[1]<4644))
            pos_tmp[ind] = j
            n_epoch[ind] += 1
        exp_name = np.array([exp_list[i] for n in range(len(obj_id))])
        a = np.array([(obj_id[ii], exp_name[ii], pos_tmp[ii]) for ii in range(len(exp_name))],
                     dtype=[('NUMBER',obj_id.dtype), ('EXP_NAME',exp_name.dtype), ('CCD_N',pos_tmp.dtype)]) 
        cat.save_as_fits(data=a, ext_name='EPOCH_{}'.format(i))
        cat.open()
    
    cat.add_col('N_EPOCH', n_epoch)

    cat.close()


@module_runner(input_module='mask_runner', version='1.0.1',
               file_pattern=['image', 'weight', 'flag'],
               file_ext=['.fits', '.fits', '.fits'],
               executes=['sex'], depends=['numpy'])
def sextractor_runner(input_file_list, output_dir, file_number_string,
                      config, w_log):

    num = file_number_string

    exec_path = config.getexpanded("SEXTRACTOR_RUNNER", "EXEC_PATH")
    dot_sex = config.getexpanded("SEXTRACTOR_RUNNER", "DOT_SEX_FILE")
    dot_param = config.getexpanded("SEXTRACTOR_RUNNER", "DOT_PARAM_FILE")

    weight_file = config.getboolean("SEXTRACTOR_RUNNER", "WEIGHT_IMAGE")
    flag_file = config.getboolean("SEXTRACTOR_RUNNER", "FLAG_IMAGE")
    psf_file = config.getboolean("SEXTRACTOR_RUNNER", "PSF_FILE")

    if config.has_option('SEXTRACTOR_RUNNER', "CHECKIMAGE"):
        check_image = config.getlist("SEXTRACTOR_RUNNER", "CHECKIMAGE")
    else:
        check_image = ['']

    if config.has_option('SEXTRACTOR_RUNNER', 'SUFFIX'):
        suffix = config.get('SEXTRACTOR_RUNNER', 'SUFFIX')
        if (suffix.lower() != 'none') & (suffix != ''):
            suffix = suffix + '_'
        else:
            suffix = ''
    else:
        suffix = ''

    output_file_name = suffix + 'sexcat{0}.fits'.format(num)
    output_file_path = '{0}/{1}'.format(output_dir, output_file_name)

    command_line = ('{0} {1} -c {2} -PARAMETERS_NAME {3} -CATALOG_NAME {4}'
                    ''.format(exec_path, input_file_list[0], dot_sex,
                              dot_param, output_file_path))

    extra = 1
    if weight_file:
        command_line += ' -WEIGHT_IMAGE {0}'.format(input_file_list[extra])
        extra += 1
    if flag_file:
        command_line += ' -FLAG_IMAGE {0}'.format(input_file_list[extra])
        extra += 1
    if psf_file:
        command_line += ' -PSF_NAME {0}'.format(input_file_list[extra])
        extra += 1
    if extra != len(input_file_list):
        raise ValueError("Incoherence between input files and keys related "
                         "to extra files: Found {} extra files, but input "
                         "file list lenght is {}"
                         .format(extra, len(input_file_list)))

    if (len(check_image) == 1) & (check_image[0] == ''):
        check_type = ['NONE']
        check_name = ['none']
    else:
        check_type = []
        check_name = []
        for i in check_image:
            check_type.append(i.upper())
            check_name.append(output_dir + '/' + suffix+i.lower()+num+'.fits')

    command_line += (' -CHECKIMAGE_TYPE {0} -CHECKIMAGE_NAME {1}'
                     ''.format(','.join(check_type), ','.join(check_name)))

    w_log.info('Calling command \'{}\''.format(command_line))

    stderr, stdout = execute(command_line)

    check_error = re.findall('error', stdout.lower())
    check_error2 = re.findall('all done', stdout.lower())

    if check_error == []:
        stderr2 = ''
    else:
        stderr2 = stdout
    if check_error2 == []:
        stderr2 = stdout

    if config.getboolean("SEXTRACTOR_RUNNER", "MAKE_POST_PROCESS"):
        f_wcs_path = config.getexpanded("SEXTRACTOR_RUNNER", "LOG_WCS")
        pos_params = config.getlist("SEXTRACTOR_RUNNER", "WORLD_POSITION")
        make_post_process(output_file_path, f_wcs_path, pos_params)

    return stdout, stderr2
