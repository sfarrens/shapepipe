# -*- coding: utf-8 -*-

"""SWARP RUNNER

This module run swarp.

:Author: Axel Guinot

"""

import re
import os

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

import sip_tpv as stp

from shapepipe.pipeline.execute import execute
from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io


def get_ra_dec(xxx, yyy):
    """Get ra/dec
       
       Transform xxx/yyy notation to ra/dec
       
       Parameters
       ----------
       xxx: int
       yyy: int

       Retunrs
       -------
       ra, dec: float, float
    """

    return xxx/2/np.cos((yyy/2-90)*np.pi/180), yyy/2-90

def prep_single_image(image_path, tmp_dir, num):
    """
    """

    single_num = re.findall('\d+', os.path.split(image_path)[1])[0]

    new_image_path = tmp_dir + '/tmp_image{}-{}.fits'.format(num, single_num)

    ori_image = fits.open(image_path)
    primary_hdu = fits.PrimaryHDU()
    hdu_list = fits.HDUList([primary_hdu])
    for i in range(40):
        tmp_header = ori_image[i+1].header
        # stp.sip_to_pv(tmp_header)
        # hdu_list.append(fits.CompImageHDU(ori_image[i+1].data-tmp_header['IMMODE'], header=tmp_header, name='CCD_{}'.format(i)))
        hdu_list.append(fits.CompImageHDU(ori_image[i+1].data, header=tmp_header, name='CCD_{}'.format(i)))
    hdu_list.writeto(new_image_path)
    
    return new_image_path

def get_history(coadd_path, image_path_list):
    """ Get history

    Write in the coadd header the single exposures used and how many CCDs from
    them.

    Parameters
    ----------
    coadd_path : str
        Path to the coadd image.
    image_path_list : list
        List of the single exposures path to check

    """
    coadd_file = io.FITSCatalog(coadd_path, hdu_no=0,
                                open_mode=io.BaseCatalog.OpenMode.ReadWrite)
    coadd_file.open()
    wcs_coadd = WCS(coadd_file.get_header())
    corner_coadd = wcs_coadd.calc_footprint().T

    for img_path in image_path_list:
        if (img_path == '\n') or (img_path == ''):
            continue
        img_path = img_path.replace('\n', '')
        img_path = img_path.replace(' ', '')

        f_tmp = io.FITSCatalog(img_path)
        f_tmp.open()
        n_hdu = len(f_tmp.get_ext_name())
        ccd_inter = 0
        for ext in range(1, n_hdu):
            h_tmp = f_tmp._cat_data[ext].header.copy()
            stp.pv_to_sip(h_tmp)
            wcs_tmp = WCS(h_tmp)
            corner_tmp = wcs_tmp.calc_footprint().T
            if (np.min(corner_coadd[0]) > np.max(corner_tmp[0]) or
                    np.max(corner_coadd[0]) < np.min(corner_tmp[0])):
                continue
            if (np.min(corner_coadd[1]) > np.max(corner_tmp[1]) or
                    np.max(corner_coadd[1]) < np.min(corner_tmp[1])):
                continue

            ccd_inter += 1
        f_tmp.close()

        if ccd_inter != 0:
            coadd_file.add_header_card("HISTORY",
                                       "From file {} {} extension(s) used"
                                       "".format(os.path.split(img_path)[1],
                                                 ccd_inter))
    coadd_file.close()


@module_runner(version='1.0',
               file_pattern=['tile'],
               file_ext=['.txt'],
               depends=['numpy', 'astropy', 'sip_tpv'])
def swarp_runner(input_file_list, run_dirs, file_number_string,
                 config, w_log):

    num = file_number_string

    exec_path = config.getexpanded("SWARP_RUNNER", "EXEC_PATH")
    dot_swarp = config.getexpanded("SWARP_RUNNER", "DOT_SWARP_FILE")
    image_prefix = config.get("SWARP_RUNNER", "IMAGE_PREFIX")
    weight_prefix = config.get("SWARP_RUNNER", "WEIGHT_PREFIX")
    bkg_sub_type = config.get("SWARP_RUNNER", "BKG_SUB_TYPE")
    bkg_sub_value = config.get("SWARP_RUNNER", "BKG_SUB_VALUE")
    tmp_dir = config.get("SWARP_RUNNER", "TMP_DIR")
    clear_tmp = config.getboolean("SWARP_RUNNER", "CLEAR_TMP")

    if config.has_option('SWARP_RUNNER', 'SUFFIX'):
        suffix = config.get('SWARP_RUNNER', 'SUFFIX')
        if (suffix.lower() != 'none') and (suffix != ''):
            suffix = suffix + '_'
        else:
            suffix = ''
    else:
        suffix = ''

    if bkg_sub_type == "MANUAL":
        try:
            from_header= False
            bkg_value = [str(float(bkg_sub_value))]
        except:
            from_header = True
    elif bkg_sub_type == "AUTO":
        from_header = False
        bkg_values = ['0']
    else:
        raise ValueError("BKG_SUB_TYPE must be in [AUTO, MANUAL]")

    output_image_name = suffix + 'image{0}.fits'.format(num)
    output_weight_name = suffix + 'weight{0}.fits'.format(num)
    output_image_path = '{0}/{1}'.format(run_dirs['output'], output_image_name)
    output_weight_path = '{0}/{1}'.format(run_dirs['output'],
                                          output_weight_name)

    # Get center position
    # tmp = os.path.split(os.path.splitext(input_file_list[0])[0])[1]
    # tmp = re.split('_|-', tmp)
    # ra, dec = tmp[1], tmp[2]
    tmp = re.findall('\d+', os.path.split(input_file_list[0])[1])
    ra, dec = get_ra_dec(int(tmp[0]), int(tmp[1]))

    # Get weight list
    #new_image_list_path = tmp_dir + '/image_list{}.txt'.format(num)
    #new_image_list_file = open(new_image_list_path, 'w')
    image_file = open(input_file_list[0])
    image_list = image_file.readlines()
    image_file.close()
    weight_list = []
    bkg_values = []
    for img_path in image_list:
        #new_path = prep_single_image(re.split('\n', img_path)[0], tmp_dir, num)
        #new_image_list_file.write(new_path + '\n')
        tmp = os.path.split(img_path)
        new_name = tmp[1].replace(image_prefix,
                                  weight_prefix).replace('\n', '')
        weight_list.append('/'.join([tmp[0], new_name]))
        if from_header:
            h = fits.getheader(re.split('\n', img_path)[0], 1)
            bkg_values.append(str(h[bkg_sub_value]))
    # new_image_list_file.close()

    command_line = '{} @{} -c {}' \
                   ' -WEIGHT_IMAGE {}' \
                   ' -IMAGEOUT_NAME {} -WEIGHTOUT_NAME {}' \
                   ' -RESAMPLE_SUFFIX _resamp{}.fits ' \
                   ' -CENTER_TYPE MANUAL -CENTER {},{} ' \
                   ' -BACK_TYPE {} -BACK_DEFAULT {} ' \
                   ''.format(exec_path, input_file_list[0], dot_swarp,
                             ','.join(weight_list), output_image_path,
                             output_weight_path, num, ra, dec,
                             bkg_sub_type, ','.join(bkg_values))

    w_log.info(command_line)
    print(command_line)

    stderr, stdout = execute(command_line)

    check_error = re.findall('error', stdout.lower())
    check_error2 = re.findall('all done', stdout.lower())

    if check_error == []:
        stderr2 = ''
    else:
        stderr2 = stdout
    if check_error2 == []:
        stderr2 = stdout

    if clear_tmp:
        new_image_list_file = open(new_image_list_path, 'r')
        tmp_image_path = new_image_list_file.readlines()
        new_image_list_file.close()
        for image_path in tmp_image_path:
            cmd_tmp = 'rm -f {}'.format(re.split('\n', image_path)[0])
            _ = execute(cmd_tmp)
        cmd_tmp = 'rm -f {}'.format(new_image_list_path)
        _ = execute(cmd_tmp)

    get_history(output_image_path, image_list)

    return stdout, stderr2
