#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:21:18 2020

@author: qyzh
"""

def ashley_model(
    bt_in=8,
    sw_in=450,
    n_in=5,
    ca_in=180.,
    coeff_fn='ASHLEY_A_coeffs.cdf',
    nbt=8,
    nmlat=40,
    nmlt=24,
    nchannel=19,
    LMAX=4,
    NMAX=4,
    cf_inf=17000,
    ref_cfs=None,
    mlts_in=None,
    mlats_in=None,
    date=None,
):
    import numpy as np
    from spacepy import pycdf
    from util import calc_cf, determine_bt_models, determine_weights, determine_weights_small_bt
    from util import rec_mlt_coeffs, rec_uniform_diff_ef
    from util import calc_slope_yint, calc_slope_yint1
    from util import calc_int_diff_ef, calc_expansion_rate

    if ref_cfs is None:
        ref_cfs = np.array([4283.2,6073.3,7957.8,9929.7,11941.6,14254.0,17590.4,22770.4])
    if mlts_in is None:
        mlts_in = np.arange(0,24,(24./nmlt))+12./nmlt
    if mlats_in is None:
        mlats_in = np.arange(nmlat)+50.5

    cdf = pycdf.CDF(coeff_fn)
    cdf_dat = cdf.copy()
    cdf.close()
    all_coeff = cdf_dat['all_coeff']
    cat0_coeff = cdf_dat['cat0_coeff']
    slope_coeff = cdf_dat['slope_coeff']
    yint_coeff = cdf_dat['yint_coeff']
    exp_slope_coeff = cdf_dat['exp_slope_coeff']
    exp_yint_coeff = cdf_dat['exp_yint_coeff']

    cf_in = calc_cf(bt_in, sw_in, n_in)
    imod1, imod2 = determine_bt_models(bt_in=cf_in, ref_bts=ref_cfs)
    coeffs1 = all_coeff[imod1,:,:,:,:]
    coeffs2 = all_coeff[imod2,:,:,:,:]
    if imod2 > 0:
        weight1, weight2 = determine_weights(bt_in=cf_in, ref_bts=ref_cfs, left=imod1, right=imod2)
    if imod2 == 0:
        weight1, weight2 = determine_weights_small_bt(bt_in=cf_in, bt1=2579., bt2=ref_cfs[0])
    all_diff_ef0 = rec_uniform_diff_ef(mlt_coeffs=cat0_coeff, nchannel=nchannel, nmlat=nmlat, nmlt=nmlt, mlts_in=mlts_in, LMAX=LMAX)
    all_diff_ef0[all_diff_ef0<0.] = 0.
    if imod1 != imod2:
        all_mlt_coeffs1 = rec_mlt_coeffs(coeffs=coeffs1, nchannel=nchannel, nmlat=nmlat, ca_in=ca_in, LMAX=LMAX, NMAX=NMAX)
        all_diff_ef1 = rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1, nchannel=nchannel, nmlat=nmlat, nmlt=nmlt, mlts_in=mlts_in, LMAX=LMAX)
        all_mlt_coeffs2 = rec_mlt_coeffs(coeffs=coeffs2, nchannel=nchannel, nmlat=nmlat, ca_in=ca_in, LMAX=LMAX, NMAX=NMAX)
        all_diff_ef2 = rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs2, nchannel=nchannel, nmlat=nmlat, nmlt=nmlt, mlts_in=mlts_in, LMAX=LMAX)
    else:
        all_mlt_coeffs1 = rec_mlt_coeffs(coeffs=coeffs1, nchannel=nchannel, nmlat=nmlat, ca_in=ca_in, LMAX=LMAX, NMAX=NMAX)
        all_diff_ef2 = rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1, nchannel=nchannel, nmlat=nmlat, nmlt=nmlt, mlts_in=mlts_in, LMAX=LMAX)
        if imod2 == 0:
            all_diff_ef1 = all_diff_ef0
        else:
            all_diff_ef1 = all_diff_ef2
    all_diff_ef1[all_diff_ef1<0] = 0.
    all_diff_ef2[all_diff_ef2<0] = 0.
    rec_diff_ef = all_diff_ef1*weight1 + all_diff_ef2*weight2
    expansion1 = 1.
    slope, yint = calc_slope_yint(slope_coeff, yint_coeff, ca_in, NMAX, nchannel)
    int_diff_ef1 = slope*cf_in + yint
    int_diff_ef1[int_diff_ef1<=0] = 1e-12
    exp_slope, exp_yint = calc_slope_yint1(exp_slope_coeff, exp_yint_coeff, ca_in, NMAX)
    rs = 90 - mlats_in
    ts = (mlts_in-6)/12*np.pi
    int_diff_ef2 = calc_int_diff_ef(ts, rs, rec_diff_ef, nchannel)
    int_diff_ef2[int_diff_ef2<=0] = 1e-12
    scaling_fac = int_diff_ef1/int_diff_ef2
    rec_diff_ef[:,:,:11] *= scaling_fac[:11]
    if imod1 == nbt-1:
        coeffs1 = all_coeff[imod1-1,:,:,:,:]
        all_mlt_coeffs1 = rec_mlt_coeffs(coeffs=coeffs1, nchannel=nchannel, nmlat=nmlat, ca_in=ca_in, LMAX=LMAX, NMAX=NMAX)
        all_diff_ef1 = rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1, nchannel=nchannel, nmlat=nmlat, nmlt=nmlt, mlts_in=mlts_in, LMAX=LMAX)
        all_diff_ef1[all_diff_ef1<0] = 0.
        all_diff_ef2[all_diff_ef2<0] = 0.
        int_diff_ef1 = calc_int_diff_ef(ts, rs, all_diff_ef1, nchannel)
        int_diff_ef2 = calc_int_diff_ef(ts, rs, all_diff_ef2, nchannel)
        rate = (int_diff_ef2-int_diff_ef1)/(ref_cfs[-1]-ref_cfs[-2])
        diff_bt = cf_in-ref_cfs[-1]
        if diff_bt < 0:
            diff_bt = 0.
        int_diff_ef3 = int_diff_ef2 + rate*diff_bt
        increment = (int_diff_ef3/int_diff_ef2)
        all_diff_ef3 = all_diff_ef2*increment
        all_diff_ef3[:,:,:11] = rec_diff_ef[:,:,:11]
        expansion1 = calc_expansion_rate(cf_in, cf_inf, ref_cfs[-1], exp_slope, exp_yint)
        int_diff_ef3 = calc_int_diff_ef(ts, rs, all_diff_ef3, nchannel)
        int_diff_ef4 = calc_int_diff_ef(ts, rs*expansion1, all_diff_ef3, nchannel)
        scale_fac = int_diff_ef3/int_diff_ef4
        rec_diff_ef = all_diff_ef3*scale_fac
    mlts_out = mlts_in
    mlats_out = 90.-(90-mlats_in)*expansion1
    diff_ef_out = rec_diff_ef*1e8
    return mlts_out, mlats_out, diff_ef_out


if __name__ == "__main__":
    ashley_model()