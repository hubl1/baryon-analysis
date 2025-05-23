#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import gvar as gv
from scipy.optimize import fsolve
import sys
import warnings
import os
from params.params import alttc_gv, name, fm2GeV
warnings.filterwarnings("ignore",category=DeprecationWarning)
from utils.create_plot import create_plot,create_plot_ex
from save_windows import read_ylimit
from filelock import FileLock
from utils.c2pt_io_h5 import h5_get_pp2pt
import json

# ii=int(sys.argv[2])
# jj=int(sys.argv[11])
# kk=int(sys.argv[13])
# print(ii,jj)
# nensemble=int(sys.argv[3])
# cov_enabled=bool(int(sys.argv[12]))
# print(cov_enabled)
# tag=sys.argv[1]
# prefix=sys.argv[9]
# baryon=sys.argv[10]
# print(cov_enabled)
def run_meff_avg_charm_quark(
    prefix, ii, nensemble, mp_string, T_strt, T_end,
    tag, baryon, jj,kk, cov_enabled, baryon_meson_h5_prefix
):

    pydata_path=baryon_meson_h5_prefix+"/pydata_eff_mass/"
    if baryon == 'PION' or baryon == 'KAON' or baryon == 'ETA_S' or ( nensemble == 10 and baryon == 'PROTON' ):
        two_state_enabled=False
    else:
        two_state_enabled=False 

    if cov_enabled:
        bootstrap_resample_data=pydata_path+baryon+'_fpi_mq_mPS_bootstrap_cov_400.npy'
    else:
        bootstrap_resample_data=pydata_path+baryon+'_fpi_mq_mPS_bootstrap_no_cov_400.npy'
    dir_name, file_name = os.path.split(bootstrap_resample_data)
    lock_file = os.path.join(dir_name, '.'+file_name+'.lock')

    with FileLock(lock_file):
        with open(bootstrap_resample_data, 'rb') as f:
            mb_mr_data = np.load(f, allow_pickle=True).item()

    alttc = gv.mean(alttc_gv[nensemble])
    # mPS=gv.gvar(sys.argv[4])

    #load the extra data using light quark in the physical point in 419

    c2pt, confs=h5_get_pp2pt(f'{baryon_meson_h5_prefix}/baryon_meson_data_h5/{tag}/c2pt_{prefix}.wp.{baryon}.h5')
    with open(os.path.expanduser("params/ensemble_used_confs_common.json"), 'r', encoding='utf-8') as f:
        common_conf_data = json.load(f)
    # print(confs,common_conf_data[f'ensemble{nensemble}'])
    # print(pp2pt.shape)
    # print(confs.shape)
    index_conf=[ i for i in range(len(confs)) if confs[i] in common_conf_data[f'ensemble{nensemble}']]

    c2pt=c2pt[index_conf,:]
    Ncnfg = c2pt.shape[0]
    T = c2pt.shape[1]
    T_hlf = T//2

    c2pt_sum = np.sum(c2pt,axis=0)
    c2pt_jcknf = ((c2pt_sum) - c2pt)/(Ncnfg-1) # jack-knife resample
    c2pt_cntrl = np.mean(c2pt_jcknf,axis=0) # jack-knife mean
    c2pt_cov = (Ncnfg-1)*np.cov(np.transpose(c2pt_jcknf,axes=(1,0))) # jack-knife covariance
    c2pt_err = np.sqrt(Ncnfg-1)*np.std(c2pt_jcknf,axis=0)

    c2pt_fwrd_jcknf = c2pt_jcknf[:,0:T_hlf+1]
    c2pt_bwrd_jcknf = c2pt_jcknf[:,T_hlf:][:,::-1]
    c2pt_avg_jcknf = (c2pt_fwrd_jcknf[:,1:]+c2pt_bwrd_jcknf)/2
    # c2pt_avg_jcknf = c2pt_fwrd_jcknf[:,1:]
    c2pt_avg_cntrl = np.mean(c2pt_avg_jcknf,axis=0) # jack-knife mean
    c2pt_avg_cov = (Ncnfg-1)*np.cov(np.transpose(c2pt_avg_jcknf,axes=(1,0))) # jack-knife covariance
    c2pt_avg_err = np.sqrt(Ncnfg-1)*np.std(c2pt_avg_jcknf,axis=0)

    m_eff_log_fwrd_jcknf = np.log(c2pt_fwrd_jcknf[:,:-1]/c2pt_fwrd_jcknf[:,1:])*fm2GeV/alttc
    m_eff_log_bwrd_jcknf = np.log(c2pt_bwrd_jcknf[:,:-1]/c2pt_bwrd_jcknf[:,1:])*fm2GeV/alttc
    m_eff_log_avg_jcknf = np.log(c2pt_avg_jcknf[:,:-1]/c2pt_avg_jcknf[:,1:])*fm2GeV/alttc

    # print(m_eff_log_bwrd_jcknf)

    m_eff_log_fwrd_cntrl = np.mean(m_eff_log_fwrd_jcknf,axis=0)
    m_eff_log_fwrd_err = np.sqrt(Ncnfg-1)*np.std(m_eff_log_fwrd_jcknf,axis=0)
    m_eff_log_bwrd_cntrl = np.mean(m_eff_log_bwrd_jcknf,axis=0)
    m_eff_log_bwrd_err = np.sqrt(Ncnfg-1)*np.std(m_eff_log_bwrd_jcknf,axis=0)
    m_eff_log_avg_cntrl = np.mean(m_eff_log_avg_jcknf,axis=0)
    m_eff_log_avg_err = np.sqrt(Ncnfg-1)*np.std(m_eff_log_avg_jcknf,axis=0)
    # print(m_eff_log_fwrd_cntrl)
    # print(m_eff_log_bwrd_cntrl)
    # exit()

    m_eff_log_cntrl=m_eff_log_avg_cntrl
    m_eff_log_err=m_eff_log_avg_err
    t_ary = np.array(range(1,T//2))
    def eff_mass_eqn(c2pt):
        # return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.exp(E0*(T/2-t_ary)) / np.exp(E0*(T/2-(t_ary+1)))
        return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(T/2-t_ary)) / np.cosh(E0*(T/2-(t_ary+1)))
    # def eff_mass_eqn_test(c2pt,E0):
    #     return (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(T/2-t_ary)) / np.cosh(E0*(T/2-(t_ary+1)))
    def fndroot(eqnf,ini):
        sol = fsolve(eqnf,ini, xtol=1e-10)
        return sol
    m_eff_cosh_jcknf = np.array([fndroot(eff_mass_eqn(c2pt),0.3*np.ones_like(t_ary)) for c2pt in c2pt_avg_jcknf])*fm2GeV/alttc
    m_eff_cosh_avg_jcknf=m_eff_cosh_jcknf
    m_eff_cosh_avg_cntrl = np.mean(m_eff_cosh_avg_jcknf,axis=0)
    # m_eff_cosh_avg_cntrl = m_eff_cosh_avg_jcknf[20,:]
    m_eff_cosh_avg_err = np.sqrt(Ncnfg-1)*np.std(m_eff_cosh_avg_jcknf,axis=0)

    t_ary=np.arange(1,T_hlf)

    def calculate_m_t(c1, c2, m1, m2, T_start, T_end):
        # if T_start <= 0:
        #     raise ValueError("T_start must be greater than 0")

        t_values = np.linspace(T_start, T_end, 1200)
        y_values = np.exp(-m1 * t_values) + c2 / c1 * np.exp(-m2 * t_values)
        m_t_values = (m1 * np.exp(-m1 * t_values) + c2 / c1 * m2 * np.exp(-m2 * t_values)) / y_values

        return m_t_values


    def calculate_central_and_error(c1s, c2s, m1s, m2s, T_start, T_end):
        num_samples = len(c1s)
        all_m_t_values = np.zeros((num_samples, 1200))

        for i in range(num_samples):
            all_m_t_values[i, :] = calculate_m_t(c1s[i], c2s[i], m1s[i], m2s[i], T_start, T_end)

        central_values = np.median(all_m_t_values, axis=0)
        errors = np.std(all_m_t_values, axis=0)

        return central_values, errors

    c1s=mb_mr_data['Zwp'][f'ensemble{nensemble}'][ii][jj][kk]
    m1s=mb_mr_data['mPS'][f'ensemble{nensemble}'][ii][jj][kk]*alttc/fm2GeV
    # print(m1s)
    if not two_state_enabled:
        c2s=np.zeros(4000)
        m2s=np.zeros(4000)
    else:
        c2s=mb_mr_data['c2'][f'ensemble{nensemble}'][ii][jj][kk]
        m2s=mb_mr_data['mp2'][f'ensemble{nensemble}'][ii][jj][kk]*alttc/fm2GeV

    central_values, errors = calculate_central_and_error(c1s, c2s, m1s, m2s, T_strt+0.5, T_end+0.5)
    all_central_values, all_errors = calculate_central_and_error(c1s, c2s, m1s, m2s, 0, T_hlf)

    # print("Central Values:", central_values)
    # print("Errors:", errors)
    print("mp:",gv.gvar(central_values[-1],errors[-1])*fm2GeV/alttc,"GeV")

    # exit()

    t_lst=t_ary[T_strt:T_end]
    meff_fitted_cntrl=central_values/alttc*fm2GeV
    meff_fitted_err=errors/alttc*fm2GeV
    all_meff_fitted_cntrl=all_central_values/alttc*fm2GeV
    all_meff_fitted_err=all_errors/alttc*fm2GeV
    # print(meff_fitted_cntrl)
    # print(meff_fitted_err)
    # print(t_ary)
    # exit()
    xshft = 0.3
    meff_data={}
    meff_data['data_range']=t_ary
    meff_data['avg_cntrl']= m_eff_cosh_avg_cntrl
    meff_data['avg_err']=m_eff_cosh_avg_err
    meff_data['fit_range']=t_lst
    meff_data['fitted_cntrl']= meff_fitted_cntrl
    meff_data['fitted_err']= meff_fitted_err
    if not os.path.exists('./temp/dict'):
        os.makedirs('./temp/dict')
    np.save("./temp/dict/"+sys.argv[1]+"_"+baryon+"_m_eff.npy",meff_data)
    # meff_y_lim_strt,meff_y_lim_end=read_ylimit(nensemble,ii,"./json/dimensionless/meff_plot_limit.json")
    # create_plot(0.8/alttc,np.linspace(T_strt+0.5, T_end+0.5, 1200),x=np.array(range(1, T_hlf)),
    if not two_state_enabled:
        meff_plot_cntrl=m_eff_cosh_avg_cntrl
        meff_plot_err=m_eff_cosh_avg_err
        label1="frwrd/bckwrd avg. cosh mass"
    else:
        meff_plot_cntrl=m_eff_log_avg_cntrl
        meff_plot_err=m_eff_log_avg_err
        label1="frwrd/bckwrd avg. log mass"
    print("about to plot meff")
    # print(meff_fitted_cntrl)
    # print(m_eff_log_avg_cntrl)
    # print(m_eff_cosh_avg_cntrl)
    create_plot_ex(0.8/alttc,np.linspace(T_strt, T_end, 1200),np.linspace(0-0.5, T_hlf-0.5, 1200),x=np.array(range(1, T_hlf)),
                y=meff_plot_cntrl,
                y_err=meff_plot_err,
                # y=m_eff_log_avg_cntrl,
                # y_err=m_eff_log_avg_err,
                # y_lim_strt=meff_y_lim_strt,
                # y_lim_end=meff_y_lim_end,
                y_lim_strt=0,
                y_lim_end=0,
                # y_lim_strt=mPS.mean-2,
                # y_lim_end=mPS.mean+2,
                xlabel=r'$t/a$',
                ylabel=r'$m_{\mathrm{eff}}(GeV)$',
                label1=label1,
                label2="best fit",
                prefix=name[nensemble]+"_"+prefix+"_"+baryon+"_m_eff",
                T_hlf=T_hlf,
                fit_fcn=meff_fitted_cntrl,
                fit_fcn_err=meff_fitted_err,
                all_fit_fcn=all_meff_fitted_cntrl,
                all_fit_fcn_err=all_meff_fitted_err)

