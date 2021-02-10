# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:12:27 2019

@author: Portatil
"""

# basic import
import os
import os.path as op
import sys
import subprocess as sp
sys.path.insert(0, op.join(op.dirname(__file__),'..','..','..'))


# aux. func. for launching bash command
def bash_cmd(str_cmd, out_file=None, err_file=None):
    'Launch bash command using subprocess library'
    _stdout = None
    _stderr = None
    if out_file:
        _stdout = open(out_file, 'w')
    if err_file:
        _stderr = open(err_file, 'w')

    s = sp.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
    s.wait()
        
    if out_file:
        _stdout.flush()
        _stdout.close()
    if err_file:
        _stderr.flush()
        _stderr.close()

# get sorted execution folders
# p_sea = r'/media/administrador/DiscoHD/Kwajalein/swan/run_cases_RCM/sea_nowind/01_stat/cases'
# p_swells = r'/home/administrador/Escritorio/HySWASH_Kwaj/run_cases/Swells/01_stat/cases'
# p_sea=r'/media/hd/teslakit/data/sites/GUAM/HYSWAN/sim/swan_projects/sea/cases'

p_swell=r'/media/administrador/HD/Dropbox/Guam/teslakit/data/sites/GUAM/HYSWAN/sim/swan_projects/swell/cases'
# p_res = r'/media/hd/teslakit/teslakit/numerical_models/swan/resources/swan_bin'
p_res=r'/media/administrador/HD/Dropbox/Guam/teslakit/teslakit/numerical_models/swan/resources/swan_bin'
sbin = op.abspath(op.join(p_res,'swan_ser.exe'))

run_dirs = sorted(os.listdir(p_swell))

# sys.exit()

for p_run in run_dirs[400:500]:
    
    # run case
    path = op.join(p_swell, p_run)
    print(p_run)
    
    # os.chdir(path)
    # os.system('chmod +rx ./swanrun')
    # os.system('./swanrun -input input')
    
    # log
    # ln input file and run swan case
    cmd = 'cd {0} && ln -sf input.swn INPUT && {1} INPUT'.format(
            path, sbin)
    bash_cmd(cmd)
        
    # ln input file nested grid and run swan case
    cmdn = 'cd {0} && ln -sf input_nest1.swn INPUT && {1} INPUT'.format(
            path, sbin)
    bash_cmd(cmdn)
        
        
    p = op.basename(p_run)
    print('SWAN CASE: {0} SOLVED'.format(p))
