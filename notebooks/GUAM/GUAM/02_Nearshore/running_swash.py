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

p_folder=r'/media/administrador/HD/Dropbox/Guam/teslakit/data/sites/GUAM/HYSWASH/projects/Guam_prf_10'

p_res = r'/media/administrador/HD/Dropbox/Guam/wrapswash-1d/lib/resources/swash_bin'

sbin = op.abspath(op.join(p_res,'swash.exe'))

run_dirs = sorted(os.listdir(p_folder))

# sys.exit()

for p_run in run_dirs[330:331]:
    
    # run case
    path = op.join(p_folder, p_run)
    print(p_run)
    
    # log
    # ln input file and run swan case
    cmd = 'cd {0} && ln -sf input.sws INPUT && {1} INPUT'.format(
            path, sbin)
    bash_cmd(cmd)
        
    p = op.basename(p_run)
    print('SWASH CASE: {0} SOLVED'.format(p))
