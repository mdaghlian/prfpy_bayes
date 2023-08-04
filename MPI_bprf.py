#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V
        
import numpy as np
import matplotlib.pyplot as plt
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel, DoG_Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter
from prfpy.rf import gauss2D_iso_cart
import os
import yaml

from prfpy_bayes.utils import *
from prfpy_bayes.prf_bayes import *

from dag_prf_utils.prfpy_functions import *
from scipy import stats 
import time

from schwimmbad import MPIPool

# setup e.g. data and ts...
# Getting the correct stimuli
from pfa_scripts.load_saved_info import *
sub = 'sub-03'
model = 'gauss'

n_walkers = 500
n_steps = 10
ivx = 0 

model_list = ['gauss']
if model != 'gauss':
    model_list += model

prf_data = load_data_prf(sub, ['AS0'], model_list)['AS0']
prfs = Prf1T1M(prf_data['gauss'], 'gauss')
real_tc = load_data_tc(sub, 'AS0')['AS0']
idx = np.where(prfs.return_vx_mask({'min-rsq':.5, 'max-ecc':5, 'max-size_1':5}))[0][:100]
real_tc = real_tc[idx,:]
for model in model_list:
    prf_data[model] = prf_data[model][idx,:]    


# load the settings
prf_settings = './fit_settings_prf_SMALL.yml'
with open(prf_settings) as f:
    prf_info = yaml.safe_load(f)
prf_info = load_data_prf(sub, ['AS0'], 'gauss', var_to_load='settings')['AS0']['gauss']
# load the stimulus
prf_stim = get_prfpy_stim(sub, 'AS0')['AS0']

# 
prfpy_model = {}
prfpy_model['gauss'] = Iso2DGaussianModel(
    stimulus=prf_stim,                                  
    hrf=prf_info['hrf']['pars'],                        
    # filter_predictions = prf_info['filter_predictions'],
    normalize_RFs= prf_info['normalize_RFs'],           
    )

prfpy_model['css'] = CSS_Iso2DGaussianModel(
    stimulus=prf_stim,                                  
    hrf=prf_info['hrf']['pars'],                        
    # filter_predictions = prf_info['filter_predictions'],
    normalize_RFs= prf_info['normalize_RFs'],           
    )

prfpy_model['dog'] = DoG_Iso2DGaussianModel(
    stimulus=prf_stim,                                  
    hrf=prf_info['hrf']['pars'],                        
    # filter_predictions = prf_info['filter_predictions'],
    normalize_RFs= prf_info['normalize_RFs'],           
    )

prfpy_model['norm'] = Norm_Iso2DGaussianModel(
    stimulus=prf_stim,                                  
    hrf=prf_info['hrf']['pars'],                        
    # filter_predictions = prf_info['filter_predictions'],
    normalize_RFs= prf_info['normalize_RFs'],           
    )
max_eccentricity = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region    
bounds = get_bounds(prf_info, max_eccentricity, model)

bprf = BayesPRF(model=model, prfpy_stim=prf_stim, normalize_RFs=prf_info['normalize_RFs'])
bprf.add_priors_from_bounds(bounds)
bprf.prep_info()


this_true_resp = real_tc[ivx,:]
this_cprf = prf_data[model][ivx,:-1]
walkers = bprf.sample_tiny_gauss_ball(this_cprf, n_walkers)

start_time = time.time()
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    np.random.seed(42)
    sampler = run_emcee_basic(
        n_walkers=n_walkers, 
        n_steps=n_steps, 
        true_resp=this_true_resp,
        bprf=bprf, 
        walkers=walkers,            
        pool=pool,
        # moves=emcee.moves.WalkMove()
        )

# Measure the end time
end_time = time.time()

# Calculate and print the time taken
total_time = end_time - start_time
print("Total time taken: {:.1f} seconds".format(total_time))    
print(f'Model {model}, walkers={n_walkers}, steps={n_steps}')
