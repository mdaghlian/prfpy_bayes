from prfpy_csenf.model import *

import numpy as np
from scipy import stats 
import emcee
from .utils import *

class BayesPRF():
    ''' BayesPRF 
    A wrapper object meant to sit on top of prfpy models (https://github.com/VU-Cog-Sci/prfpy/tree/main/prfpy)
    Basically it sets up all the important elements needed for PRF fitting using MCMC (emcee toolbox)

    Functions:
        add_prior               Add a prior for each parameter. Easiest thing is just to use bounds (as you would do in standard prfpy, see function below) 
        add_prior_from_bounds   Automatically add uniform priors, based on bounds. Also used to 'hide' fixed parameters from the model
        prep_info               Based on the priors added, set everything up... 


    designed by Marcus Daghlian
    '''

    def __init__(self, model, prfpy_stim=[], **kwargs):
        ''' __init__
        Set up the object; with important info
        '''
        # Setup and save model information:
        self.model = model              # which model being fit? ['gauss', 'css', 'dog', 'norm', 'csf']
        self.model_labels = prfpy_params_dict()[self.model]
        self.model_labels.pop('rsq')
        self.n_params = len(self.model_labels)
        self.model_labels_inv = {value: key for key, value in self.model_labels.items()}
        
        # Setup prfpy specific information
        self.prfpy_stim = prfpy_stim                                # Stimulus object from prfpy
        self.normalize_RFs = kwargs.get('normalize_RFs', False)     # Normalize prfs? (i.e., to stop confound b/w size and amplitude... generally not needed)
        if self.prfpy_stim !=[]:
            self.load_prfpy_model()                                 # Load specified prfpy models
        
        # MCMC specific information
        self.model_sampler = {}                                     # A sampler for each parameter (e.g., randomly pick "x" b/w -5 and 5)
        self.model_prior = {}                                       # Prior for each parameter (e.g., normal distribution at 0 for "x")
        self.model_prior_type = {}                                  # For each parameter: can be "uniform" (just bounds), or "normal"
        self.fixed_vals = {}                                        # We can fix some parameters. To speed things up, don't include them in the MCMC process
        self.bounds = {}
        self.init_walker_method = kwargs.get('init_walker_method', 'random_prior')  # How to setup the walkers? "random_prior", "gauss_ball" (see emcee docs) and 
                                                                                    # "initialise_walkers" below for more details       

        self.init_walker_ps = kwargs.get('init_walker_ps', np.zeros(self.n_params)) # if using "gauss_ball" to initialize the walkers, this is the jitter point        
        self.gauss_ball_jitter = kwargs.get('gauss_ball_jitter', 1e-4)              # How much to jitter each parameter by
        self.save_like = kwargs.get('save_like', True)      # save likelihood for each MCMC step as well as the log_prob (see blobs in emcee)        

    def add_prior(self, pid, **kwargs):
        ''' 
        Adds the prior to each parameter:
        Used for 
        [1] evaluating the posterior (e.g., to enforce parameter bounds)
        [2] Initialising walker positions (if init_walker_method=='random_prior')
        > randomly sample parameters from the prior distributions

        Options:
            fixed:      will 'hide' the parameter from MCMC fitting procedure (not really a prior...)
            uniform:    uniform probability b/w the specified bounds (vmin, vmax). Otherwise infinite
            normal:     normal probability. (loc, scale)
            none:       The parameter can still vary, but it will not influence the outcome... 
        '''        
        if pid not in self.model_labels.keys(): # Is this a valid parameter to add? 
            print('error...')
            return
        prior_type = kwargs.get('prior_type')   # Which prior? uniform, fixed, normal
        self.model_prior_type[pid] = prior_type 
        if prior_type=='normal':
            # Get loc, and scale
            loc = kwargs.get('loc')    
            scale = kwargs.get('scale')    
            self.model_sampler[pid] = PriorNorm(loc, scale).sampler 
            self.model_prior[pid]   = PriorNorm(loc, scale).prior            

        elif prior_type=='uniform' :
            vmin = kwargs.get('vmin')
            vmax = kwargs.get('vmax')
            self.bounds[pid] = [vmin, vmax]
            self.model_sampler[pid] = PriorUniform(vmin, vmax).sampler
            self.model_prior[pid] = PriorUniform(vmin, vmax).prior            

        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.fixed_vals[pid] = fixed_val
            self.model_prior[pid] = PriorFixed(fixed_val).prior

        elif prior_type=='none':
            self.model_prior[pid] = PriorNone().prior            
    
    def add_priors_from_bounds(self, bounds):
        '''
        Used to setup uninformative priors: i.e., uniform between the bouds
        Can setup more informative, like a normal using the other methods
        '''
        for i_p, v_p in enumerate(self.model_labels.keys()):
            if bounds[i_p][0]!=bounds[i_p][1]: 
                self.add_prior(
                    pid=v_p,
                    prior_type = 'uniform',
                    vmin = bounds[i_p][0],
                    vmax = bounds[i_p][1],
                    )
            else: # If upper & lower bound are the same, make it a fixed parameter
                self.add_prior(
                    pid=v_p,
                    prior_type = 'fixed',
                    fixed_val = bounds[i_p][0],
                    )
                

    def prep_info(self):
        ''' prep_info
        Set up object for fitting...
        i.e., check which parameters are fixed etc.
        '''
        self.n_params2fit = self.n_params - len(self.fixed_vals)        # Number of parameters to fit (i.e., not the ones being fixed)
        self.all_p_list = list(self.model_labels.keys())                # List of *all* parameters names
        self.fix_p_list = list(self.fixed_vals.keys())                  # List of *fixed* parameter names 
        self.fit_p_list = list(
            x for x in self.all_p_list if x not in self.fix_p_list)     # List of *fitted* parameter names 
        self.init_p_id = {}                                             # Map from fitted parameter name to index (so we don't get lost)
        for i_p,p in enumerate(self.fit_p_list):
            self.init_p_id[p] = i_p
        
        # If we are initialising walkers using the "small gauss ball" approach (i.e., jittering around a fixed point)
        # we only want to include the fitted parameters
        if len(self.init_walker_ps)!=self.n_params2fit:
            self.init_walker_ps = self.params_full2fit(self.init_walker_ps)
        

    def initialise_walkers(self, n_walkers, **kwargs):
        ''' initialise_walkers
        For a certain number of walkers setup a
        '''
        if self.init_walker_method == "random_prior":
            # Create walkers from random prior
            walker_start = self.init_walker_random_prior(n_walkers)
        if self.init_walker_method == "gauss_ball":
            walker_start = self.init_walker_gauss_ball(n_walkers, **kwargs)

        return walker_start


    def init_walker_random_prior(self, n_walkers):
        # Only initialise params to fit...
        walker_start = []
        for iw in range(n_walkers):
            params = np.zeros(self.n_params2fit)
            for i_p,p in enumerate(self.fit_p_list):
                params[i_p] = self.model_sampler[p](1)
            walker_start.append(np.array(params))
        return walker_start
    
    def init_walker_gauss_ball(self, n_walkers, params_in=None, eps=None):
        '''
        Start the walkers as a little gaussian ball around a specified point in parameter space. Essentially adding jitter.
        Can specify here (params_in)
        Or use default: self.init_p
        Automatically resizes them to be the fit parameters...
        '''
        if params_in is not None:
            # Specified params_in
            if params_in.shape[0]==self.n_params2fit:
                p2fit_in = self.params_full2fit(params_in)
            else:
                p2fit_in = params_in
        else:
            p2fit_in = self.init_walker_ps
        if eps is None:
            eps = self.gauss_ball_jitter
        walker_start = p2fit_in + eps * np.random.randn(n_walkers, self.n_params2fit)
        return walker_start        
        
    def load_prfpy_model(self):
        if self.model=='gauss':
            self.prfpy_model = Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='css':
            self.prfpy_model = CSS_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='dog':
            self.prfpy_model = DoG_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='norm':
            self.prfpy_model = Norm_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='csf':
            self.prfpy_model = CSenFModel(stimulus=self.prfpy_stim )

    def params_full2fit(self, params):
        '''
        Takes array of all parameters (length self.n_params) and removes the
        fixed parameters. 
        '''
        assert len(params)==self.n_params
        # Make a new array for parameters
        new_params = np.zeros(self.n_params2fit)
        # Instert the fitted parameters
        for p in self.fit_p_list:
            fit_p_id = self.init_p_id[p]
            full_p_id = self.model_labels[p]
            new_params[fit_p_id] = params[full_p_id]
        return new_params

    def params_fit2full(self, params):
        '''
        Takes array of parameters being fit (length self.n_params2fit)
        And adds in the fixed parameters, so that the full list of parameters 
        can be passed to the prfpy_model
        '''
        assert len(params)==self.n_params2fit
        # Make a new array for parameters
        new_params = np.zeros(self.n_params)
        # Instert the fitted parameters
        for p in self.fit_p_list:
            fit_p_id = self.init_p_id[p]
            full_p_id = self.model_labels[p]
            new_params[full_p_id] = params[fit_p_id]
        # Insert fixed params:
        for v_p in self.fixed_vals.keys():
            i_p = self.model_labels[v_p]        
            new_params[i_p] = self.fixed_vals[v_p]
        return new_params

    def prfpy_model_wrapper(self, params):
        new_params = self.params_fit2full(params)
        pred = np.nan_to_num(np.squeeze(self.prfpy_model.return_prediction(*list(np.array(new_params)))))
        return pred        

    def ln_likelihood(self, params, response):
        model_response = self.prfpy_model_wrapper(params)
        return -0.5 * np.sum((response - model_response)**2)

    # Log-prior function for the model
    def ln_prior(self, params):
        p_out = 0.0
        for v_p in self.fit_p_list:
            i_p = self.init_p_id[v_p]
            p_out += self.model_prior[v_p](params[i_p])
        return p_out    

    # Log-posterior function for the model
    # -> technically this is the joint distribution as we are not normalizing
    # by the marginal_likelihood... But this doesn't really apply here...     
    def ln_posterior(self, params, response):
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params, response)
        if self.save_like:        
            return prior + like, like # save both...
        else:
            return prior


def run_emcee_pervox(n_walkers, n_steps, true_resp, bprf, walkers):
    sampler = run_emcee_basic(
        n_walkers=n_walkers, 
        n_steps=n_steps, 
        true_resp=true_resp,
        bprf=bprf, 
        walkers=walkers,)
    sampler_info = {
        'chain': sampler.get_chain(discard=0, flat=False),
        'log_prob': sampler.get_log_prob(discard=0, flat=False),
        'acceptance_fraction': sampler.acceptance_fraction,
        # 'acor': sampler.get_autocorr_time(),
        # Add any other information you want to store
    }    
    return sampler_info    

def process_ivx(ivx, this_tc_data, bprf, n_walkers, n_steps, **kwargs):
    # np.random.seed(ivx)
    # walkers = bprf[model].sample_tiny_gauss_ball(this_cprf, n_walkers)    
    walkers = bprf.initialise_walkers(n_walkers)    
    sampler = run_emcee_basic(
        n_walkers=n_walkers, 
        n_steps=n_steps, 
        true_resp=this_tc_data,
        bprf=bprf, 
        walkers=walkers,            
        )
    discard = n_steps//10
    # Save memory... only keep vx>0.1
    # this_sample = {
    #     'chain': sampler.get_chain(discard=discard, flat=True),
    #     'log_prob': sampler.get_log_prob(discard=discard, flat=True),
    # }

    # this_sample['rsq'] = logprob2rsq(this_sample['log_prob'], this_tc_data)

    # this_sample['chain']    = this_sample['chain'][this_sample['rsq']>0.1,:]  
    # this_sample['log_prob'] = this_sample['log_prob'][this_sample['rsq']>0.1] 
    # this_sample['rsq']      = this_sample['rsq'][this_sample['rsq']>0.1]  

    
    this_sample = {
        'chain': sampler.get_chain(discard=0, flat=False),
        'log_prob': sampler.get_log_prob(discard=0, flat=False),
        'acceptance_fraction': sampler.acceptance_fraction,        
        # 'acor': sampler.get_autocorr_time(),
        # Add any other information you want to store
    }
    this_sample['rsq'] = logprob2rsq(this_sample['log_prob'], this_tc_data)
    # this_sample['full_flat'] = mcmc_out_to_full_flat(
    #     mcmc_out=sampler, 
    #     this_tc_data=this_tc_data, 
    #     bprf=bprf, 
    #     burn_in=0)
    return this_sample

def mcmc_out_to_full_flat(sample_out, bprf, burn_in=0, max_step=None):
    '''
    Take the output of the bprf fitting
    * Remove the burn in
    * Flatten the chains
    * Add in the fixed parameters
    * plus the rsquared 
    '''

    flat_out = []
    for ivx in range(len(sample_out)):
        if len(sample_out[ivx]['chain'].shape)>2:
            chain = sample_out[ivx]['chain'][burn_in:max_step,:,:]
            chain_shape = chain.shape
            part_flat = chain.reshape(
                chain_shape[0]*chain_shape[1], chain_shape[-1]
            )
        else:
            part_flat = sample_out[ivx]['chain'].copy()

        assert part_flat.shape[-1]==bprf.n_params2fit
        # Make a new array for parameters
        full_flat = np.zeros((part_flat.shape[0], bprf.n_params+1)) # plus 1 for rsq
        # Instert the fitted parameters
        for p in bprf.fit_p_list:
            fit_p_id = bprf.init_p_id[p]
            full_p_id = bprf.model_labels[p]
            full_flat[:,full_p_id] = part_flat[:,fit_p_id]
        # Insert fixed params:
        for v_p in bprf.fixed_vals.keys():
            i_p = bprf.model_labels[v_p]        
            full_flat[:,i_p] = bprf.fixed_vals[v_p]
        
        # add the rsq
        if len(sample_out[ivx]['chain'].shape)>2:
            flat_rsq = sample_out[ivx]['rsq'][burn_in:max_step,:].flatten()
        else:
            flat_rsq = sample_out[ivx]['rsq']
        full_flat[:,-1] = flat_rsq
        flat_out.append(full_flat)
    return flat_out



# def mcmc_out_to_full_flat(mcmc_out, this_tc_data, bprf, burn_in=0):
#     '''
#     Take the output of the bprf fitting
#     * Remove the burn in
#     * Flatten the chains
#     * Add in the fixed parameters
#     * plus the rsquared 
#     '''
#     part_flat = mcmc_out.get_chain(discard=burn_in, flat=True)
#     assert part_flat.shape[-1]==bprf.n_params2fit
#     # Make a new array for parameters
#     full_flat = np.zeros((part_flat.shape[0], bprf.n_params+1)) # plus 1 for rsq
#     # Instert the fitted parameters
#     for p in bprf.fit_p_list:
#         fit_p_id = bprf.init_p_id[p]
#         full_p_id = bprf.model_labels[p]
#         full_flat[:,full_p_id] = part_flat[:,fit_p_id]
#     # Insert fixed params:
#     for v_p in bprf.fixed_vals.keys():
#         i_p = bprf.model_labels[v_p]        
#         full_flat[:,i_p] = bprf.fixed_vals[v_p]
    
#     # add the rsq
#     full_flat[:,-1] = logprob2rsq(mcmc_out.get_log_prob(discard=burn_in, flat=True), this_tc_data)

#     return full_flat

def run_emcee_basic(n_walkers, n_steps, true_resp, bprf, pool=None, walkers=None, kwargs_sampler={}, kwargs_run={}):

    if walkers is None:
        walkers = bprf.initialise_walkers(n_walkers=n_walkers)
    sampler = emcee.EnsembleSampler(
        len(walkers), 
        bprf.n_params2fit, 
        bprf.ln_posterior, 
        args=(true_resp,),
        pool=pool,
        **kwargs_sampler,
        # moves=moves,        
        )
    sampler.run_mcmc(walkers, n_steps, **kwargs_run)
    return sampler

# *** PRIORS ***
class PriorNorm():    
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def sampler(self, n_samples):
        return np.random.normal(self.loc, self.scale, n_samples)
    def prior(self, p):
        return stats.norm.logpdf(p, self.loc, self.scale)

class PriorUniform():
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
    def sampler(self, n_samples):
        return np.random.uniform(self.vmin, self.vmax, n_samples)
    def prior(self, p):
        return 0 if self.vmin <= p <= self.vmax else -np.inf

class PriorFixed():
    def __init__(self, fixed_val):
        self.fixed_val = fixed_val
    def prior(self, p):
        return p*0.0

class PriorNone():
    def __init__(self):
        self.bounds = 'None'        
    def prior(self,p):
        return p*0.0



# ********
# 
from dag_prf_utils.prfpy_functions import *
# def bprf_data2bobj_OLD(bprf_data, bprf):
#     flat_sample = mcmc_out_to_full_flat(bprf_data, bprf)
#     # Get best vx
#     bbest = []
#     for i in flat_sample:    
#         if len(i)==0:
#             bbest.append(np.ones(bprf.n_params+1)*np.nan)
#         else:
#             ibest = np.argmax(i[:,-1])

#             bbest.append(i[ibest])    
#     bbest = np.vstack(bbest)      # best parameters...
#     bobj = Prf1T1M(bbest, bprf.model)
#     # Add quartiles, median and uncertainty  (for fit parameters)
#     p25 = np.zeros((bobj.n_vox, bprf.n_params+1))
#     p50 = np.zeros((bobj.n_vox, bprf.n_params+1))
#     p75 = np.zeros((bobj.n_vox, bprf.n_params+1))
#     n_post = np.zeros(bobj.n_vox) # number of samples
#     for vx_id,this_vx in enumerate(flat_sample):
#         n_post[vx_id] = len(this_vx)
#         if n_post[vx_id]<100: # ==0
#             n_post[vx_id] = np.nan
#             p25[vx_id,:] = np.nan
#             p50[vx_id,:] = np.nan
#             p75[vx_id,:] = np.nan
#         else:
#             p25[vx_id,:] = np.percentile(this_vx, 25, axis=0)
#             p50[vx_id,:] = np.percentile(this_vx, 50, axis=0)
#             p75[vx_id,:] = np.percentile(this_vx, 75, axis=0)
#     pUNC = p75 - p25
#     bobj.pd_params[f'n_post'] = n_post
#     for p in bobj.model_labels.keys():
#         pidx = bobj.model_labels[p]
#         bobj.pd_params[f'{p}_25'] = p25[:,pidx]
#         bobj.pd_params[f'{p}_50'] = p50[:,pidx]
#         bobj.pd_params[f'{p}_75'] = p75[:,pidx]
#         bobj.pd_params[f'{p}_UNC'] = pUNC[:,pidx]
#     nan_loc = np.isnan(n_post)
#     bobj.pd_params.loc[nan_loc] = np.nan

#     return bobj        


def bprf_data2bobj(bprf_data, bprf):
    # PD VERSION
    flat_sample = mcmc_out_to_full_flat(bprf_data, bprf)
    # Get best vx
    bbest = []
    for i in flat_sample:    
        if len(i)==0:
            bbest.append(np.ones(bprf.n_params+1)*np.nan)
        else:
            ibest = np.argmax(i[:,-1])

            bbest.append(i[ibest])    
    bbest = np.vstack(bbest)      # best parameters...
    bobj = Prf1T1M(bbest, bprf.model)
    # Add quartiles, median and uncertainty  (for fit parameters)
    # PD VERSION (get everything...)            
    quantiles = [0.25, 0.50, 0.75]
    n_vox = bobj.pd_params.shape[0]
    n_col = bobj.pd_params.shape[1]

    # pd_sum = 
    p25_cols = [f'{col}_25' for col in bobj.pd_params.columns]
    p50_cols = [f'{col}_50' for col in bobj.pd_params.columns]
    p75_cols = [f'{col}_75' for col in bobj.pd_params.columns]
    # df for each quartile
    pX_df = {col : np.zeros(n_vox)*np.nan for col in bobj.pd_params.columns}
    pX_df = pd.DataFrame(pX_df)
    p25_df = pX_df.copy()
    p50_df = pX_df.copy()
    p75_df = pX_df.copy()

    n_post = np.zeros(n_vox) * np.nan
    for vx_id, this_vx in enumerate(flat_sample):
        if len(this_vx)>=100:
            n_post[vx_id] = len(this_vx)
            this_vx_obj = Prf1T1M(this_vx, bprf.model)
            p25_df.loc[vx_id] = this_vx_obj.pd_params.quantile(q=0.25)
            p50_df.loc[vx_id] = this_vx_obj.pd_params.quantile(q=0.50)
            p75_df.loc[vx_id] = this_vx_obj.pd_params.quantile(q=0.75)

    # Create uncertainty
    pUNC_df = p75_df - p25_df
    # Rename parameters
    p25_df = p25_df.add_suffix('_25')
    p50_df = p50_df.add_suffix('_50')
    p75_df = p75_df.add_suffix('_75')
    pUNC_df = pUNC_df.add_suffix('_UNC')
    # Add them to bobj.pd_params
    bobj.pd_params = pd.concat(
        [bobj.pd_params,p25_df, p50_df, p75_df, pUNC_df], axis=1)
    bobj.pd_params['n_post'] = n_post
    nan_loc = np.isnan(n_post)
    bobj.pd_params.loc[nan_loc] = np.nan
    return bobj    