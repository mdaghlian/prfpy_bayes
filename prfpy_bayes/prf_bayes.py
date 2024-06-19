from prfpy_csenf.model import *

import numpy as np
from scipy import stats 
import emcee
from .utils import *

from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.prfpy_ts_plotter import *

class BayesPRF(TSPlotter):
    ''' BayesPRF 
    A wrapper object meant to sit on top of prfpy models (https://github.com/VU-Cog-Sci/prfpy/tree/main/prfpy)
    Basically it sets up all the important elements needed for PRF fitting using MCMC (emcee toolbox)

    Functions:
        add_prior               Add a prior for each parameter. Easiest thing is just to use bounds (as you would do in standard prfpy, see function below) 
        add_prior_from_bounds   Automatically add uniform priors, based on bounds. Also used to 'hide' fixed parameters from the model
        prep_info               Based on the priors added, set everything up... 


    designed by Marcus Daghlian
    '''

    def __init__(self, prf_params, model, prfpy_model, real_ts,  **kwargs):
        super().__init__(prf_params, model=model, prfpy_model=prfpy_model, real_ts=real_ts, **kwargs)
        ''' __init__
        Set up the object; with important info
        '''
        # Setup and save model information:
        self.n_params = len(self.model_labels)
        if self.incl_rsq:
            self.n_params -= 1
        self.model_labels_inv = {value: key for key, value in self.model_labels.items()}
        
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
        self.walker_start = [None] * self.n_vox             # Where to start the walkers...
        self.sampler = [None] * self.n_vox                  # The sampler object for each voxel

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
            if v_p=='rsq':
                continue
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
        self.fit_p_list = [i for i in self.fit_p_list if i!='rsq']
        self.init_p_id = {}                                             # Map from fitted parameter name to index (so we don't get lost)
        for i_p,p in enumerate(self.fit_p_list):
            self.init_p_id[p] = i_p        
        

    def initialise_walkers(self, ivx, n_walkers, **kwargs):
        ''' initialise_walkers for a given voxel
        For a certain number of walkers setup a
        '''
        if self.init_walker_method == "random_prior":
            # Create walkers from random prior
            walker_start = self.init_walker_random_prior(n_walkers=n_walkers)
        elif self.init_walker_method == "gauss_ball":
            walker_start = self.init_walker_gauss_ball(n_walkers=n_walkers, ivx=ivx, **kwargs)
        elif self.init_walker_method == "fixed":
            walker_start = self.init_walker_fixed(n_walkers=n_walkers, ivx=ivx)
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
    
    def init_walker_gauss_ball(self, n_walkers, ivx, params_in=None, eps=None):
        '''
        Start the walkers as a little gaussian ball around a specified point in parameter space. Essentially adding jitter.
        - i.e., the grid fit
        Or use default: self.init_p
        Automatically resizes them to be the fit parameters...
        '''
        if params_in is not None:
            # Specified params_in
            if params_in.shape[0]==self.n_params:
                p2fit_in = self.params_full2fit(params_in)
            elif params_in.shape[0]==self.n_params2fit:
                p2fit_in = params_in
        else:
            params_in = self.prf_params_np[ivx, :-1]
            p2fit_in = self.params_full2fit(params_in)
        if eps is None:
            print('blooooooop')
            eps = self.gauss_ball_jitter
        walker_start = p2fit_in + eps * np.random.randn(n_walkers, self.n_params2fit)
        return walker_start          
    
    def init_walker_fixed(self, n_walkers, ivx, params_in=None):
        '''
        Start the walkers all on an identical specified point in parameter space 
        can use the starting params for this voxel
        Or can specify here (params_in)        
        Automatically resizes them to be the fit parameters...
        '''
        if params_in is not None:
            # Specified params_in
            if params_in.shape[0]==self.n_params:
                p2fit_in = self.params_full2fit(params_in)
            elif params_in.shape[0]==self.n_params2fit:
                p2fit_in = params_in
        else:
            params_in = self.prf_params_np[ivx, :-1]
            p2fit_in = self.params_full2fit(params_in)
        walker_start = p2fit_in + np.zeros(n_walkers, self.n_params2fit)
        return walker_start
        

    def params_full2fit(self, params):
        '''
        Takes array of all parameters (length self.n_params) and removes the
        fixed parameters. 
        '''
        print(params)
        print(params.shape)
        print(self.n_params)
        assert len(params)==self.n_params
        # Make a new array for parameters
        new_params = np.zeros(self.n_params2fit)
        # Instert the fitted parameters
        for p in self.fit_p_list:
            print(p)
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

    def fit_voxel(self, ivx, n_walkers, n_steps, **kwargs):
        pool            = kwargs.pop('pool', None)
        kwargs_sampler  = kwargs.get('kwargs_sampler', {})
        kwargs_run      = kwargs.get('kwargs_run', {})
        walkers = self.initialise_walkers(ivx=ivx, n_walkers=n_walkers, **kwargs)
        sampler = emcee.EnsembleSampler(
            len(walkers), 
            self.n_params2fit, 
            self.ln_posterior, 
            args=(self.real_ts[ivx,:],),
            pool=pool,
            **kwargs_sampler,
            )
        sampler.run_mcmc(walkers, n_steps, **kwargs_run)
        # Return the chain, log_prob, 
        chain = sampler.get_chain(discard=0, flat=False)
        flat_params = chain.reshape(-1, self.n_params2fit)
        
        logprob = sampler.get_log_prob(discard=0, flat=False)
        rsq = logprob2rsq(logprob, self.real_ts[ivx,:])   
        flat_rsq = rsq.reshape(-1, 1)     

        walker_id, step_id = np.meshgrid(np.arange(n_walkers), np.arange(n_steps))
        walker_id = walker_id.flatten()
        step_id = step_id.flatten()

        # make them full params
        full_params = np.zeros((flat_params.shape[0], self.n_params))
        for i_p, p in enumerate(flat_params):
            full_params[i_p] = self.params_fit2full(p)
        full_params[:,-1] = np.squeeze(flat_rsq)
        # Now make a PRF object
        self.sampler[ivx] = TSPlotter(
            prf_params=full_params,
            model=self.model,
            prfpy_model=self.prfpy_model,
            real_ts=np.repeat(self.real_ts[ivx,:][np.newaxis,...], full_params.shape[0], axis=0),
        )
        self.sampler[ivx].pd_params['walker_id'] = walker_id
        self.sampler[ivx].pd_params['step_id'] = step_id




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