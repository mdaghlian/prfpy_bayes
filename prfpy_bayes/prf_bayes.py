try:
    from prfpy.model import *
except:
    from prfpy_csenf.model import *

import numpy as np
from scipy import stats 
import emcee
from .utils import *

from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.prfpy_ts_plotter import *


prfpy_global_model = PrfpyModelGlobal()
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
        ''' __init__
        Set up the object; with important info
        '''
        self.kwargs = kwargs
        # If no prf_params are passed, make an empty array 
        model_labels = prfpy_params_dict()[model] # Get names for different model parameters...
        n_params = len(model_labels)
        if prf_params is None: 
            prf_params = np.zeros((real_ts.shape[0], n_params))
        
        # Call the parent class
        super().__init__(prf_params, model=model, prfpy_model=prfpy_model, real_ts=real_ts, **kwargs)
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
        self.init_walker_method = kwargs.get('init_walker_method', 'gauss_ball')   # How to setup the walkers? "random_prior", "gauss_ball" (see emcee docs)
        # How to estimate the offset and slope for time series. "glm" or "mcmc" (is it just another MCMC parameter, or use glm )                
        # NOT IMPLEMENTED (YET)
        self.amp_method = kwargs.get('amp_method', 'glm')        
        self.gauss_ball_jitter = kwargs.get('gauss_ball_jitter', 1e-4)              # How much to jitter each parameter by
        self.sampler = [None] * self.n_vox                  # The sampler object for each voxel

    def add_prior(self, pid, prior_type, **kwargs):
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
            self.bounds[pid] = [fixed_val, fixed_val]
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

            if bounds[v_p][0]!=bounds[v_p][1]: 
                self.add_prior(
                    pid=v_p,
                    prior_type = 'uniform',
                    vmin = bounds[v_p][0],
                    vmax = bounds[v_p][1],
                    )
            else: # If upper & lower bound are the same, make it a fixed parameter
                self.add_prior(
                    pid=v_p,
                    prior_type = 'fixed',
                    fixed_val = bounds[v_p][0],
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
        

    def initialise_walkers(self, idx, n_walkers, **kwargs):
        ''' initialise_walkers for a given voxel
        '''
        init_walker_method = kwargs.get('init_walker_method', self.init_walker_method)
        initial_guess = kwargs.get('initial_guess', None) # What is our starting point
        if initial_guess is not None:
            kwargs['params_in'] = initial_guess # Set the initial guess
        ow_walkers = kwargs.get('walkers', None)
        if ow_walkers is not None:
            return ow_walkers
        if init_walker_method == "random_prior":
            # Create walkers from random prior
            walker_start = self.init_walker_random_prior(n_walkers=n_walkers)
        elif init_walker_method == "gauss_ball":
            walker_start = self.init_walker_gauss_ball(n_walkers=n_walkers, idx=idx, **kwargs)
        elif init_walker_method == "fixed":
            walker_start = self.init_walker_fixed(n_walkers=n_walkers, idx=idx, **kwargs)
        return walker_start

    def init_walker_random_prior(self, n_walkers):
        '''Create n_walkers 
        Randomly sample from the prior for each parameter
        '''
        # Only initialise params to fit...
        walker_start = []
        for iw in range(n_walkers):
            params = np.zeros(self.n_params2fit)
            for i_p,p in enumerate(self.fit_p_list):
                params[i_p] = self.model_sampler[p](1)
            walker_start.append(np.array(params))
        return walker_start
    
    def init_walker_gauss_ball(self, n_walkers, idx, **kwargs):
        '''
        Start the walkers as a little gaussian ball around a specified point in parameter space. Essentially adding jitter.
        - i.e., the grid fit
        Or use default: self.init_p
        Automatically resizes them to be the fit parameters...
        '''
        params_in = kwargs.get('params_in', None)
        eps = kwargs.get('eps', self.gauss_ball_jitter)        
        if params_in is not None:
            # Specified params_in
            if len(params_in)==self.n_params:
                p2fit_in = self.params_full2fit(params_in)
            elif len(params_in)==self.n_params2fit:
                p2fit_in = params_in
        else:
            # Take it from the initial best guess....
            params_in = self.prf_params_np[idx, :-1]
            p2fit_in = self.params_full2fit(params_in)

        walker_start = p2fit_in + eps * np.random.randn(n_walkers, self.n_params2fit)
        return walker_start          
    
    def init_walker_fixed(self, n_walkers, idx, **kwargs):
        '''
        Start the walkers all on an identical specified point in parameter space 
        can use the starting params for this voxel
        Or can specify here (params_in)        
        Automatically resizes them to be the fit parameters...
        '''
        params_in = kwargs.get('params_in', None)
        if params_in is not None:
            # Specified params_in
            if params_in.shape[0]==self.n_params:
                p2fit_in = self.params_full2fit(params_in)
            elif params_in.shape[0]==self.n_params2fit:
                p2fit_in = params_in
        else:
            params_in = self.prf_params_np[idx, :-1]
            p2fit_in = self.params_full2fit(params_in)
        walker_start = p2fit_in + np.zeros((n_walkers, self.n_params2fit))
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

    # Log-likelihood function for the model
    def ln_likelihood(self, params, response):
        ''' THIS IS DODGY - ASK REMCO
        Vaguely following:
        https://github.com/Joana-Carvalho/Micro-Probing/blob/master/computing_mcmc_tiny_ica.m
        '''
        # [1] Get the predicted time series
        model_response = self.prfpy_model_wrapper(params)        
        if np.all(model_response == 0):
            # For some reason 0
            return -np.inf        
        # [2] Calculate the residuals
        residuals = response - model_response

        # [3] Fit a normal distribution to the residuals
        # Assume mean of residuals is 0
        # muhat, sigmahat = stats.norm.fit(residuals)
        muhat = 0
        _, sigmahat = stats.norm.fit(residuals, floc=0) 

        # Even faster? If we know mean is 0, don't want to fit std, but calculate it?
        # sigmahat = np.std(residuals)           

        # Calculate the log likelihood of the residuals
        # given the fitted normal distribution
        # then add it up for all time points
        # SLOWER TO CALL OUT TO LIBRARIES: log_like = stats.norm.logpdf(residuals, muhat, sigmahat).sum()
        log_like = -0.5 * np.sum((residuals / sigmahat) ** 2 + np.log(2 * np.pi * sigmahat**2))

        return log_like

    # Log-prior function for the model
    def ln_prior(self, params):
        p_out = 0.0
        for v_p in self.fit_p_list:
            i_p = self.init_p_id[v_p]
            p_out += self.model_prior[v_p](params[i_p])
        return p_out    

    def ln_posterior(self, params, response):
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params, response)
        return prior + like, like # save both...

    def run_mcmc_fit(self, idx, n_walkers, n_steps, **kwargs):
        ''' run_mcmc_fit
        Run the MCMC fit for a given voxel
        '''
        pool            = kwargs.pop('pool', None)
        save_mode       = kwargs.pop('save_mode', 'minimal') # What to save in 'sampler': can be obj or minimal
        save_top_kpsc   = kwargs.pop('save_top_kpsc', 15) # Keep the top n % (in terms of model fit)
        save_min_rsq    = kwargs.pop('save_min_rsq', 0.1) # Only save if the rsq is above this
        burn_in         = kwargs.pop('burn_in', 0) # How many steps to burn in
        kwargs_sampler  = kwargs.get('kwargs_sampler', {})
        kwargs_run      = kwargs.get('kwargs_run', {})
        walkers = self.initialise_walkers(idx=idx, n_walkers=n_walkers, **kwargs)        
        # Run a test
        self.ln_prior(walkers[0])
        self.ln_likelihood(walkers[0], self.real_ts[idx,:])
        self.ln_posterior(walkers[0], self.real_ts[idx,:])
        # Ok - now we can run the sampler!!
        # Running in parallel?
        if pool is not None:
            print('Running in parallel')
            prfpy_global_model.set_model(self.prfpy_model)
            # Now make the fast one
            bpfast = BPFast(model=self.model, real_ts=self.real_ts[idx,:].copy(), **self.kwargs)
            # Add bounds
            bpfast.add_priors_from_bounds(self.bounds)
            bpfast.prep_info()
            # Run a quick test
            bpfast.ln_posterior(walkers[0])
            sampler = emcee.EnsembleSampler(
                nwalkers=len(walkers), 
                ndim=self.n_params2fit, 
                log_prob_fn=bpfast.ln_posterior, 
                pool=pool,
                **kwargs_sampler, # Any other arguments that you want to pass to the sampler
                ) 
        else:
            print('Running in serial')
            sampler = emcee.EnsembleSampler(
                nwalkers=len(walkers), 
                ndim=self.n_params2fit, 
                log_prob_fn=self.ln_posterior, 
                args=(self.real_ts[idx,:],),
                pool=pool,
                **kwargs_sampler,
                )
        sampler.run_mcmc(walkers, n_steps, **kwargs_run)
        # Return the chain, log_prob, 
        chain = sampler.get_chain(discard=burn_in, flat=False)
        flat_params = chain.reshape(-1, self.n_params2fit)        
        logprob = sampler.get_log_prob(discard=burn_in, flat=False)

        n_out = n_steps - burn_in
        walker_id, step_id = np.meshgrid(np.arange(n_walkers), np.arange(n_out)+burn_in)
        walker_id = walker_id.flatten()
        step_id = step_id.flatten()

        # make them full params
        full_params = np.zeros((flat_params.shape[0], self.n_params+1))
        for i_p, p in enumerate(flat_params):
            full_params[i_p,:-1] = self.params_fit2full(p)
        
        # Recalculate rsq 
        preds = self.prfpy_model.return_prediction(
            *list(np.array(full_params[:,:-1].T))
        )
        rsq = dag_get_rsq(tc_target=self.real_ts[idx,:], tc_fit=preds)        
        full_params[:,-1] = np.squeeze(rsq)
        id_to_keep = np.ones((len(rsq),), dtype=bool)
        if save_top_kpsc is not None:
            best_fits = np.argsort(rsq)[::-1][:int(save_top_kpsc/100 * len(rsq))]
            print(f'Keeping top {save_top_kpsc}% of fits: = {len(best_fits)} fits')
            id_to_keep = np.zeros((len(rsq),), dtype=bool)
            id_to_keep[best_fits] = True
        if save_min_rsq is not None:
            id_to_keep &= (rsq > save_min_rsq)
        # Get the best fits
        full_params = full_params[id_to_keep,:]
        logprob = logprob[id_to_keep]
        walker_id = walker_id[id_to_keep]
        step_id = step_id[id_to_keep]

        if save_mode=='obj':        
            # Now make a PRF object
            self.sampler[idx] = TSPlotter(
                prf_params=full_params,
                model=self.model,
                prfpy_model=self.prfpy_model,
                real_ts=np.repeat(self.real_ts[idx,:][np.newaxis,...], full_params.shape[0], axis=0),
            )
            self.sampler[idx].pd_params['walker_id'] = walker_id
            self.sampler[idx].pd_params['step_id'] = step_id
            self.sampler[idx].pd_params['logprob'] = logprob.flatten()
        elif save_mode=='minimal':
            # Save only the important stuff... (don't want it to be too big)
            self.sampler[idx] = {}
            for p in self.model_labels.keys():
                self.sampler[idx][p] = full_params[:,self.model_labels[p]]
            self.sampler[idx]['rsq'] = full_params[:,-1]
            self.sampler[idx]['walker_id'] = walker_id
            self.sampler[idx]['step_id'] = step_id
            self.sampler[idx]['logprob'] = logprob.flatten()
            
class BPFast():
    ''' BayesPRF 
    '''

    def __init__(self, model, real_ts,  **kwargs):        
        ''' __init__
        Set up the object; with important info
        '''
        self.model_labels = prfpy_params_dict()[model]
        self.real_ts = real_ts.squeeze()
        # Setup and save model information:
        self.n_params = len(self.model_labels) - 1
        self.model_labels_inv = {value: key for key, value in self.model_labels.items()}        
        # MCMC specific information
        self.model_sampler = {}                                     # A sampler for each parameter (e.g., randomly pick "x" b/w -5 and 5)
        self.model_prior = {}                                       # Prior for each parameter (e.g., normal distribution at 0 for "x")
        self.model_prior_type = {}                                  # For each parameter: can be "uniform" (just bounds), or "normal"        
        self.fixed_vals = {}                                        # We can fix some parameters. To speed things up, don't include them in the MCMC process
        self.bounds = {}

    def add_prior(self, pid, prior_type, **kwargs):
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
            self.bounds[pid] = [fixed_val, fixed_val]
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

            if bounds[v_p][0]!=bounds[v_p][1]: 
                self.add_prior(
                    pid=v_p,
                    prior_type = 'uniform',
                    vmin = bounds[v_p][0],
                    vmax = bounds[v_p][1],
                    )
            else: # If upper & lower bound are the same, make it a fixed parameter
                self.add_prior(
                    pid=v_p,
                    prior_type = 'fixed',
                    fixed_val = bounds[v_p][0],
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
        pred = prfpy_global_model.prfpy_model.return_prediction(*list(np.array(new_params)))
        pred = np.nan_to_num(np.squeeze(pred))
        return pred        

    # Log-likelihood function for the model
    def ln_likelihood(self, params):
        ''' 
        '''
        response = self.real_ts.copy()
        # [1] Get the predicted time series
        model_response = self.prfpy_model_wrapper(params)        
        if np.all(model_response == 0):
            # For some reason 0
            return -np.inf        
        # [2] Calculate the residuals
        residuals = response - model_response

        # [3] Fit a normal distribution to the residuals
        # Assume mean of residuals is 0
        # muhat, sigmahat = stats.norm.fit(residuals)
        muhat = 0
        _, sigmahat = stats.norm.fit(residuals, floc=0) 

        # Even faster? If we know mean is 0, don't want to fit std, but calculate it?
        # sigmahat = np.std(residuals)           

        # Calculate the log likelihood of the residuals
        # given the fitted normal distribution
        # then add it up for all time points
        # SLOWER TO CALL OUT TO LIBRARIES: log_like = stats.norm.logpdf(residuals, muhat, sigmahat).sum()
        log_like = -0.5 * np.sum((residuals / sigmahat) ** 2 + np.log(2 * np.pi * sigmahat**2))
        return log_like

    # Log-prior function for the model
    def ln_prior(self, params):
        p_out = 0.0
        for v_p in self.fit_p_list:
            i_p = self.init_p_id[v_p]
            p_out += self.model_prior[v_p](params[i_p])
        return p_out    

    def ln_posterior(self, params):
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params)
        return prior + like, like # save both...

# *** PRIORS ***
class PriorNorm():    
    def __init__(self, loc, scale):
        self.loc = loc # mean
        self.scale = scale # standard deviation
    def sampler(self, n_samples):
        # Sample from the normal distribution
        return np.random.normal(self.loc, self.scale, n_samples)
    def prior(self, p):
        # Return the log probability of the parameter given the normal distribution
        return stats.norm.logpdf(p, self.loc, self.scale)

class PriorUniform():
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
    def sampler(self, n_samples):
        return np.random.uniform(self.vmin, self.vmax, n_samples)
    def prior(self, param):
        return 0 if self.vmin <= param <= self.vmax else -np.inf

class PriorFixed():
    def __init__(self, fixed_val):
        self.fixed_val = fixed_val
        self.vmin = fixed_val
        self.vmax = fixed_val
    def prior(self, param):
        # I know it seems silly, but it ensures the datatypes are the same
        return param*0.0

class PriorNone():
    def __init__(self):
        self.bounds = 'None'        
    def prior(self,param):
        # I know it seems silly, but it ensures the datatypes are the same
        return param*0.0 