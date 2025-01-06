import numpy as np
from scipy import stats 
import emcee
import copy

# Import the prfpy model
try: 
    from prfpy_csenf.model import *
except:
    from prfpy.model import *
from .utils import *
from dag_prf_utils.prfpy_ts_plotter import TSPlotter
from dag_prf_utils.utils import dag_get_rsq

# Don't set it...
prfpy_global_model = PrfpyModelGlobal()

class MicroProbe():
    '''Class to do some mico-probing of the data    
    '''
    def __init__(self, prfpy_model, real_ts, **kwargs):
        '''Initialise the class
        prfpy_model: prfpy model object
        real_ts: the real time series
        kwargs: other arguments
        '''
        self.kwargs = copy.deepcopy(kwargs)
        self.model='gauss'
        self.prfpy_model = prfpy_model
        self.real_ts = real_ts
        # DEFAULTS 
        self.bounds = kwargs.get('bounds', [-5, 5])
        # -> HRF parameters (coefficient for the derivative of the HRF and dispersion)
        self.hrf_deriv = kwargs.get('hrf_deriv', prfpy_model.hrf_params[1]) 
        self.hrf_disp = kwargs.get('hrf_disp', prfpy_model.hrf_params[2])
        # -> size of microprobe
        self.tiny_prf_size = kwargs.get('tiny_prf_size', 0.01) # 0.01 degrees
        self.fixed_baseline = kwargs.get('fixed_baseline', None) # Fix the baseline during fitting?
        self.init_walker_method = kwargs.get('init_walker_method', 'grid')          
        self.gauss_ball_jitter = kwargs.get('gauss_ball_jitter', 1)              # How much to jitter each parameter by
        # -> where we will save the fits to the data
        self.sampler = [None] * len(real_ts)

    def ln_prior(self, params):
        '''Log prior
        Assume uniform priors for x,y
        Check the parameters (x,y) are they in the bound? 
        If not return -inf
        '''
        p_out = 0.0
        for p in params:
            if p < self.bounds[0] or p > self.bounds[1]:
                p_out += -np.inf # Log(0) = -inf
            else:
                p_out += 0.0 # Log(1) = 0
        return p_out
        
    def ln_likelihood(self, params, response):
        '''Log likelihood        
        '''
        # Get the predicted time series
        pred = self.prfpy_model.return_prediction(
            mu_x=np.array([params[0]]),
            mu_y=np.array([params[1]]),
            size=np.array([self.tiny_prf_size]),
            beta=np.array([1]), # beta is the amplitude of the response
            baseline=np.array([0]),
        ).squeeze()
        if np.all(pred == 0):
            return -np.inf
        log_like = ln_likelihood(
            pred=pred, response=response, 
            fixed_baseline=self.fixed_baseline, 
        )
        return log_like
    
    def ln_posterior(self, params, response):
        '''Log posterior
        Here we combine the prior and likelihood
        Return both; it is useful for tracking...
        '''
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params, response)
        return prior + like, like # Save both the prior and likelihood

    def initialise_walkers(self, n_walkers, **kwargs):
        ''' initialise_walkers for a given voxel
        '''
        init_walker_method = kwargs.get('init_walker_method', self.init_walker_method)
        initial_guess = kwargs.get('initial_guess', [0,0]) # What is our starting point
        eps = kwargs.get('gaus_ball_jitter', self.gauss_ball_jitter)      # The amount of noise to add 
        ow_walkers = kwargs.get('walkers', None)
        if ow_walkers is not None:
            return ow_walkers
        if init_walker_method == "random":
            # Create walkers from random prior
            walker_start = np.random.uniform(
                low=self.bounds[0], high=self.bounds[1], size=(n_walkers, len(initial_guess))
            )
        elif init_walker_method == "gauss_ball":
            walker_start = initial_guess + eps * np.random.randn(n_walkers, len(initial_guess))
        elif init_walker_method == "grid":
            # x grid 
            x = np.linspace(self.bounds[0], self.bounds[1], int(np.sqrt(n_walkers)))
            y = np.linspace(self.bounds[0], self.bounds[1], int(np.sqrt(n_walkers)))
            x, y = np.meshgrid(x, y)
            walker_start = np.vstack([x.flatten(), y.flatten()]).T
        return walker_start
    
    def run_mcmc_fit(self, idx, n_walkers, n_steps, **kwargs):
        '''Run the mcmc fitting!
        '''        
        target_timeseries = self.real_ts[idx,:].copy() # Make a copy of the time series 
        # Optional arguments... process with defaults 
        save_mode       = kwargs.pop('save_mode', 'minimal') # What to save in 'sampler': can be obj or minimal
        save_top_kpsc   = kwargs.pop('save_top_kpsc', None) # Keep the top n % (in terms of model fit)
        save_min_rsq    = kwargs.pop('save_min_rsq', None) # Only save if the rsq is above this
        burn_in         = kwargs.pop('burn_in', 0) # How many steps to burn in
        pool            = kwargs.pop('pool', None)        
        kwargs_sampler  = kwargs.get('kwargs_sampler', {})
        kwargs_run      = kwargs.get('kwargs_run', {})
        initial_guess   = kwargs.pop('initial_guess', [0, 0])
        walkers = self.initialise_walkers(
            n_walkers=n_walkers, initial_guess=initial_guess, **kwargs)
        if n_walkers != len(walkers):
            print('Used grid - so we got more walkers than requested')
            print(f'updating n_walkers to {len(walkers)}')
            n_walkers = len(walkers)
        n_paramss2fit = len(initial_guess)
        # Quick test first - is everything going to work?
        self.ln_prior(walkers[0])
        self.ln_likelihood(walkers[0], target_timeseries)
        self.ln_posterior(walkers[0], target_timeseries)
        # Ok - now we can run the sampler!!
        # Running in parallel?
        if pool is not None:
            # make a fast one
            # First set the prfpy model to the "global" one
            print('Running in parallel')
            prfpy_global_model.set_model(self.prfpy_model)
            # Now make the fast one
            mpfast = MPFast(real_ts=target_timeseries, **self.kwargs)
            # Run a quick test
            mpfast.ln_posterior(walkers[0])
            sampler = emcee.EnsembleSampler(
                nwalkers=len(walkers), 
                ndim=n_paramss2fit, 
                log_prob_fn=mpfast.ln_posterior, 
                pool=pool,
                **kwargs_sampler, # Any other arguments that you want to pass to the sampler
                )            
        else:
            print('Running in serial')
            sampler = emcee.EnsembleSampler(
                nwalkers=len(walkers), 
                ndim=n_paramss2fit, 
                log_prob_fn=self.ln_posterior, 
                args=(target_timeseries,),
                **kwargs_sampler, # Any other arguments that you want to pass to the sampler
                )
        sampler.run_mcmc(
            walkers, 
            n_steps, 
            **kwargs_run # Any other arguments that you want to pass to the run_mcmc
            )
        # Return the chain, log_prob, (see emcee documentation)
        chain = sampler.get_chain(discard=burn_in, flat=False)
        # Flatten the chains....
        flat_params = chain.reshape(-1, n_paramss2fit)
        logprob = sampler.get_log_prob(discard=burn_in, flat=True)

        n_out = n_steps - burn_in
        # Get the index of the walkers, and the steps
        walker_id, step_id = np.meshgrid(np.arange(n_walkers), np.arange(n_out)+burn_in)
        walker_id = walker_id.flatten()
        step_id = step_id.flatten()

        # Here i'm not fitting the slope or baseline (i.e., the GLM bit) inside the MCMC
        # So we need to do this now
        slopes, baselines = self._return_amp_and_bl(flat_params, target_timeseries)

        # Normally when you run prfpy for gauss fit you get array (x, y, size, beta, baseline, hrf1, hrf2, rsq)
        # Lets put it into that format
        full_params = np.zeros((flat_params.shape[0], 8)) # +1 for rsq
        full_params[:,0] = flat_params[:,0] # x
        full_params[:,1] = flat_params[:,1] # y
        full_params[:,2] = self.tiny_prf_size # size
        full_params[:,3] = slopes # beta
        full_params[:,4] = baselines # baseline
        full_params[:,5] = 4.6 # hrf1
        full_params[:,6] = 0 # hrf2    
        
        # Recalculate rsq 
        preds = self.prfpy_model.return_prediction(
            mu_x=full_params[:,0],
            mu_y=full_params[:,1],
            size=full_params[:,2],
            beta=full_params[:,3],
            baseline=full_params[:,4],
        )
        
        rsq = dag_get_rsq(tc_target=target_timeseries, tc_fit=preds)
        full_params[:,-1] = rsq
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

        if save_mode == 'obj':
            # Now make a PRF object
            # I've created a useful class for plotting...
            prf_plotter = TSPlotter(
                prf_params=full_params,
                model='gauss',
                prfpy_model=self.prfpy_model,
                real_ts=np.repeat(target_timeseries[np.newaxis,...], full_params.shape[0], axis=0), # Repeat the target timeseries for each parameter set
            )
            # Also useful to know...
            prf_plotter.pd_params['logprob'] = logprob.flatten()
            prf_plotter.pd_params['walker_id'] = walker_id
            prf_plotter.pd_params['step_id'] = step_id

            # Save the sampler
            self.sampler[idx] = prf_plotter
        elif save_mode == 'minimal':
            # Save only the important stuff... (don't want it to be too big)
            self.sampler[idx] = {
                'x': full_params[:,0],
                'y': full_params[:,1],
                'amp_1' : full_params[:,3],
                'bold_baseline' : full_params[:,4],
                'rsq': full_params[:,-1],
                'logprob': logprob,
                'walker_id': walker_id,
                'step_id': step_id,
            }


    # OTHER USEFULT FUNCTION
    def _return_amp_and_bl(self, params, response):
        preds = self.prfpy_model.return_prediction(
            mu_x=params[:,0],
            mu_y=params[:,1],
            size=np.array([self.tiny_prf_size]),
            beta=np.array([1]),
            baseline=np.array([0]),
        )

        slopes, baselines = quick_glm(preds, response, fixed_baseline=self.fixed_baseline)
        return slopes, baselines
    


class MPFast():
    '''Same as above, but smaller stuff so it is faster to pickle
    (should speed up the parallel processing)
    ''' 
    def __init__(self, real_ts, **kwargs):
        '''Initialise the class
        prfpy_model: prfpy model object
        real_ts: the real time series
        kwargs: other arguments
        '''
        self.model='gauss'
        self.real_ts = real_ts.squeeze()
        self.sumd = np.sum(self.real_ts) # Add it once, not on every loop... (save a bit of time)

        # DEFAULTS 
        self.bounds = kwargs.get('bounds', [-5, 5])
        # -> HRF parameters (coefficient for the derivative of the HRF and dispersion)
        self.hrf_deriv = kwargs.get('hrf_deriv', 4.6) 
        self.hrf_disp = kwargs.get('hrf_disp', 0)
        # -> size of microprobe
        self.tiny_prf_size = kwargs.get('tiny_prf_size', 0.01) # 0.01 degrees
        self.fixed_baseline = kwargs.get('fixed_baseline', None) # Fix the baseline during fitting?

    def ln_prior(self, params):
        '''Log prior
        Assume uniform priors for x,y
        Check the parameters (x,y) are they in the bound? 
        If not return -inf
        '''
        p_out = 0.0
        for p in params:
            if p < self.bounds[0] or p > self.bounds[1]:
                p_out += -np.inf # Log(0) = -inf
            else:
                p_out += 0.0 # Log(1) = 0
        return p_out
        
    def ln_likelihood(self, params):
        '''Log likelihood
        > only pickle the response once (this is for 1 vx)
        > assume the prfpy model is global
        '''
        response = self.real_ts
        # Get the predicted time series
        pred = prfpy_global_model.prfpy_model.return_prediction(
            mu_x=np.array([params[0]]),
            mu_y=np.array([params[1]]),
            size=np.array([self.tiny_prf_size]),
            beta=np.array([1]), # beta is the amplitude of the response
            baseline=np.array([0]),
        ).squeeze()
        if np.all(pred == 0):
            return -np.inf        
        log_like = ln_likelihood(
            pred=pred, response=response, 
            fixed_baseline=self.fixed_baseline, 
            sumd=self.sumd,
        )

        return log_like
    
    def ln_posterior(self, params):
        '''Log posterior
        Here we combine the prior and likelihood
        Return both; it is useful for tracking...
        '''
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params)
        return prior + like, like # Save both the prior and likelihood