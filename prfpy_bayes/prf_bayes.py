from prfpy.model import *
import numpy as np
from scipy import stats 
import emcee
from .utils import prfpy_index

class BayesPRF():

    def __init__(self, model, prfpy_stim, **kwargs):
        self.model = model
        self.normalize_RFs = kwargs.get('normalize_RFs', False)
        self.prfpy_stim = prfpy_stim
        self.model_labels = prfpy_index()[self.model]
        self.model_labels.pop('rsq')
        self.n_params = len(self.model_labels)
        self.model_labels_inv = {value: key for key, value in self.model_labels.items()}
        self.load_prfpy_model()
        self.model_sampler = {}
        self.model_prior = {}
        self.model_prior_type = {}
        self.fixed_vals = {}

    def add_prior(self, pid, **kwargs):
        ''' 
        Adds the prior to each parameter:
        This will determine the sampling for initial parameters
        > is this valid? 

        Can be uniform: b/w 2 bouds
        Fixed: then the emcee will not search for it..
        Other.. e.g., normal distributions
        '''
        if pid not in self.model_labels.keys():
            print('error...')
            return
        prior_type = kwargs.get('prior_type')
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
            self.model_sampler[pid] = PriorUniform(vmin, vmax).sampler
            self.model_prior[pid] = PriorUniform(vmin, vmax).prior            

        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.fixed_vals[pid] = fixed_val
            self.model_prior[pid] = PriorFixed(fixed_val).prior
    
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
            else:
                self.add_prior(
                    pid=v_p,
                    prior_type = 'fixed',
                    fixed_val = bounds[i_p][0],
                    )
                

    def prep_info(self):
        # Get everything ready for fitting:
        # Which parameters are being fit or fixed?
        self.n_params2fit = self.n_params - len(self.fixed_vals)
        self.all_p_list = list(self.model_labels.keys())
        self.fix_p_list = list(self.fixed_vals.keys())
        self.fit_p_list = list(x for x in self.all_p_list if x not in self.fix_p_list)
        self.init_p_id = {}
        for i_p,p in enumerate(self.fit_p_list):
            self.init_p_id[p] = i_p

    def sample_initial_params(self, n_walkers):
        # Only initialise params to fit...
        initial_params = []
        for iw in range(n_walkers):
            params = np.zeros(self.n_params2fit)
            for i_p,p in enumerate(self.fit_p_list):
                params[i_p] = self.model_sampler[p](1)
            initial_params.append(np.array(params))
        return initial_params
    
    def sample_tiny_gauss_ball(self, params_in, n_walkers, eps=1e-4):
        p2fit_in = self.params_full2fit(params_in)
        initial_params = p2fit_in + eps * np.random.randn(n_walkers, self.n_params2fit)
        return initial_params        
        
    def load_prfpy_model(self):
        if self.model=='gauss':
            self.prfpy_model = Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='css':
            self.prfpy_model = CSS_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='dog':
            self.prfpy_model = DoG_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )
        elif self.model=='norm':
            self.prfpy_model = Norm_Iso2DGaussianModel(stimulus=self.prfpy_stim, normalize_RFs = self.normalize_RFs )

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

    # Log-prior function for the Gaussian pRF model
    def ln_prior(self, params):
        p_out = 0.0
        for v_p in self.fit_p_list:
            i_p = self.init_p_id[v_p]
            p_out += self.model_prior[v_p](params[i_p])
        return p_out    

    # Log-posterior function for the Gaussian pRF model
    def ln_posterior(self, params, response):
        prior = self.ln_prior(params)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.ln_likelihood(params, response)


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

def run_emcee_basic(n_walkers, n_steps, true_resp,bprf, pool=None, walkers=None, **kwargs):
    # bloop
    if walkers is None:
        walkers = bprf.sample_initial_params(n_walkers=n_walkers)
    sampler = emcee.EnsembleSampler(
        len(walkers), 
        bprf.n_params2fit, 
        bprf.ln_posterior, 
        args=(true_resp,),
        pool=pool,
        **kwargs
        # moves=moves,        
        )
    sampler.run_mcmc(walkers, n_steps)
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