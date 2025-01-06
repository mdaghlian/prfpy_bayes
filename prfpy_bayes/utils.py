import numpy as np
import os
opj = os.path.join
from contextlib import contextmanager

@contextmanager
def try_n_times(max_attempts=2):
    """
    Context manager that retries the enclosed block up to max_attempts times.
    Sometimes first time round fitting fails for some reason...

    :param max_attempts: Maximum number of retry attempts.
    """
    attempt = 0

    while attempt < max_attempts:
        try:
            yield  # Executes the block inside the 'with' statement
            break  # Exit if the block executes successfully
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                print("All attempts failed. Giving up.")
                raise
            print("Retrying...")

class PrfpyModelGlobal():
    '''Make a prfpy model object that can be passed around
    (so that it doesn't have to be pickled for multiprocessing)
    Makes everything faster
    '''
    def __init__(self, model=None):
        self.prfpy_model = model

    def set_model(self, model):
        self.prfpy_model = model

def ln_likelihood(pred, response, **kwargs):
    '''Calculate the loglikelihood from the residuals of the prediction and the data
    
    '''
    do_glm = kwargs.get('do_glm', False)    # Do the glm inside?
    # Automatic rejection - if all predictions are 0
    if np.all(pred == 0):
        return -np.inf
    
    # A GLM, before?
    if do_glm:
        slopes, baselines = quick_glm(
            pred, response, **kwargs
            )
        pred = pred* slopes + baselines

    # Residuals
    residuals = response - pred

    # Likelihood: we always assume residuals are gaussian
    # -> SOME NOTES ON WAYS YOU CAN DO THIS:
    # A. Assume distribution, centred on 0, with std=std(residuals)
    # B. Assume distribution, centred on 0, with std=1 (or some std based on what you expect the noise to be)   
    #       perhaps this could even be a parameter to fit...?
    # C. Assume distribution, with mean and std estimated from the residuals

    # Note that fitting the mean or std of residuals is equivalent to 
    # taking the mean and std; if they are gaussian (because optimizing MLE breaks down to this)

    # Recommendation from Remco is (A)
    sigmahat = residuals.std()
    # SLOWER TO CALL OUT TO LIBRARIES: log_like = stats.norm.logpdf(residuals, 0, sigmahat).sum()
    log_like = -0.5 * np.sum((residuals / sigmahat) ** 2 + np.log(2 * np.pi * sigmahat**2))
    return log_like







def quick_glm(preds, response, **kwargs):
    '''Quick and dirty GLM (Generalized Linear Model) function that estimates the slopes and baselines for the given prediction data.
    
    Args:
    preds (ndarray): 2D array (n_preds x n_timepoints) where each row corresponds to a set of predictions at various timepoints.
    response (ndarray): 2D array (n_timepoints x 1) representing the observed responses at each timepoint.
    **kwargs: Additional optional arguments:
        - 'fixed_baseline': If provided, this will be used as the baseline value and slopes will be computed based on this fixed value.
        - 'sumd': A pre-computed sum of the response values
    
    Returns:
    slopes (ndarray): The calculated slopes for each predictor.
    baselines (ndarray): The estimated baseline for each predictor.
    '''
    # Get optional arguments
    fixed_baseline = kwargs.get('fixed_baseline', None)    
    # If preds is a 1D array, convert it into a 2D array (single row)
    if len(preds.shape) == 1:
        preds = preds[np.newaxis, :]        
    
    # Get the number of predictors (n_preds) and the number of timepoints (n_timepoints)
    n_preds, n_timepoints = preds.shape
    # Initialize arrays for slopes and baselines, with zeros
    slopes = np.zeros((n_preds,))
    baselines = np.zeros((n_preds,))
    
    # Compute the square of the norm of the predictions along each row (for each predictor)
    square_norm_preds = np.sum(preds**2, axis=-1)    
    # Compute the sum of predictions for each predictor across timepoints
    sum_preds = np.sum(preds, axis=-1)

    # If a fixed baseline is provided, use it to compute slopes
    if fixed_baseline is not None:
        
        baseline = fixed_baseline        
        # Compute the slope for each predictor based on the provided fixed baseline
        slopes = (np.dot(response - baseline, preds.T)) / (square_norm_preds)                           
        baselines = baseline * np.ones_like(slopes)
    else:
        # If no fixed baseline is provided, compute slopes and baselines using least-squares estimation        
        # Compute the slopes for each predictor based on the formula
        sumd = kwargs.get('sumd', np.sum(response, axis=0)) # pre-computed sum of the response values (speed!)
        slopes = (n_timepoints * np.dot(response, preds.T) - sumd * sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
        
        # Compute the baselines using the formula derived from the linear system
        baselines = (sumd - slopes * sum_preds) / n_timepoints        
    
    # Return the calculated slopes and baselines
    return slopes, baselines



def prfpy_params_dict():
    '''
    Easy look up table for prfpy model parameters
    name to index...
    '''
    p_order = {}
    # [1] gauss. Note hrf_1, and hrf_2 are idx 5 and 6, if fit...
    p_order['gauss'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'hrf_deriv'     :  5, # *hrf_1
        'hrf_disp'      :  6, # *hrf_2
        'rsq'           : -1, # ... 
    }    
    # [2] css. Note hrf_1, and hrf_2 are idx 6 and 7, if fit...
    p_order['css'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'n_exp'         :  5, # n
        'hrf_deriv'     :  6, # *hrf_1
        'hrf_disp'      :  7, # *hrf_2        
        'rsq'           : -1, # ... 
    }

    # [3] dog. Note hrf_1, and hrf_2 are idx 7 and 8, if fit...
    p_order['dog'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'hrf_deriv'     :  7, # *hrf_1
        'hrf_disp'      :  8, # *hrf_2        
        'rsq'           : -1, # ... 
    }

    p_order['norm'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'b_val'         :  7, # neural_baseline 
        'd_val'         :  8, # surround_baseline
        'hrf_deriv'     :  9, # *hrf_1
        'hrf_disp'      : 10, # *hrf_2        
        'rsq'           : -1, # rsq
    }            

    p_order['csf']  ={
        'width_r'       : 0,
        'SFp'           : 1,
        'CSp'          : 2,
        'width_l'       : 3,
        'crf_exp'       : 4,
        'amp_1'         : 5,
        'bold_baseline' : 6,
        'hrf_1'         : 7,
        'hrf_2'         : 8,
        'rsq'           : -1,
    }

    return p_order

def get_bounds(prf_settings, max_eccentricity, model):
    if model=='norm': # ******************************** NORM
        ext_custom_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            (prf_settings['norm']['neural_baseline_bound']),        # neural baseline (b) 
            (prf_settings['norm']['surround_baseline_bound']),      # surround baseline (d)
            ] 
    elif model=='gauss':
        ext_custom_bounds = []

    elif model=='dog': # ******************************** DOG
        ext_custom_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            ]

    elif model=='css': # ******************************** CSS
        ext_custom_bounds = [
            (prf_settings['css']['css_exponent_bound']),  # css exponent 
            ]

    if model in ('gauss', 'dog', 'css', 'norm'):
        # Combine the bounds 
        # first create the standard bounds
        standard_bounds = [
            (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
            (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
            (1e-1, max_eccentricity*3),                             # prf size bounds
            (prf_settings['prf_ampl']),                             # prf amplitude
            (prf_settings['bold_bsl']),                             # bold baseline (fixed)
        ]    
    elif model == 'csf':
        standard_bounds = [
            (prf_settings['csf_bounds']['width_r']),     # width_r
            (prf_settings['csf_bounds']['sf0']),     # sf0
            (prf_settings['csf_bounds']['maxC']),    # maxC
            (prf_settings['csf_bounds']['width_l']),     # width_l
            (prf_settings['csf_bounds']['beta']),   # beta
            (prf_settings['csf_bounds']['baseline']),      # baseline
        ]
        ext_custom_bounds = []

    # & the hrf bounds. these will be overwritten later by the vx wise hrf parameters
    # ( inherited from previous fits)
    hrf_bounds = [
        (prf_settings['hrf']['deriv_bound']),                   # hrf_1 bound
        (prf_settings['hrf']['disp_bound']),                    # hrf_2 bound
    ]
    ext_bounds = standard_bounds.copy() + ext_custom_bounds.copy() + hrf_bounds.copy()
    return ext_bounds