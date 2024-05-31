import numpy as np
import os
opj = os.path.join

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

# def prfpy_index():
#     '''
#     Easy look up table for prfpy model parameters
#     name to index...
#     '''
#     p_order = {}
#     # [1] gauss. Note hrf_1, and hrf_2 are idx 5 and 6, if fit...
#     p_order['gauss'] = {
#         'x'             :  0, # mu_x
#         'y'             :  1, # mu_y
#         'size_1'        :  2, # size
#         'amp_1'         :  3, # beta
#         'bold_baseline' :  4, # baseline 
#         'hrf_deriv'     :  5, # *hrf_1
#         'hrf_disp'      :  6, # *hrf_2
#         'rsq'           : -1, # ... 
#     }    
#     # [2] css. Note hrf_1, and hrf_2 are idx 6 and 7, if fit...
#     p_order['css'] = {
#         'x'             :  0, # mu_x
#         'y'             :  1, # mu_y
#         'size_1'        :  2, # size
#         'amp_1'         :  3, # beta
#         'bold_baseline' :  4, # baseline 
#         'n_exp'         :  5, # n
#         'hrf_deriv'     :  6, # *hrf_1
#         'hrf_disp'      :  7, # *hrf_2        
#         'rsq'           : -1, # ... 
#     }

#     # [3] dog. Note hrf_1, and hrf_2 are idx 7 and 8, if fit...
#     p_order['dog'] = {
#         'x'             :  0, # mu_x
#         'y'             :  1, # mu_y
#         'size_1'        :  2, # prf_size
#         'amp_1'         :  3, # prf_amplitude
#         'bold_baseline' :  4, # bold_baseline 
#         'amp_2'         :  5, # srf_amplitude
#         'size_2'        :  6, # srf_size
#         'hrf_deriv'     :  7, # *hrf_1
#         'hrf_disp'      :  8, # *hrf_2        
#         'rsq'           : -1, # ... 
#     }

#     p_order['norm'] = {
#         'x'             :  0, # mu_x
#         'y'             :  1, # mu_y
#         'size_1'        :  2, # prf_size
#         'amp_1'         :  3, # prf_amplitude
#         'bold_baseline' :  4, # bold_baseline 
#         'amp_2'         :  5, # srf_amplitude
#         'size_2'        :  6, # srf_size
#         'b_val'         :  7, # neural_baseline 
#         'd_val'         :  8, # surround_baseline
#         'hrf_deriv'     :  9, # *hrf_1
#         'hrf_disp'      : 10, # *hrf_2        
#         'rsq'           : -1, # rsq
#     }            

#     p_order['csf']  ={
#         'width_r'       :  0,
#         'sf0'           :  1,
#         'maxC'          :  2,
#         'width_l'       :  3,
#         'amp_1'         :  4,
#         'bold_baseline' :  5,
#         'hrf_deriv'     :  6, # *hrf_1
#         'hrf_disp'      :  7, # *hrf_2        
#         'rsq'           : -1,
#     }

#     return p_order


def logprob2rsq(log_prob, ts):
    SS_res = -2 * log_prob    
    SS_tot = np.sum((ts-ts.mean())**2)
    rsq = 1 - (SS_res/SS_tot)
    return rsq