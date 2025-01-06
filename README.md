# prfpy_bayes

Extending prfpy (https://github.com/VU-Cog-Sci/prfpy/tree/main/prfpy) for bayesian (MCMC analysis)

NOTE IN DEVELOPMENT -> much may change!!! 

Need to have installed 

dag_prf_utils

https://github.com/mdaghlian/dag_prf_utils


Run the setup script and get all the associated packages


Also a bit of an apology... There is some inconsistency in naming between prfpy and dag_prf_utils. I promise there is a good reason for this. I want prfpy to be backwards compatible. But, some names interfere with each other, when you start including other models (e.g., size, compared to size_1, and size_2 in the DN model). Here is the mapping for the gaussian model:

```
prfpy       dag_prf_utils

mu_x        x
mu_y        y
size        size_1
beta        amp_1
baseline    bold_baseline
hrf_1       hrf_deriv
hrf_2       hrf_disp


```

Most of the time you shouldn't have to think about it...

written by Marcus Daghlian