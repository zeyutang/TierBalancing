import numpy as np
from TB_utils.pipeline_util import workhorse
import random

random.seed(42)
np.random.seed(42)

n_agent = 1000
n_step = 5
alpha_D = 0.1
alpha_Y = 0.08

# simulation setting perfect predictor
print('Setting: simulation w/ perfect predictor')
workhorse(n_agent=n_agent,
          n_step=n_step,
          alpha_D=alpha_D,
          alpha_Y=alpha_Y,
          scenario='simulation',
          decision_type='sim_perfect',
          init_tier_cdf_option='uniform',
          feature_retrieve_option='all',
          init_accept_p=.9)

# simultion setting CF predictor
print('Setting: simulation w/ CF predictor')
workhorse(n_agent=n_agent,
          n_step=n_step,
          alpha_D=alpha_D,
          alpha_Y=alpha_Y,
          scenario='simulation',
          decision_type='CF',
          init_tier_cdf_option='uniform',
          feature_retrieve_option='CF',
          init_accept_p=.9)

# creditscore setting accuracy_first predictor
print('Setting: creditscore w/ accuracy_first predictor')
workhorse(n_agent=n_agent,
          n_step=n_step,
          alpha_D=alpha_D,
          alpha_Y=alpha_Y,
          scenario='creditscore',
          decision_type='accurate',
          init_tier_cdf_option=None,
          feature_retrieve_option='all',
          init_accept_p=.9)

# creditscore setting CF predictor
print('Setting: creditscore w/ CF predictor')
workhorse(n_agent=n_agent,
          n_step=n_step,
          alpha_D=alpha_D,
          alpha_Y=alpha_Y,
          scenario='creditscore',
          decision_type='CF',
          init_tier_cdf_option=None,
          feature_retrieve_option='CF',
          init_accept_p=.9)
