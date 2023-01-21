import numpy as np
import random

random.seed(42)
np.random.seed(42)


class Agent:
    """Agent who applies for credit.

    H \in (0, 1], called tier;
    Y \in {0, 1}, called Y_ori, Y_obs;
    A discrete, called protected_feature;
    X multi-dimensional real, called ordinary_feature.
    D \in {0, 1}, called D (decision)

    Will record (H, Y_ori, Y_obs, D) history after register decision.

    [NOTE] Always train an LR to fit (A, X_all) -> Y_obs.
    Updated at each T = t. Useful to estimate tier if needed.
    Applicable to both simulation and real-world data.
    Always treat tier and Y_ori as if they were intrinsic


    """
    def __init__(self,
                 index,
                 scenario,
                 protected_feature,
                 feature_generate_func,
                 init_tier_cdf=None,
                 init_ordinary_feature=None,
                 init_ground_truth=None,
                 tier_model=None,
                 init_accept_p=1):
        """Initialization for Agent.

        T = 1 as the start.

        If for simulation initialization,
        sample e from P_E (not change with time),
        and sample h from P_H.
        If for real-world data initialization,
        need to estimate H first, or give CDF to sample from.

        Args:
            `index`: id of the agent
            `scenario`: option for experiment
                'simulation': generate X according to simulation.
                    (not None == init_tier_cdf)
                'creditscore': TransUnion Credit Score data set.
                    A uniformally distribution (balanced class),
                    and CDF of H is available, just reverse_sample from it.
                    (not None == init_tier_cdf)

            `protected_feature`: the protected feature A

            `feature_generate_func`: model specification
                A parameterized model for simulation or real data.
                (A, H) -> (X_not_protected_descendent, X_protected_descendent).
                This func is re-used for different T = t (stable module assumption).
                X_'s need to be of np.ndarray type

            `init_tier_cdf`: only used for 'simulation' and 'creditscore'
                A dictionary with A as the key, conditional CDF as value.
                For example, {0: U[0.1, 0.6], 1: U[.3, .7], 2: U[.2, .8]}

            `init_ordinary_feature`: only used for real data
                Directly input X from data set.

            `init_ground_truth`: only used for real data
                Directly input Y_ori using Y from data.

            `tier_model`: only used for real data.
                For unknown H, need to get LogicticRegressio estimate
                of E[ Y | features ] as LR_estimate, and then initialize
                with estimated H and Y_ori.
                (A, X_not_protected_descendent, X_protected_descendent) -> H_pred

            `init_accept_p`: initial successful p for binomial (D at t = 0)
                Mask Y_ori when t = 1 (start) and get Y_obs,
                if sample 1 Yori is observed, else Yori is not observed.
                Set p to 1 by default, observe all Y_ori at the begining.
        """
        self.index = index

        # protected feature, no need to change it for future t
        self.protected_feature = protected_feature

        # record of history, start from T = 1
        self.tier_history = []
        self.Y_ori_history = []
        self.Y_obs_history = []
        self.D_history = []

        # --> initialize H
        # initialize H
        self.init_tier(scenario, init_tier_cdf, init_ordinary_feature,
                       tier_model)

        # --> initialize Y_ori
        self.init_ground_truth_original(scenario, init_ground_truth)

        # --> initialize Y_obs
        # sample decision at time 0, no need to record
        self.decision_prehistory = np.random.binomial(1, init_accept_p)
        # mask Y_ori to get Y_obs
        self.Y_obs = self.Y_ori if 1 == self.decision_prehistory else -1

        # --> initialize D
        # D to be determined by decision module
        self.D = -1  # D at T = 1 is not determined yet

        # --> initializa X, seperate nondescendent/descendent of A
        # generate X according to scenario, module not changing over time
        self.X_not_protected_descendent, self.X_protected_descendent = feature_generate_func(
            self.protected_feature, self.tier)

        # --> initialize flags
        # flag: if decision is registered
        self.decision_registered = 0
        # flag: if data dynamics goes from T = t to T = t + 1
        self.dynamics_registered = 1  # initialized

    def init_tier(self,
                  scenario,
                  init_tier_cdf=None,
                  init_ordinary_feature=None,
                  tier_model=None):
        """Initialize value of H.

        """
        if scenario in ['simulation', 'creditscore']:
            # choose a pre-defined distribution from dict
            inverted_cdf_func = init_tier_cdf[self.protected_feature]
            self.tier = inverted_cdf_func(np.random.random_sample())
        else:
            raise Exception('Invalid scenario option.')

        # regulate lower and upper bound
        self.tier_regulate()

    def init_ground_truth_original(self, scenario, init_ground_truth):
        """Initialize value of Y_ori.

        Call after initializing self.tier

        For 'simulation', sample Y_ori, and get Y_obs by masking Y_ori.
        For 'creditscore', Y not given, same as 'simulation'.

        """
        if scenario in ['simulation', 'creditscore']:
            # initialize Y_ori, no matter decision_pre_history
            self.Y_ori = np.random.binomial(1, self.tier)
        else:
            raise Exception('Invalid scenario option.')

    def register_decision(self, decision_type, decision_model):
        """Decision register module, call from outside.

        Register D_t for current step T = t, and update history.
        Record current to history before calling dynamics update.

        Change in self: (D_t)

        Everyone should receive a decision.

        Args:
            decision_type: option
                'sim_perfect': simulation only, D_t = Y_ori_t,
                'accurate': real-data, accuracy prioritized LogisticRegression,
                'CF': simulation & real-data, use level-1 CF implementation
                    on fitted LogisticRegression, i.e., X_not_protected_descendent.

            decision_model: None or fitted sklearn.linear_model.LogisticRegression.
                if 'sim_perfect': return D = Y_ori,
                if 'accurate': return LR_decision.predict(all feature),
                if 'CF': return LR_decision.predict(non-descent of A).
        """
        assert 0 == self.decision_registered, 'Decision already registered.'
        assert 1 == self.dynamics_registered, 'Latest dynamics not registered.'

        if 'sim_perfect' == decision_type:
            self.D = self.Y_ori
        elif 'accurate' == decision_type:
            _all_feature, _ = self.feature_label_retrieve('all')
            self.D = int(decision_model.predict(_all_feature.reshape(
                1, -1)))  # 2D array requirement
        elif 'CF' == decision_type:
            cf_feature, _ = self.feature_label_retrieve('CF')
            self.D = int(decision_model.predict(cf_feature.reshape(
                1, -1)))  # 2D array requirement
        else:
            raise Exception('Decision type not supported.')

        # already updated D_t
        self.decision_registered = 1
        # wait for (H, Y_ori, Y_obs)_t+1
        self.dynamics_registered = 0

        # --> record current to history
        self.record_history()

    def register_dynamics(self,
                          scenario,
                          feature_generate_func,
                          alpha_D,
                          alpha_Y,
                          tier_model=None):
        """Update the (A, H, X, Y_ori, Y_obs) using the specified dynamics.

        Call from outside.

        Calculate situation changes in T = t + 1, using current data at T = t.
        This function ignore whether H and Y_ori are estimated or intrinsic.

        Change in self: (A_t+1, H_t+1, Y_ori_t+1, Y_obs_t+1)

        If D_t == 0, set Y_obs_t+1 = -1.
        Might need to estimate H_t and sample Y_ori_t first.

        Args:
            scenario: option
                'simulation': input (A_t, X_t, H_t, Y_ori_t, D_t)
                'creditscore':
                    input (A_t, X_t, Y_obs_t, D_t), i.e.,
                    will need to estimate H_t, and Y_ori_t first,
                    then use them as proxy input.
            tier_model: None, or LR_estimate for real-world data

        """
        assert 1 == self.decision_registered, 'Latest decision not registered.'
        assert 0 == self.dynamics_registered, 'Dynamics already registered.'

        # if esmating current (H_t, Y_ori_t) is needed
        if 'simulation' == scenario:  # simulation
            pass  # just use intrinsic H_t and Y_ori_t
        else:
            # estimate H_t
            self.tier = self.estimate_tier(tier_model)

            # estimate Y_ori_t
            self.Y_ori = np.random.binomial(1, self.tier)
            # for those observed Y_obs_t, Y_obs_t overrule
            if self.Y_obs in (0, 1):
                self.Y_ori = self.Y_obs

        # --> update H_t+1
        self.tier = self.tier * (1 + alpha_D * (2 * self.D - 1) + alpha_Y *
                                 (2 * self.Y_ori - 1))
        self.tier_regulate()

        # --> update Y_ori_t+1
        # sample Y_ori_t+1
        self.Y_ori = np.random.binomial(1, self.tier)

        # --> update Y_obs_t+1
        # update Y_obs_t+1 according to D_t
        self.Y_obs = self.Y_ori if 1 == self.D else -1

        # --> update X_t+1, seperate nondescendent/descendent of A
        # generate X according to scenario, module not changing over time
        self.X_not_protected_descendent, self.X_protected_descendent = feature_generate_func(
            self.protected_feature, self.tier)

        # wait for D_t+1
        self.decision_registered = 0
        # already updated set (H, Y_ori, Y_obs)_t+1
        self.dynamics_registered = 1

    def tier_regulate(self):
        # upper and lower bound
        self.tier = max(1e-6, self.tier)
        self.tier = min(1 - 1e-6, self.tier)

    def record_history(self):
        """Update history record list.

        [NOTE] Should NOT call from outside, will duplicate record.

        Call after current decision registered,
        but before dynamics registered.

        Record the current H_t, Y_ori_t, Y_obs_t, D_t
        """
        assert 1 == self.decision_registered, 'Not correct call'
        assert 0 == self.dynamics_registered, 'Not correct call'

        self.tier_history.append(self.tier)
        self.Y_ori_history.append(self.Y_ori)
        self.Y_obs_history.append(self.Y_obs)
        self.D_history.append(self.D)

    def estimate_tier(self, tier_model):
        """Estimate value of hidden H.

        Real-world use only.

        Args:
            tier_model: estimate H, as proxy.
                (A, X_not_protected_descendent, X_protected_descendent) -> H_pred

        """
        assert not None == tier_model, 'not available'

        # use all feature
        _all_feature, _ = self.feature_label_retrieve('all')
        tier_pred = tier_model.predict_proba(_all_feature.reshape(
            1, -1))[0, 1]  # 2D array requirement

        return tier_pred

    def debug_check_tier_estimate(self, tier_model):
        """Check estimated H vs. self.tier in simulation.

        Only for debug purpose, do not call in regular pipeline.

        [NOTE] tier_model need to be re-trained for each T = t.

        Call after decision registered and tier_model updated,
        but before registering new dynamics.
        """
        # timing of debug
        assert 1 == self.decision_registered, 'Check debug timing.'
        assert 0 == self.dynamics_registered, 'Check debug timing.'

        # self.tier assigned by func
        tier_pred = self.estimate_tier(tier_model)
        print(f'[Debug] id {self.index}: self.tier value is {self.tier},')
        print(f'        id {self.index}: estimated value is {tier_pred}.')

    def feature_label_retrieve(self, feature_retrieve_option):
        """Retrieve current feature and label.

        feature_retrieve_option = 'all' or 'CF' (counterfactual fair)

        Returns:
            feature: np.ndarray size = (n_dim, )
                for 'all' option, (A, X_not_protected_descendent, X_protected_descendent),
                for 'CF' option, (X_not_protected_descendent)

            Y_obs: int {0, 1}
        """
        if 'all' == feature_retrieve_option:
            feature_retrieve = np.hstack(
                (self.protected_feature,
                 self.X_not_protected_descendent.reshape(-1),
                 self.X_protected_descendent.reshape(-1)))
        elif 'CF' == feature_retrieve_option:
            feature_retrieve = np.hstack(
                (self.X_not_protected_descendent.reshape(-1)))
        else:
            raise Exception('Invalid feature_retrieve_option')

        Y_obs = self.Y_obs

        return feature_retrieve, Y_obs
