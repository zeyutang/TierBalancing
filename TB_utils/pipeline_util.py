# # --> pipeline related utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from TB_utils.Agent import Agent
from TB_utils.density_util import simulation_tier_cdf
from TB_utils.density_util import creditscore_tier_cdf
from TB_utils.density_util import predefine_feature_generate
import random

random.seed(42)
np.random.seed(42)


def workhorse(n_agent,
              n_step,
              alpha_D,
              alpha_Y,
              scenario,
              decision_type,
              init_tier_cdf_option=None,
              feature_retrieve_option=None,
              init_accept_p=1):
    """Workhorse function.

    Generate n_agent Agents, run n_step times (T = 1 ... n_step).
    For the purpose of balanced group samples, draw A from uniform.

    Initialization
        agent           initialize dynamics (A, H, X, Y_ori, Y_obs)
    Loop after initialization:
        | population    (mandatory for real data) get df_feature, df_Y_obs
        | population    (mandatory for real data) estimate H
        | population    fit decision-maker (if estimate H, i.e., 'accurate')
        | agent         register decision
        | agent         (debug) debug_check_tier_estimate
        | population    (debug) get df_D
        | population    (debug) instantaneous evaluation
        | agent         register_dynamics
    Take history records and evaluate

    Args:
        `n_agent`: number of agent generated or recorded
        `n_step`: number of steps to run (T = 1 as start)
        `scenario`: 'simulation', 'creditscore'
        `decision_type`: 'sim_perfect', 'accurate', 'CF'
            type of decision maker, see Agent.register_desicion()
        `init_tier_cdf_option`: option for tier cdf init
            For 'simulation', {'uniform', 'truncate_gaussian', 'trigonometric'}
            type of distribution used for simulation cdf initialization.
            For 'creditscore', None.
        `feature_retrieve_option`: 'all', 'CF' when fitting decision_model
            what feature to retrieve when fitting LR,
            see Agent.feature_label_retrieve()
        `init_accept_p`: see init_accept_p in (class) Agent.__init__()

    Return:
        `dict_agent`: dict of agents, each agent contains its history record.
    """
    dict_agent = dict()

    # simulation study
    if 'simulation' == scenario:
        # A binary
        population_A = np.random.choice(np.array([0, 1]), size=(n_agent, ))

        # define tier cdf
        init_tier_cdf = simulation_tier_cdf(init_tier_cdf_option)

        # --> initialize all agents
        for index in range(n_agent):
            # Agent require (not None == init_tier_cdf)
            dict_agent[index] = Agent(
                index=index,
                scenario=scenario,
                protected_feature=population_A[index],
                feature_generate_func=predefine_feature_generate,
                init_tier_cdf=init_tier_cdf,
                init_accept_p=init_accept_p)

        # --> start loop
        for step in range(n_step):
            print(f'Step: {step}')
            # get df_feature and df_Y_obs
            df_feature, df_Y_obs = dict_agent_feature_label_retrieve_current(
                dict_agent, feature_retrieve_option)

            # (optional) actaully no need to estimate H in simulation
            # _all_feature, _ = dict_agent_feature_label_retrieve_current(
            #     dict_agent, 'all')
            # tier_model = LogisticReg_fit_with_all(_all_feature, df_Y_obs)

            # fit decision-maker
            if 'CF' == decision_type:
                assert decision_type == feature_retrieve_option, \
                    'Invalid choice, here should be \'CF\'.'
                decision_model = LogisticReg_CF_level1(df_feature, df_Y_obs)
            elif decision_type in ('sim_perfect', 'accurate'):
                decision_model = None
                # 'sim_perfect' is used by default
                decision_type = 'sim_perfect'
            else:
                raise Exception('Invalid option for decision')

            # register decision (history record updated in Agent)
            for index in range(n_agent):
                dict_agent[index].register_decision(decision_type,
                                                    decision_model)

                # [debug] H_pred and H itself
                # dict_agent[index].debug_check_tier_estimate(tier_model)

            # get df_D
            df_D = dict_agent_decision_retrieve_current(dict_agent)

            # flatten to avoid issue
            df_Y_obs, df_D = df_Y_obs.flatten(), df_D.flatten()

            # [debug] instantaneous evaluate
            print(
                f'  Accuracy (among those with observed Y): {np.count_nonzero(df_D[-1 != df_Y_obs] == df_Y_obs[-1 != df_Y_obs]) / np.count_nonzero(-1 != df_Y_obs)}'
            )

            # register dynamics
            for index in range(n_agent):
                dict_agent[index].register_dynamics(
                    scenario=scenario,
                    feature_generate_func=predefine_feature_generate,
                    alpha_D=alpha_D,
                    alpha_Y=alpha_Y,
                    tier_model=None)

    # creditscore data set
    elif 'creditscore' == scenario:
        # A cardinality = 4
        population_A = np.random.choice(np.array([
            0,
            1,
            2,
            3,
        ]),
                                        size=(n_agent, ))

        # define tier cdf
        init_tier_cdf = creditscore_tier_cdf()

        # --> initialize all agents
        for index in range(n_agent):
            # Agent require (not None == init_tier_cdf)
            dict_agent[index] = Agent(
                index=index,
                scenario=scenario,
                protected_feature=population_A[index],
                feature_generate_func=predefine_feature_generate,
                init_tier_cdf=init_tier_cdf,
                init_accept_p=init_accept_p)

        # --> start loop
        for step in range(n_step):
            print(f'Step: {step}')
            # get df_feature and df_Y_obs
            df_feature, df_Y_obs = dict_agent_feature_label_retrieve_current(
                dict_agent, feature_retrieve_option)

            # need to estimate H
            _all_feature, _ = dict_agent_feature_label_retrieve_current(
                dict_agent, 'all')
            tier_model = LogisticReg_fit_with_all(_all_feature, df_Y_obs)

            # fit decision-maker
            if 'CF' == decision_type:
                assert decision_type == feature_retrieve_option, \
                    'Invalid choice, here should be \'CF\'.'
                decision_model = LogisticReg_CF_level1(df_feature, df_Y_obs)
            elif 'accurate' == decision_type:
                decision_model = tier_model  # fit with all
            else:
                raise Exception('Invalid option for decision')

            for index in range(n_agent):
                # register decision (history record updated in Agent)
                dict_agent[index].register_decision(decision_type,
                                                    decision_model)

            # get df_D
            df_D = dict_agent_decision_retrieve_current(dict_agent)

            # flatten to avoid issue
            df_Y_obs, df_D = df_Y_obs.flatten(), df_D.flatten()

            # [debug] instantaneous evaluate
            print(
                f'  Accuracy (among those with observed Y): {np.count_nonzero(df_D[-1 != df_Y_obs] == df_Y_obs[-1 != df_Y_obs]) / np.count_nonzero(-1 != df_Y_obs)}'
            )

            # register dynamics
            for index in range(n_agent):
                dict_agent[index].register_dynamics(
                    scenario=scenario,
                    feature_generate_func=predefine_feature_generate,
                    alpha_D=alpha_D,
                    alpha_Y=alpha_Y,
                    tier_model=tier_model)
    else:
        print('Invalid option.')

    print('Done!')


def LogisticReg_fit_with_all(_all_feature, Y_obs):
    """Logistic Regressor fitted with all features.

    Call on group-level after initializing/registering dynamics for every agent.
    Can also be used for 'accurate' decision option.

    _all_feature = (A, X_not_protected_descendent, X_protected_descendent)

    Y_obs, binary, in {0, 1},
    for -1 entries, these agents did not receive approval previously

    [NOTE] Always train an LR to fit (A, X_all) -> Y_obs.
    Applicable to both simulation and real-world data.

    The purpose is to provide an estimate of tier,
    see Agent.estimate_tier().

    """
    LR = LogisticRegression()

    Y_obs = Y_obs.flatten()

    # only train on those whose Y_obs not -1, i.e., Y_obs \in {0, 1}
    feature_usable = _all_feature[-1 != Y_obs, :]

    # in case Y_obs \in {-1, 0}, or \in {-1, 1}
    # just randomize 10% of those 1 to 0 (otherwise cannot fit LR, only 1 class)
    _eps = 0.1
    if 0 == np.count_nonzero(0 == Y_obs):  # there is no default, Y_obs = 1
        print(
            'NOTE: all observed Y_obs is 1, i.e., all previously approved credit are repaid!'
        )
        loc_positive = np.array(np.where(1 == Y_obs)).flatten()
        number_to_flip = int(_eps * len(loc_positive)) + 1
        Y_obs[np.random.choice(loc_positive, size=number_to_flip)] = 0
    elif 0 == np.count_nonzero(1 == Y_obs):  # there is no repayment, Y_obs = 0
        print('NOTE: all observed Y_obs is 0, something is wrong!')
    else:
        pass

    LR.fit(feature_usable, Y_obs[-1 != Y_obs])

    return LR


def LogisticReg_CF_level1(X_not_protected_descendent, Y_obs):
    """Logistic Regressor fitted with X_not_protected_descendent.

    Call on group-level after initializing/registering dynamics for every agent.
    Can also be used for 'CF' decision option (level 1 implementation of CF).

    Only use X_not_protected_descendent, (X_not_protected_descendent) -> Y_obs.

    Y_obs, binary, in {0, 1},
    for -1 entries, these agents did not receive approval previously

    Applicable to both simulation and real-world data.

    [NOTE] Not for the purpose of tier estimation.

    """
    LR = LogisticRegression()

    Y_obs = np.array(Y_obs).flatten()  # (-1, )

    # only train on those whose Y_obs not -1, i.e., Y_obs \in {0, 1}
    feature_usable = X_not_protected_descendent[-1 != Y_obs, :]
    LR.fit(feature_usable, Y_obs[-1 != Y_obs])

    return LR


def dict_agent_feature_label_retrieve_current(dict_agent,
                                              feature_retrieve_option='all'):
    """Retrieve current feature and label from dictionary of Agent.

    Just retrieve current feature and label, not the history record.
    See Agent.feature_label_retrieve()

    Returns:
        df_feature: (A, X_not_protected_descendent, X_protected_descendent)
            np.ndarray size = (n_agent, n_dim_all)
        df_Y_obs: Y_obs np.ndarray size = (n_agent, )
    """
    n_agent = len(dict_agent)
    l_feature = []
    l_Y_obs = []

    for index in range(n_agent):
        feature, label = dict_agent[index].feature_label_retrieve(
            feature_retrieve_option)
        l_feature.append(feature)
        l_Y_obs.append(label)

    return np.array(l_feature), np.array(l_Y_obs)


def dict_agent_decision_retrieve_current(dict_agent):
    """Retrieve current decision from dictionary of Agent.

    Just retrieve current decision, not the history record.

    Returns:
        df_D: D np.ndarray size = (n_agent, )
    """
    n_agent = len(dict_agent)
    l_D = []

    for index in range(n_agent):
        l_D.append(dict_agent[index].D)

    return np.array(l_D)
