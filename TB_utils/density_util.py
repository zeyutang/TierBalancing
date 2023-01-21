# # --> density related util
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import random
from TB_utils.data_util import data_fetch

random.seed(42)
np.random.seed(42)


def inverse_cdf(domain_axis, pdf_axis):
    """Get CDF from PDF (unnormalized is fine).

    How to sample? E.g., cdf(np.random.random_sample(size=(n, )))
    """
    assert not np.amin(pdf_axis) < 0, 'Value error in PDF'

    # (unnormalized) PDF to CDF
    cdf_axis = cumtrapz(pdf_axis, x=domain_axis)
    # normalize
    cdf_axis = cdf_axis / np.amax(cdf_axis)
    # make sure zero start
    cdf_axis = np.insert(cdf_axis, 0, 0)
    cdf_axis = interp1d(cdf_axis, domain_axis, kind='linear')

    return cdf_axis


def simulation_tier_cdf(option):
    """Generate CDF for simulation.

    For the purpose of better illustration, let A \in {0, 1}.
    A = 1 is the advantaged group, H \in (0, 1].

    [NOTE] double check shape of PDF

    Available options:
        'uniform'
            H | A = 0 ~ U[0.1, 0.7]
            H | A = 1 ~ U[0.3, 0.9]
        'truncate_gaussian'
            H | A = 0 ~ N(.3, .5)
            H | A = 1 ~ N(.6, .5)
        'trigonometric'
            H | A = 0 ~ cos(h + .5)
            H | A = 1 ~ sin(h + .5)

    """
    domain_axis = np.linspace(1e-6, 1 - 1e-6, 1000)
    init_cdf = dict()

    # define PDF (not normalized), will normalize CDF later
    if 'uniform' == option:
        domain_axis_disadv = np.linspace(0.1, 0.7, 1000)
        domain_axis_adv = np.linspace(0.3, 0.9, 1000)
        # shared
        pdf_axis = np.ones_like(domain_axis_adv)

        # dict
        init_cdf[0] = inverse_cdf(domain_axis_disadv, pdf_axis)
        init_cdf[1] = inverse_cdf(domain_axis_adv, pdf_axis)

    elif 'truncate_gaussian' == option:
        pdf_axis_disadv = np.exp(-0.5 * np.square((domain_axis - 0.3) / 0.5))
        pdf_axis_adv = np.exp(-0.5 * np.square((domain_axis - 0.6) / 0.5))

        # dict
        init_cdf[0] = inverse_cdf(domain_axis, pdf_axis_disadv)
        init_cdf[1] = inverse_cdf(domain_axis, pdf_axis_adv)

    elif 'trigonometric' == option:
        pdf_axis_disadv = np.cos(domain_axis + 0.5)
        pdf_axis_adv = np.sin(domain_axis + 0.5)

        # dict
        init_cdf[0] = inverse_cdf(domain_axis, pdf_axis_disadv)
        init_cdf[1] = inverse_cdf(domain_axis, pdf_axis_adv)

    else:
        raise Exception('Invalid initialization option')

    return init_cdf


def predefine_feature_generate(protected_feature, tier):
    """Generate X from (A, H).

    For 'simulation' and 'creditscore'. This func is re-used
    for different T = t (stable module assumption).

    (A, H) -> (X_not_protected_descendent, X_protected_descendent).

    Output X_ (np.ndarray).

    """

    X_child_H = 1.5 * tier + np.random.uniform(-0.1, 0.1)
    X_child_H_and_A = tier + 0.8 * \
        protected_feature + np.random.uniform(-0.1, 0.1)
    X_child_A = protected_feature + np.random.normal(0, 0.1)

    # output
    X_not_protected_descendent = np.array([
        X_child_H,
    ])
    X_protected_descendent = np.hstack((X_child_H_and_A, X_child_A))
    return X_not_protected_descendent, X_protected_descendent


def creditscore_tier_cdf():
    """Preprocess 'creditscore' risk cdf data.

    Protected feature: race,
        0 for Black, 1 for Asian, 2 for Hispanic, 3 for White.

    If needed, use totals.csv to estimate marginal of A.
    Otherwise, for the purpose of balanced sample among groups,
    just assume A is uniform distributed.

    Use transrisk_cdf_by_race_ssa.csv to calculate
    cumulative distribution function for H | A, so that
    one can sample from it as necessary.

    """
    # fetch data
    data_fetch('creditscore')

    with open('transrisk_cdf_by_race_ssa.csv', 'r') as f:
        df_cdf = pd.read_csv(f)

    # re-name the column
    df_cdf = df_cdf.rename(columns={'Non- Hispanic white': 'White'})

    # re-order columns
    df_cdf = df_cdf.reindex(['Score', 'Black', 'Asian', 'Hispanic', 'White'],
                            axis=1)

    # normalize
    df_cdf = (df_cdf / 100.).to_numpy()

    # add a 0 start for CDF
    df_cdf[0, 0] = 1e-6  # avoid zero-out issue
    df_cdf = np.insert(df_cdf, 0, 0, axis=0)

    # column 0 act as np.linspace
    init_cdf = dict()

    for i in range(4):
        init_cdf[i] = interp1d(df_cdf[:, i + 1], df_cdf[:, 0], kind='linear')

    return init_cdf
