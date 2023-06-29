"""
Module containing constants for the Belgian Pandemic Model
"""
import numpy as np
import model_utils as util


CITIZENS_BELGIUM = np.array([1260000, 1300000, 1400000, 1480000, 1500000, 1580000, 1340000, 900000, 530000, 110000])
CASES_INITIAL_BELGIUM = [8, 8, 9, 9, 9, 9, 9, 8, 8, 8] # total 85 in belgium on march 12 2020
# transition rates
EXPOSED_TO_PRESYMP = 0.5  # lambda : choose_from_distrib(np.arange(0.809, 0.901, 0.001), 0.024, 0.853)   # γ
# TODO PRESYMP_TO_ASYMP is not a fitting name for this constant
PRESYMP_TO_ASYMP = 0.31  # lambda : choose_from_distrib(np.arange(3.606, 3.836, 0.001), 0.059, 3.724)     # θ
MILD_TO_SEVERE_RATE = 0.47  # lambda : choose_from_distrib(np.arange(2.965, 3.401, 0.001), 0.112, 3.176)  # ω
PRESYMP_TO_ASYMP_PROB = np.array([0.94, 0.90, 0.84, 0.61, 0.49, 0.21, 0.02, 0.02, 0.02, 0.02])    # p
RECOVERY_VECTOR_MILD = np.array([0.98, 0.98, 0.79, 0.79, 0.67, 0.67, 0.50, 0.35, 0.32, 0.32])    # Φ0
SEVERE_TO_MILD_PROB = np.array([1, 1, 0.85, 0.85, 0.76, 0.76, 0.73, 0.69, 0.74, 0.74])           # Φ1

# Infectious factor
# q = beta
ASYMP_INFECTIOUS_FACTOR = 1.17  # q asymp
SYMP_INFECTIOUS_FACTOR = 0.6   # q symp

LOCKDOWN_MEASURES = False

# recovery rates
ASYMP_RECOVERY = 1/1.367  # lambda : choose_from_distrib(np.arange(1.259, 1.477, 0.001), 0.056, 1.367)      # δ1
MILD_RECOVERY = 1/1.367  # lambda : choose_from_distrib(np.arange(1.259, 1.477, 0.001), 0.056, 1.367)       # δ2
HOSP_RECOVERY = 1/11.5            # δ3
ICU_RECOVERY = 1/13.3             # δ4



# death probability
HOSP_DEATH_PROB = np.array([0.000094, 0.00022, 0.00091, 0.0018, 0.004, 0.013, 0.046, 0.098, 0.18, 0.18])        # μ hosp
ICU_DEATH_PROB = np.array([0.000094, 0.00022, 0.00091, 0.0018, 0.004, 0.013, 0.046, 0.098, 0.18, 0.18])         # μ icu

INFECTION_RATE_DEFAULT = 2.9
INFECTION_RATE = 2.9  #R0
RECOVERY_RATE = ((ASYMP_RECOVERY*np.array([1]*10) + MILD_RECOVERY*RECOVERY_VECTOR_MILD + HOSP_RECOVERY*util.inverse_probability_vector(HOSP_DEATH_PROB)+ ICU_RECOVERY*util.inverse_probability_vector(ICU_DEATH_PROB))/4)#T inf

# death rates
HOSP_DEATH_RATE = 1/7          # τ1
ICU_DEATH_RATE = 1/7           # τ2

SOCIAL_CONTACT_MATRIX_ASYMP = np.array([
                        [7.71, 1.2, 0.93, 2.03, 1.03, 0.89, 0.65, 0.34, 0.27, 0.28],      # 0-10
                        [1.23, 10.38, 2.25, 2.01, 2.19, 0.79, 0.42, 0.44, 0.11, 0.88],     # 10-20
                        [1.03, 2.44, 6.06, 3.35, 2.95, 2.74, 0.8, 0.49, 0.85, 1.64],      # 20-30
                        [2.4, 2.32, 3.57, 5.53, 4.38, 3.18, 1.75, 0.94, 0.75, 1.91],      # 30-40
                        [1.36, 2.83, 3.51, 4.9, 5.56, 3.72, 2.15, 1.89, 0.99, 1.08],      # 40-50
                        [1.06, 0.93, 2.95, 3.22, 3.36, 3.82, 1.89, 10.8, 1.09, 1.27],     # 50-60
                        [0.59, 0.37, 0.65, 1.35, 1.48, 1.44, 1.96, 1.19, 0.74, 1.03],     # 60-70
                        [0.24, 0.3, 0.31, 0.55, 1, 0.63, 0.92, 1.5, 0.56, 0.61],          # 70-80
                        [0.11, 0.04, 0.31, 0.26, 0.3, 0.37, 0.33, 0.32, 1, 1.37],         # 80-90
                        [0.1, 0.04, 0.07, 0.08, 0.04, 0.05, 0.05, 0.04, 0.16, 0.97]])     # 90+


def PROPORTION_FACTOR():
    proportion_f = []

    for i in range(10):
        if not LOCKDOWN_MEASURES:
            q = INFECTION_RATE / (RECOVERY_RATE[i] * sum(CITIZENS_BELGIUM) * sum(SOCIAL_CONTACT_MATRIX_ASYMP[i]))
        else:
            q = INFECTION_RATE / (RECOVERY_RATE[i] * sum(CITIZENS_BELGIUM) * sum(SOCIAL_CONTACT_MATRIX_INTERVENTION[i]))

        proportion_f.append(q)

    return np.array(proportion_f)


def FOI(asymptomatic_infected, symptomatic_infected):    # λ
    """
    Calculates the force of infection denoted as delta

    Returns
        force_of_infection: numpy.array object of force infection for each age category
    """
    # calculate FOI for asymptomatic cases
    foi_asymp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    foi_symp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    prop_factor = PROPORTION_FACTOR()
    prop_factor_symp = prop_factor * SYMP_INFECTIOUS_FACTOR
    prop_factor_asymp = prop_factor * ASYMP_INFECTIOUS_FACTOR

    for age_category1 in range(0, 10):
        # check whether to use the lockdown social contact matrix or the normal one
        if not LOCKDOWN_MEASURES:
            social_contact = SOCIAL_CONTACT_MATRIX_ASYMP[age_category1]
        else:
            social_contact = SOCIAL_CONTACT_MATRIX_INTERVENTION[age_category1]

        for age_category2 in range(0, 10):
            foi_asymp[age_category1] += social_contact[age_category2] * asymptomatic_infected[age_category2]
            foi_symp[age_category1] += social_contact[age_category2] * symptomatic_infected[age_category2]

    return (foi_symp * prop_factor_symp + foi_asymp * prop_factor_asymp)


def calculate_FOI_OLD_VERSION(asymptomatic_infected, symptomatic_infected):    # λ
    """
    Calculates the force of infection denoted as delta

    Returns
        force_of_infection: numpy.array object of force infection for each age category
    """
    # calculate FOI for asymptomatic cases
    foi_asymp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    foi_symp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for age_category1 in range(0, 10):
        if not LOCKDOWN_MEASURES:
            social_contact = SOCIAL_CONTACT_MATRIX_ASYMP[age_category1]
        else:
            social_contact = SOCIAL_CONTACT_MATRIX_INTERVENTION[age_category1]

        for age_category2 in range(0, 10):
            foi_asymp[age_category1] += social_contact[age_category2] * asymptomatic_infected[age_category2] * ASYMP_INFECTIOUS_FACTOR
            foi_symp[age_category1] += social_contact[age_category2] * symptomatic_infected[age_category2] * SYMP_INFECTIOUS_FACTOR

    return (foi_symp + foi_asymp) / sum(CITIZENS_BELGIUM)


# matrix of social interactions after intervention Age x Age for ages
# [0,10) [10,20) ... [80,90) [90,+)
SOCIAL_CONTACT_MATRIX_INTERVENTION = np.array([
                        [1.13, 0.43, 0.32, 1.13, 0.43, 0.32, 0.28, 0.15, 0.04, 0.11],
                        [0.44, 1.57, 0.59, 0.53, 1.3, 0.29, 0.16, 0.2, 0.06, 0.21],
                        [0.36, 0.64, 1.52, 0.71, 0.79, 1.03, 0.26, 0.15, 0.51, 1.36],
                        [1.34, 0.61, 0.75, 1.53, 0.95, 0.78, 0.57, 0.3, 0.44, 1.51],
                        [0.56, 1.69, 0.94, 1.06, 1.57, 0.86, 0.54, 0.62, 0.46, 0.85],
                        [0.38, 0.34, 1.11, 0.79, 0.77, 1.38, 0.58, 0.32, 0.49, 0.7],
                        [0.26, 0.14, 0.21, 0.44, 0.37, 0.44, 0.82, 0.41, 0.26, 0.48],
                        [0.1, 0.14, 0.09, 0.18, 0.33, 0.18, 0.32, 0.69, 0.25, 0.5],
                        [0.02, 0.02, 0.18, 0.15, 0.14, 0.16, 0.11, 0.14, 0.85, 1.34],
                        [0.01, 0.01, 0.06, 0.06, 0.03, 0.03, 0.03, 0.03, 0.16, 0.97]])
