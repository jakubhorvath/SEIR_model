"""
Module containing constants for the Belgian Pandemic Model
"""
import numpy as np


CITIZENS_BELGIUM = 11400000
# transition rates
EXPOSED_TO_PRESYMP = 0.5  # lambda : choose_from_distrib(np.arange(0.809, 0.901, 0.001), 0.024, 0.853)   # γ
# TODO PRESYMP_TO_ASYMP is not a fitting name for this constant
PRESYMP_TO_ASYMP = 0.31  # lambda : choose_from_distrib(np.arange(3.606, 3.836, 0.001), 0.059, 3.724)     # θ
MILD_TO_SEVERE_RATE = 0.47  # lambda : choose_from_distrib(np.arange(2.965, 3.401, 0.001), 0.112, 3.176)  # ω
PRESYMP_TO_ASYMP_PROB = 0.4
RECOVERY_VECTOR_MILD = 0.64
SEVERE_TO_MILD_PROB = 0.81

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
HOSP_DEATH_PROB = 0.052
ICU_DEATH_PROB = 0.052

INFECTION_RATE_DEFAULT = 2.9
INFECTION_RATE = 2.9  #R0
RECOVERY_RATE = ((ASYMP_RECOVERY + MILD_RECOVERY*RECOVERY_VECTOR_MILD + HOSP_RECOVERY*(1-HOSP_DEATH_PROB)+ ICU_RECOVERY*(1-ICU_DEATH_PROB))/4)#T inf

# death rates
HOSP_DEATH_RATE = 1/7          # τ1
ICU_DEATH_RATE = 1/7           # τ2

SOCIAL_CONTACT_MATRIX_ASYMP = 1.6692


def PROPORTION_FACTOR():

    for i in range(10):
        if not LOCKDOWN_MEASURES:
            proportion_f = INFECTION_RATE / (RECOVERY_RATE * CITIZENS_BELGIUM * SOCIAL_CONTACT_MATRIX_ASYMP)
        else:
            proportion_f = INFECTION_RATE / (RECOVERY_RATE * CITIZENS_BELGIUM * SOCIAL_CONTACT_MATRIX_INTERVENTION)

    return proportion_f


def FOI(asymptomatic_infected, symptomatic_infected):    # λ
    """
    Calculates the force of infection denoted as delta

    Returns
        force_of_infection: numpy.array object of force infection for each age category
    """
    # calculate FOI for asymptomatic cases

    prop_factor = PROPORTION_FACTOR()
    prop_factor_symp = prop_factor * SYMP_INFECTIOUS_FACTOR
    prop_factor_asymp = prop_factor * ASYMP_INFECTIOUS_FACTOR

    if not LOCKDOWN_MEASURES:
        social_contact = SOCIAL_CONTACT_MATRIX_ASYMP
    else:
        social_contact = SOCIAL_CONTACT_MATRIX_INTERVENTION


    foi_asymp = social_contact * asymptomatic_infected
    foi_symp = social_contact * symptomatic_infected

    return (foi_symp * prop_factor_symp + foi_asymp * prop_factor_asymp)




# matrix of social interactions after intervention Age x Age for ages
# [0,10) [10,20) ... [80,90) [90,+)
SOCIAL_CONTACT_MATRIX_INTERVENTION = 0.5292
