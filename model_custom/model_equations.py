import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import model_constants as const
import sys
import model_utils as util


if len(sys.argv) < 3:
    print("please provide what variable to visualise")
    print("Model [MAX_TIME] and (\"Age\" and Parameter) or (Parameters)2")
    sys.exit(0)
dic = {"S":(0,10), "E":(10,20), "Presymp":(20,30), "Asymp":(30,40), "Mild":(40,50), "Severe":(50,60), "Hosp":(60,70), "Icu":(70,80), "D":(80,90), "R":(90,100), "Infected": (20, 60)}


def model(variables, t, restrictions_time=None, r0_at_time={}):
    """
    Parameters
    variables : array of variables that change with the function

    t : int type variable representing discrete time

    restrictions_time : int type variable representing the time to put the restrictions into action

    R0_at_time : dictionary mapping time to R0
        if a time value is missing, the default R0 of the model is used
    """
    #NOTE considering a simple event where at time t the social contact matrix is switched
    if restrictions_time is not None and t >= restrictions_time:
        const.LOCKDOWN_MEASURES = True
    else:
        const.LOCKDOWN_MEASURES = False

    if t in r0_at_time:
        const.INFECTION_RATE = r0_at_time[t]
    else:
        const.INFECTION_RATE = const.INFECTION_RATE_DEFAULT


    S0 = np.array(variables[0:10])        #0
    E0 = np.array(variables[10:20])       #1
    I_presym0 = np.array(variables[20:30])#2
    I_asym0 = np.array(variables[30:40])  #3
    I_mild0 = np.array(variables[40:50])  #4
    I_sev0 = np.array(variables[50:60])   #5
    I_hosp0 = np.array(variables[60:70])  #6
    I_icu0 = np.array(variables[70:80])   #7
    # this is a different R0 than the infection rate value
    R0 = np.array(variables[80:90])       #8
    D0 = np.array(variables[90:100])      #9

    all_infected_asymp = I_presym0 + I_asym0
    all_infected_symp = I_mild0 + I_sev0
    force_of_infection = const.FOI(all_infected_asymp, all_infected_symp)
    #force_of_infection = const.calculate_FOI_OLD_VERSION(all_infected_asymp, all_infected_symp)
    # dS(t)/dt = -λ(t)S(t)
    S = -force_of_infection * S0

    # dE(t)/dt = λ(t)S(t) − γE(t)
    E = force_of_infection * S0 - const.EXPOSED_TO_PRESYMP * E0

    # dIpresym(t)/dt = γE(t) − θIpresym(t)
    I_presym = const.EXPOSED_TO_PRESYMP * E0 - const.PRESYMP_TO_ASYMP * I_presym0

    # dIasym(t)/dt = = θpI presym (t) − δ 1 I asym(t)
    I_asym = const.PRESYMP_TO_ASYMP * const.PRESYMP_TO_ASYMP_PROB * I_presym0 - const.ASYMP_RECOVERY * I_asym0

    # dImild(t)/dt = θ(1 − p)Ipresym(t) − {(1 − φ0)ω + φ0 δ2}Imild(t)
    I_mild = const.PRESYMP_TO_ASYMP*(util.inverse_probability_vector(const.PRESYMP_TO_ASYMP_PROB))*I_presym0 - (util.inverse_probability_vector(const.RECOVERY_VECTOR_MILD)*const.MILD_TO_SEVERE_RATE + const.RECOVERY_VECTOR_MILD*const.MILD_RECOVERY)*I_mild0

    # dIsev(t)/dt = (1 − φ0)ωImild(t) − ωIsev(t)
    I_sev = util.inverse_probability_vector(const.RECOVERY_VECTOR_MILD)*const.MILD_TO_SEVERE_RATE*I_mild0 - const.MILD_TO_SEVERE_RATE*I_sev0

    # dIhosp(t)/dt = ωφ1Isev(t) − δ3(1 − μhosp) + τ1μhospIhosp(t)
    I_hosp = const.MILD_TO_SEVERE_RATE*const.SEVERE_TO_MILD_PROB*I_sev0 - (const.HOSP_RECOVERY*util.inverse_probability_vector(const.HOSP_DEATH_PROB)+const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB)*I_hosp0

    # dIicu(t)/dt = ω(1 − φ1)Isev(t) − {δ4(1 − μicu) + τ2μicu}Iicu(t)
    I_icu = const.MILD_TO_SEVERE_RATE*util.inverse_probability_vector(const.SEVERE_TO_MILD_PROB)*I_sev0 - (const.ICU_RECOVERY*util.inverse_probability_vector(const.ICU_DEATH_PROB)+const.ICU_DEATH_PROB)*I_icu0

    # dD(t)/dt = τ1.μhosp.Ihosp(t) + τ2.μicu.Iicu(t)
    D = const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB*I_hosp0 + const.ICU_DEATH_RATE*const.ICU_DEATH_PROB*I_icu0

    # dR(t)/dt = δ1.Iasym(t) + δ2.φ0.Imild(t) + δ3(1 − μhosp)Ihosp(t) + δ4(1 − μicu)Iicu(t)
    R = const.ASYMP_RECOVERY*I_asym0 + const.MILD_RECOVERY*const.RECOVERY_VECTOR_MILD*I_mild0 + const.HOSP_RECOVERY*util.inverse_probability_vector(const.HOSP_DEATH_PROB)*I_hosp0 + const.ICU_RECOVERY*util.inverse_probability_vector(const.ICU_DEATH_PROB)*I_icu0


    output = []
    output.extend(S)
    output.extend(E)
    output.extend(I_presym)
    output.extend(I_asym)
    output.extend(I_mild)
    output.extend(I_sev)
    output.extend(I_hosp)
    output.extend(I_icu)
    output.extend(D)
    output.extend(R)

    return np.array(output)
           # 0  1      2        3       4      5       6      7    8  9
    #return [S, E, I_presym, I_asym, I_mild, I_sev, I_hosp, I_icu, D, R]


def main():
    # number of belgian citizens in 2020 split into age categories
    array = list(const.CITIZENS_BELGIUM)
    array.extend([0]*90)
    variables0 = np.array(array)

    variables0[20] = 10
    variables0[25] = 10
    result = []
    t = np.array([i for i in range(int(sys.argv[1]))])
    # This line generates a random array of R0 values
    r0_at_time = util.generate_random_r0_around(max(t))
    for time in t:
        variables_changed = model(variables0, time)
        variables0 = np.add(variables0, variables_changed, casting="unsafe")
        result.append(variables0)

    # if told to plot age groups
    if sys.argv[2] == "Age":
        group = sys.argv[3]
        fr, to = dic[group]
        for age_group in range(fr, to):
            plt.plot(t, [r[age_group] for r in result], label=str(age_group-fr)+"0+ yrs")
    # if told to plot different infectious groups with ages merged
    else:
        groups_to_display = sys.argv[2:]
        for group in groups_to_display:
            fr, to = dic[group]
            plt.plot(t, [sum(r[fr:to]) for r in result], label=group)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.savefig(sys.argv[2] + "_" + sys.argv[3] + "graph.png")
    plt.show()


if __name__ == "__main__":
    main()
