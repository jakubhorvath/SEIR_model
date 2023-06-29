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



def model(variables, t, r0, restriction_measures):
    """
    Parameters
    variables : array of variables that change with the function

    t : int type variable representing discrete time

    restrictions_time : int type variable representing the time to put the restrictions into action

    R0_at_time : dictionary mapping time to R0
        if a time value is missing, the default R0 of the model is used
    """
    # NOTE considering a simple event where at time t the social contact matrix is switched'
    const.LOCKDOWN_MEASURES = restriction_measures

    const.INFECTION_RATE = r0

    S0 = np.array(variables[0:10])          # 0
    E0 = np.array(variables[10:20])         # 1
    I_presym0 = np.array(variables[20:30])  # 2
    I_asym0 = np.array(variables[30:40])    # 3
    I_mild0 = np.array(variables[40:50])    # 4
    I_sev0 = np.array(variables[50:60])     # 5
    I_hosp0 = np.array(variables[60:70])    # 6
    I_icu0 = np.array(variables[70:80])     # 7
    # this is a different R0 than the infection rate value
    D0 = np.array(variables[80:90])         # 8
    R0 = np.array(variables[90:100])        # 9

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
    I_mild = const.PRESYMP_TO_ASYMP*(util.inverse_probability_vector(const.PRESYMP_TO_ASYMP_PROB))*I_presym0 - (util.inverse_probability_vector(const.SYMPTOMATIC_TO_MILD)*const.MILD_TO_SEVERE_RATE + const.SYMPTOMATIC_TO_MILD*const.MILD_RECOVERY)*I_mild0

    # dIsev(t)/dt = (1 − φ0)ωImild(t) − ωIsev(t)
    I_sev = util.inverse_probability_vector(const.SYMPTOMATIC_TO_MILD)*const.MILD_TO_SEVERE_RATE*I_mild0 - const.MILD_TO_SEVERE_RATE*I_sev0

    # dIhosp(t)/dt = ωφ1Isev(t) − δ3(1 − μhosp) + τ1μhospIhosp(t)
    I_hosp = const.MILD_TO_SEVERE_RATE*const.HOSP_NOT_TO_ICU*I_sev0 - (const.HOSP_RECOVERY*util.inverse_probability_vector(const.HOSP_DEATH_PROB)+const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB)*I_hosp0

    # dIicu(t)/dt = ω(1 − φ1)Isev(t) − {δ4(1 − μicu) + τ2μicu}Iicu(t)
    I_icu = const.MILD_TO_SEVERE_RATE*util.inverse_probability_vector(const.HOSP_NOT_TO_ICU)*I_sev0 - (const.ICU_RECOVERY*util.inverse_probability_vector(const.ICU_DEATH_PROB)+const.ICU_DEATH_PROB*const.ICU_DEATH_RATE)*I_icu0

    # dD(t)/dt = τ1.μhosp.Ihosp(t) + τ2.μicu.Iicu(t)
    D = const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB*I_hosp0 + const.ICU_DEATH_RATE*const.ICU_DEATH_PROB*I_icu0

    # dR(t)/dt = δ1.Iasym(t) + δ2.φ0.Imild(t) + δ3(1 − μhosp)Ihosp(t) + δ4(1 − μicu)Iicu(t)
    R = const.ASYMP_RECOVERY*I_asym0 + const.MILD_RECOVERY*const.SYMPTOMATIC_TO_MILD*I_mild0 + const.HOSP_RECOVERY*util.inverse_probability_vector(const.HOSP_DEATH_PROB)*I_hosp0 + const.ICU_RECOVERY*util.inverse_probability_vector(const.ICU_DEATH_PROB)*I_icu0


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
    time = int(sys.argv[1])
    citizens = list(const.CITIZENS_BELGIUM)
    citizens.extend([0]*90)

    counts0 = np.array(citizens)
    counts0[20] = 10
    counts0[25] = 10

    n = time * 10 + 1
    t = np.linspace(0, time, n)

    S = np.array([[0]*10 for _ in t])
    S[0] = counts0[0:10]

    E = np.array([[0]*10 for _ in t])
    E[0] = counts0[10:20]

    Presymp = np.array([[0]*10 for _ in t])
    Presymp[0] = counts0[20:30]

    Asymp = np.array([[0]*10 for _ in t])
    Asymp[0] = counts0[30:40]

    Mild = np.array([[0]*10 for _ in t])
    Mild[0] = counts0[40:50]

    Severe = np.array([[0]*10 for _ in t])
    Severe[0] = counts0[50:60]

    Hosp = np.array([[0]*10 for _ in t])
    Hosp[0] = counts0[60:70]

    Icu = np.array([[0]*10 for _ in t])
    Icu[0] = counts0[70:80]

    D = np.array([[0]*10 for _ in t])
    D[0] = counts0[80:90]

    R = np.array([[0]*10 for _ in t])
    R[0] = counts0[90:100]

    dic = {"S": S, "E": E, "Presymp": Presymp, "Asymp": Asymp, "Mild": Mild, "Severe": Severe, "Hosp": Hosp, "Icu": Icu, "D": D, "R": R, "Infected": Presymp+Asymp+Mild+Severe}
    #r0_at_time = util.generate_random_r0_around(t)
    r0 = np.zeros(n)
    restrictions = np.empty(n)
    for i in range(len(r0)):
        r0[i] = 2.9
        restrictions[i] = False
    #restrictions[300:] = True


    for i in range(1, n):
        tspan = [t[i-1], t[i]]

        counts = odeint(model, counts0, tspan, args=(r0[i], restrictions[i]))

        # store solutions for plotting
        S[i] = counts[1][0:10]
        E[i] = counts[1][10:20]
        Presymp[i] = counts[1][20:30]
        Asymp[i] = counts[1][30:40]
        Mild[i] = counts[1][40:50]
        Severe[i] = counts[1][50:60]
        Hosp[i] = counts[1][60:70]
        Icu[i] = counts[1][70:80]
        D[i] = counts[1][80:90]
        R[i] = counts[1][90:100]

        counts0 = counts[1]
        print("Calculating iteration: "+str(i))
    dic = {"S": [timestamp for timestamp in S], "E": [timestamp for timestamp in E],
           "Presymp": [timestamp for timestamp in Presymp], "Asymp": [timestamp for timestamp in Asymp],
           "Mild": [timestamp for timestamp in Mild], "Severe": [timestamp for timestamp in Severe],
           "Hosp": [timestamp for timestamp in Hosp], "Icu": [timestamp for timestamp in Icu],
           "D": [timestamp for timestamp in D], "R": [timestamp for timestamp in R]}
    group_sums = {"S": [sum(timestamp) for timestamp in S], "E": [sum(timestamp) for timestamp in E],
                  "Presymp": [sum(timestamp) for timestamp in Presymp],
                  "Asymp": [sum(timestamp) for timestamp in Asymp],
                  "Mild": [sum(timestamp) for timestamp in Mild], "Severe": [sum(timestamp) for timestamp in Severe],
                  "Hosp": [sum(timestamp) for timestamp in Hosp], "Icu": [sum(timestamp) for timestamp in Icu],
                  "D": [sum(timestamp) for timestamp in D], "R": [sum(timestamp) for timestamp in R]}
    if sys.argv[2] == "Aged":

        group = dic[sys.argv[3]]
        for i in range(10):
            out = [g[i] for g in group]
            plt.plot(t, out, label=str(i*10)+"-"+str((i+1)*10))

    elif sys.argv[2] == "DailyIncrease":
        axes = plt.gca()
        if sys.argv[3] == "D":
            axes.set_ylim([0, 400])
        group_total = group_sums[sys.argv[3]]
        increase_list = [0]
        for i in range(1, len(group_total)):
            day_to_day_increase = group_total[i] - group_total[i-1]
            if day_to_day_increase > 0:
                increase_list.append(day_to_day_increase)
            else:
                increase_list.append(0)
        plt.plot(t, increase_list, label="daily_increase" + sys.argv[3])

    elif sys.argv[2] == "Proportions":

        group = dic[sys.argv[3]]
        proportional = []
        for timestamp in group:
            proportional.append(timestamp / citizens[0:10])
        for i in range(10):
            out = [g[i] for g in proportional]
            plt.plot(t, out, label=str(i*10)+"-"+str((i+1)*10))
    else:

        groups_to_display = sys.argv[2:]
        for group in groups_to_display:
            plt.plot(t, group_sums[group], label=group)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('count')
    plt.savefig("./" + sys.argv[2] + "_" + sys.argv[3] + ".png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
