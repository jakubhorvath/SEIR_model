import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import model_constants as const
from sys import argv

# the delta function of time
def sum_infected(variables):
    return variables[2]+variables[3]+variables[4]+variables[5]+variables[6]+variables[7]

def inverse_probability_vector(vector):
    return np.array([1]*10) - vector

def model(variables, t):

    # delta needs to be dependent on the number of infected in time t
    asymptomatic_infected = sum(variables[2:4])
    symptomatic_infected = sum(variables[4:6])

    # dS(t)/dt = -λ(t)S(t)
    S = -const.FOI(asymptomatic_infected, symptomatic_infected) * variables[0]

    # dE(t)/dt = λ(t)S(t) − γE(t)
    E = const.FOI(asymptomatic_infected, symptomatic_infected)*variables[0] - const.EXPOSED_TO_PRESYMP*variables[1]

    # dIpresym(t)/dt = γE(t) − θIpresym(t)
    I_presym = const.EXPOSED_TO_PRESYMP*variables[1] - const.PRESYMP_TO_ASYMP*variables[2]

    # dIasym(t)/dt = = θpI presym (t) − δ 1 I asym(t)
    I_asym = const.PRESYMP_TO_ASYMP*const.PRESYMP_TO_ASYMP_PROB*variables[2] - const.ASYMP_RECOVERY*variables[3] #TODO find out what θ represents

    # dImild(t)/dt = θ(1 − p)Ipresym(t) − {(1 − φ0)ω + φ0 δ2}Imild(t)
    I_mild = const.PRESYMP_TO_ASYMP*(1-(const.PRESYMP_TO_ASYMP_PROB))*variables[2] - (1-(const.RECOVERY_VECTOR_MILD)*const.MILD_TO_SEVERE_RATE + const.RECOVERY_VECTOR_MILD*const.MILD_RECOVERY)*variables[4]

    # dIsev(t)/dt = (1 − φ0)ωImild(t) − ωIsev(t)
    I_sev = (1-const.RECOVERY_VECTOR_MILD)*const.MILD_TO_SEVERE_RATE - const.MILD_TO_SEVERE_RATE*variables[5]

    # dIhosp(t)/dt = ωφ1Isev(t) − δ3(1 − μhosp) + τ1μhospIhosp(t)
    I_hosp = const.MILD_TO_SEVERE_RATE*const.SEVERE_TO_MILD_PROB*variables[5] - (const.HOSP_RECOVERY*(1-const.HOSP_DEATH_PROB)+const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB)*variables[6]

    # dIicu(t)/dt = ω(1 − φ1)Isev(t) − {δ4(1 − μicu) + τ2μicu}Iicu(t)
    I_icu = const.MILD_TO_SEVERE_RATE*(1-const.SEVERE_TO_MILD_PROB)*variables[5] - (const.ICU_RECOVERY*(1-const.ICU_DEATH_PROB)+const.ICU_DEATH_PROB)*variables[7]

    # dD(t)/dt = τ1.μhosp.Ihosp(t) + τ2.μicu.Iicu(t)
    D = const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB*variables[6] + const.ICU_DEATH_RATE*const.ICU_DEATH_PROB*variables[7]

    # dR(t)/dt = δ1.Iasym(t) + δ2.φ0.Imild(t) + δ3(1 − μhosp)Ihosp(t) + δ4(1 − μicu)Iicu(t)
    R = const.ASYMP_RECOVERY*variables[3] + const.MILD_RECOVERY*const.RECOVERY_VECTOR_MILD*variables[4] + const.HOSP_RECOVERY*(1-const.HOSP_DEATH_PROB)*variables[6] + const.ICU_RECOVERY*(1-const.ICU_DEATH_PROB)*variables[7]

          # 0  1      2        3       4      5       6      7    8  9
    return [S, E, I_presym, I_asym, I_mild, I_sev, I_hosp, I_icu, D, R]


def main():
    time_range = 50


    # number of belgian citizens in 2020 split into age categories
    counts0 = np.array([11400000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    counts0[2] = 10  #Presymptomatic
    counts0[3] = 10  #Asymptomatic
    counts0[4] = 3   #Mild

    # for some reason this is *10+1
    n = 401
    t = np.linspace(0, 40, n)

    S = np.empty_like(t)
    S[0] = counts0[0]

    E = np.empty_like(t)
    E[0] = counts0[1]

    Presymp = np.empty_like(t)
    Presymp[0] = counts0[2]

    Asymp = np.empty_like(t)
    Asymp[0] = counts0[3]

    Mild = np.empty_like(t)
    Mild[0] = counts0[4]

    Severe = np.empty_like(t)
    Severe[0] = counts0[5]

    Hosp = np.empty_like(t)
    Hosp[0] = counts0[6]

    Icu = np.empty_like(t)
    Icu[0] = counts0[7]

    R = np.empty_like(t)
    R[0] = counts0[9]

    D = np.empty_like(t)
    D[0] = counts0[8]

    dic = {"S": S, "E": E, "Presymp": Presymp, "Asymp": Asymp, "Mild": Mild, "Severe": Severe, "Hosp": Hosp, "Icu": Icu, "D": D, "R": R, "Infected": Presymp+Asymp+Mild+Severe}

    #Solve ODE
    for i in range(1, n):
        # span for next time step
        tspan = [t[i - 1], t[i]]
        counts = odeint(model, counts0, tspan)

        # store solutions for plotting
        S[i] = counts[1][0]
        E[i] = counts[1][1]
        Presymp[i] = counts[1][2]
        Asymp[i] = counts[1][3]
        Mild[i] = counts[1][4]
        Severe[i] = counts[1][5]
        Hosp[i] = counts[1][6]
        Icu[i] = counts[1][7]
        R[i] = counts[1][9]
        D[i] = counts[1][8]

        counts0 = counts[1]
    groups_to_display = argv[1:]
    for group in groups_to_display:
        plt.plot(t, dic[group], label=group)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('count')
    plt.show()

if __name__ == "__main__":
    main()
