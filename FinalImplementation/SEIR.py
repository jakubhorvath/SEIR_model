
from numpy.core.fromnumeric import mean
from constants import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys


#TODO only for debugging purposes 
sys.argv.append("DisplayAll")
sys.argv.append("Hosp")
def inv_probability(vector):
    return np.array([1]*len(vector)) - vector

def total_infected(dict):
    Presymptomatic = dict["Presymp"]
    Asymptomatic = dict["Asymp"]
    Mild = dict["Mild"]
    Severe = dict["Severe"]
    Hospitalized = dict["Hosp"]
    Icu = dict["Icu"]

    total = []
    total_hosp = []
    total_hosp_and_icu = []
    proportion_hosp = []
    
    for timestamp in range(len(Presymptomatic)):
        inf_sum = sum(Presymptomatic[timestamp]) + sum(Asymptomatic[timestamp]) + sum(Mild[timestamp]) + sum(Severe[timestamp]) + sum(Hospitalized[timestamp]) + sum(Icu[timestamp])
        total.append(inf_sum)
        total_hosp.append(sum(Hospitalized[timestamp]))
        total_hosp_and_icu.append(sum(Hospitalized[timestamp]) + sum(Icu[timestamp]))
        proportion_hosp.append((sum(Hospitalized[timestamp]) + sum(Icu[timestamp]))/inf_sum)
        
    return total, total_hosp, total_hosp_and_icu, proportion_hosp


def model(variables, t, R0, h):
    
    S = np.array(variables[0:3])           # 0
    E = np.array(variables[3:6])           # 1
    I_Presymp = np.array(variables[6:9])   # 2
    I_Asymp = np.array(variables[9:12])    # 3
    I_Mild = np.array(variables[12:15])    # 4
    I_Severe = np.array(variables[15:18])  # 5
    I_Hosp = np.array(variables[18:21])    # 6
    I_Icu = np.array(variables[21:24])     # 7
    
    Infected = sum(I_Asymp + I_Presymp + I_Mild + I_Severe)

    # stays the same
    Susceptible = - (R0 / (CITIZENS_CZECH * T_inf)) * S * Infected
    
    # stays the same 
    Exposed = (R0 / (CITIZENS_CZECH * T_inf)) * S * Infected - (E / T_inc)
    
    #Presymptomatic = E / T_inc - PRESYMP_PERIOD * I_Presymp
    
    Presymptomatic = (E / T_inc) - (I_Presymp / (T_inf/1.5))
    
    Asymptomatic = PRESYMP_PERIOD * ASYMPTOMATIC_CASES_PROPORTION * I_Presymp - AVG_INFECTIOUS_PERIOD_ASYMP * I_Asymp
    
    Mild = PRESYMP_PERIOD * inv_probability(ASYMPTOMATIC_CASES_PROPORTION) * I_Presymp - (SYMPTOMATIC_TO_SEVERE * SYMPTOMATIC_TO_HOSPITALIZED + inv_probability(SYMPTOMATIC_TO_SEVERE) * AVG_INFECTIOUS_PERIOD_MILD) * I_Mild
    
    Severe = SYMPTOMATIC_TO_SEVERE * SYMPTOMATIC_TO_HOSPITALIZED * I_Mild - SYMPTOMATIC_TO_HOSPITALIZED * I_Severe
    
    Hospitalized = SYMPTOMATIC_TO_HOSPITALIZED * inv_probability(HOSPITALIZED_TO_ICU) * I_Severe - (HOSPITALIZED_RECOVERY_RATE * inv_probability(FATALITY_RATE) + HOSPITAL_STAY_HOSPITALIZED * FATALITY_RATE) * I_Hosp
    
    Icu = SYMPTOMATIC_TO_HOSPITALIZED * HOSPITALIZED_TO_ICU * I_Severe - (ICU_RECOVERY_RATE * inv_probability(FATALITY_RATE) + HOSPITAL_STAY_ICU * FATALITY_RATE) * I_Icu

    Dead = HOSPITAL_STAY_HOSPITALIZED * FATALITY_RATE * I_Hosp + HOSPITAL_STAY_ICU * FATALITY_RATE * I_Icu
    
    Recovered = AVG_INFECTIOUS_PERIOD_ASYMP * I_Asymp + AVG_INFECTIOUS_PERIOD_MILD * SYMPTOMATIC_TO_SEVERE  * I_Mild + HOSPITALIZED_RECOVERY_RATE * inv_probability(FATALITY_RATE) * I_Hosp  + ICU_RECOVERY_RATE * inv_probability(FATALITY_RATE) * I_Icu
   
    output = []
    output.extend(Susceptible)
    output.extend(Exposed)
    output.extend(Presymptomatic)
    output.extend(Asymptomatic)
    output.extend(Mild)
    output.extend(Severe)
    output.extend(Hospitalized)
    output.extend(Icu)
    output.extend(Dead)
    output.extend(Recovered)
    
    return output


def main():
    hosp_recovered = np.array([0,0,0])
    hosp_dead = np.array([0,0,0])
    icu_recovered = np.array([0,0,0])
    icu_increase = np.array([0,0,0])
    
    time = len(INFECTIOUS_FACTOR)+100
    average_r0 = mean(INFECTIOUS_FACTOR)
    citizens = list(CITIZENS_CR)
    citizens.extend([0]*27)

    counts0 = np.array(citizens)
    counts0[3] = 10
    counts0[4] = 10

    n = time * 10
    t = np.linspace(0, time, n)

    S = np.array([[0]*3 for _ in t])
    S[0] = counts0[0:3]

    E = np.array([[0]*3 for _ in t])
    E[0] = counts0[3:6]

    Presymp = np.array([[0]*3 for _ in t])
    Presymp[0] = counts0[6:9]

    Asymp = np.array([[0]*3 for _ in t])
    Asymp[0] = counts0[9:12]

    Mild = np.array([[0]*3 for _ in t])
    Mild[0] = counts0[12:15]

    Severe = np.array([[0]*3 for _ in t])
    Severe[0] = counts0[15:18]

    Hosp = np.array([[0]*3 for _ in t])
    Hosp[0] = counts0[18:21]

    Icu = np.array([[0]*3 for _ in t])
    Icu[0] = counts0[21:24]

    D = np.array([[0]*3 for _ in t])
    D[0] = counts0[24:27]

    R = np.array([[0]*3 for _ in t])
    R[0] = counts0[27:30]

    dic = {"S": S, "E": E, "Presymp": Presymp, "Asymp": Asymp, "Mild": Mild, "Severe": Severe, "Hosp": Hosp, "Icu": Icu, "D": D, "R": R, "Infected": Presymp+Asymp+Mild+Severe}
    
    
    for i in range(1, n):
        tspan = [t[i-1], t[i]]
        if i//10 < len(INFECTIOUS_FACTOR):
            	counts = odeint(model, counts0, tspan, args=(INFECTIOUS_FACTOR[i//10], 0))
        else:
                counts = odeint(model, counts0, tspan, args=(average_r0, 0))
        S[i] = np.array(counts[1][0:3])  # 0
        E[i] = np.array(counts[1][3:6])  # 1
        Presymp[i] = np.array(counts[1][6:9])  # 2
        Asymp[i] = np.array(counts[1][9:12])  # 3
        Mild[i] = np.array(counts[1][12:15])  # 4
        Severe[i] = np.array(counts[1][15:18])  # 5
        Hosp[i] = np.array(counts[1][18:21])  # 6
        Icu[i] = np.array(counts[1][21:24])  # 7
        D[i] = np.array(counts[1][24:27])  # 8
        R[i] = np.array(counts[1][27:30])  # 9
        counts0 = counts[1]
        print("Calculating iteration: "+str(i))
        
        
    dic_totals = {"S": [sum(t) for t in S], "E": [sum(t) for t in E], "Presymp": [sum(t) for t in Presymp],
                  "Asymp": [sum(t) for t in Asymp], "Mild": [sum(t) for t in Mild], "Severe": [sum(t) for t in Severe],
                  "Hosp": [sum(t) for t in Hosp], "Icu": [sum(t) for t in Icu], "D": [sum(t) for t in D],
                  "R": [sum(t) for t in R]}
    if sys.argv[1] == "CurrentInfected":
        group = dic[sys.argv[2]]
        for i in range(3):
            out = [g[i] for g in group]
            plt.plot(t, out, label=str(i*3)+"-"+str((i+1)*3))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')
        # aktualny pocet nakazenych
    elif sys.argv[1] == "DailyIncrease":
        pass
        # denny prirastok nakazenych
    elif sys.argv[1] == "DisplayAll":
        for group in dic_totals:
            if group != "S" and group != "R" and group != "Icu":
                plt.plot(t, dic_totals[group], label=group)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')            
    elif sys.argv[1] == "InfectedCount":
        pass 
    elif sys.argv[1] == "HospToInfected":
        # take sum of all age categories at each particular moment for hosp and infected
        sum_infected, sum_hosp, sum_hosp_icu, proportion = total_infected(dic)
        #TODO if number don't match, try calculating from sum_hosp_icu
        plt.plot(t, proportion, label="Hospitalized to Infected")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Proportion')
        # podiel hospitalizovanych a nakazenych
    elif sys.argv[1] == "HospitalizedCount":
        hosp = [sum(g) for g in dic["Hosp"]]
        icu = [sum(g) for g in dic["Icu"]]
        plt.plot(t, hosp, label="Icu count")
        plt.plot(t, icu, label="Hospitalized count")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')
        # aktualny pocet hospitalizovanych
    elif sys.argv[1] == "IcuCount":
        out = [sum(g) for g in dic["Icu"]]
        plt.plot(t, out, label="Icu count")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')
        # aktualny pocet v ICU
    elif sys.argv[1] == "DeathCount":
        out = [sum(g) for g in dic["D"]]
        plt.plot(t, out, label="Death count")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')
        # aktualny pocet umrti
    elif sys.argv[1] == "RecoveredCount":
        out = [sum(g) for g in dic["R"]]
        plt.plot(t, out, label="Recovered count")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('count')
        # pocet vyzdravenych


    plt.show()
    
    
if __name__ == "__main__":
    main()