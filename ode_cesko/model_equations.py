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


#TODO save the number of hospitalized patients that died and recovered
#hosp_recovered = const.HOSP_RECOVERY*util.inverse_probability_vector(const.HOSP_DEATH_PROB)*I_hosp0
#hosp_dead = const.HOSP_DEATH_RATE*const.HOSP_DEATH_PROB*I_hosp0
hosp_recovered = [[0,0,0]]
hosp_dead = [[0,0,0]]
icu_recovered = [[0,0,0]]
icu_increase = [[0,0,0]]
def model(variables, t, r0, hmm):
    """
    Parameters
    variables : array of variables that change with the function

    t : int type variable representing discrete time

    restrictions_time : int type variable representing the time to put the restrictions into action

    R0_at_time : dictionary mapping time to R0
        if a time value is missing, the default R0 of the model is used
    """
    # NOTE considering a simple event where at time t the social contact matrix is switched'

    const.INFECTION_RATE = r0

    S0 = np.array(variables[0:3])          # 0
    E0 = np.array(variables[3:6])         # 1
    I_presym0 = np.array(variables[6:9])  # 2
    I_asym0 = np.array(variables[9:12])    # 3
    I_mild0 = np.array(variables[12:15])    # 4
    I_sev0 = np.array(variables[15:18])     # 5
    I_hosp0 = np.array(variables[18:21])    # 6
    I_icu0 = np.array(variables[21:24])     # 7

    all_infected_asymp = I_presym0 + I_asym0
    all_infected_symp = I_mild0 + I_sev0
    force_of_infection = const.FOI(all_infected_asymp, all_infected_symp, r0)
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


def main():
    #time = int(sys.argv[1])
    #r0 = [2.5, 2.4954761904761904, 2.490952380952381, 2.486428571428571, 2.4819047619047616, 2.477380952380952, 2.4728571428571424, 2.468333333333333, 2.4638095238095232, 2.4592857142857136, 2.454761904761904, 2.4502380952380944, 2.445714285714285, 2.4411904761904752, 2.4366666666666656, 2.432142857142856, 2.4276190476190465, 2.423095238095237, 2.4185714285714273, 2.4140476190476177, 2.409523809523808, 2.4049999999999985, 2.400476190476189, 2.3959523809523793, 2.3914285714285697, 2.38690476190476, 2.3823809523809505, 2.377857142857141, 2.3733333333333313, 2.3688095238095217, 2.364285714285712, 2.3597619047619025, 2.355238095238093, 2.3507142857142833, 2.3461904761904737, 2.341666666666664, 2.3371428571428545, 2.332619047619045, 2.3280952380952353, 2.3235714285714257, 2.319047619047616, 2.3145238095238065, 2.309999999999997, 2.3054761904761873, 2.3009523809523778, 2.296428571428568, 2.2919047619047586, 2.287380952380949, 2.2828571428571394, 2.2783333333333298, 2.27380952380952, 2.2692857142857106, 2.264761904761901, 2.2602380952380914, 2.255714285714282, 2.251190476190472, 2.2466666666666626, 2.242142857142853, 2.2376190476190434, 2.233095238095234, 2.228571428571424, 2.2240476190476146, 2.219523809523805, 2.2149999999999954, 2.210476190476186, 2.2059523809523762, 2.2014285714285666, 2.196904761904757, 2.1923809523809474, 2.187857142857138, 2.1833333333333282, 2.1788095238095186, 2.174285714285709, 2.1697619047618995, 2.16523809523809, 2.1607142857142803, 2.1561904761904707, 2.151666666666661, 2.1471428571428515, 2.142619047619042, 2.1380952380952323, 2.1335714285714227, 2.129047619047613, 2.1245238095238035, 2.119999999999994, 2.1154761904761843, 2.1109523809523747, 2.106428571428565, 2.1019047619047555, 2.097380952380946, 2.0928571428571363, 2.0883333333333267, 2.083809523809517, 2.0792857142857075, 2.074761904761898, 2.0702380952380883, 2.0657142857142787, 2.061190476190469, 2.0566666666666595, 2.05214285714285, 2.0476190476190403, 2.0430952380952307, 2.038571428571421, 2.0340476190476116, 2.029523809523802, 2.0249999999999924, 2.0204761904761828, 2.015952380952373, 2.0114285714285636, 2.006904761904754, 2.0023809523809444, 1.9978571428571348, 1.9933333333333252, 1.9888095238095156, 1.984285714285706, 1.9797619047618964, 1.9752380952380868, 1.9707142857142772, 1.9661904761904676, 1.961666666666658, 1.9571428571428484, 1.9526190476190388, 1.9480952380952292, 1.9435714285714196, 1.93904761904761, 1.9345238095238004, 1.9299999999999908, 1.9254761904761812, 1.9209523809523716, 1.916428571428562, 1.9119047619047524, 1.9073809523809429, 1.9028571428571333, 1.8983333333333237, 1.893809523809514, 1.8892857142857045, 1.8847619047618949, 1.8802380952380853, 1.8757142857142757, 1.871190476190466, 1.8666666666666565, 1.8621428571428469, 1.8576190476190373, 1.8530952380952277, 1.848571428571418, 1.8440476190476085, 1.839523809523799, 1.8349999999999893, 1.8304761904761797, 1.8259523809523701, 1.8214285714285605, 1.816904761904751, 1.8123809523809413, 1.8078571428571317, 1.8033333333333221, 1.7988095238095125, 1.794285714285703, 1.7897619047618933, 1.7852380952380837, 1.7807142857142741, 1.7761904761904646, 1.771666666666655, 1.7671428571428454, 1.7626190476190358, 1.7580952380952262, 1.7535714285714166, 1.749047619047607, 1.7445238095237974, 1.7399999999999878, 1.7354761904761782, 1.7309523809523686, 1.726428571428559, 1.7219047619047494, 1.7173809523809398, 1.7128571428571302, 1.7083333333333206, 1.703809523809511, 1.6992857142857014, 1.6947619047618918, 1.6902380952380822, 1.6857142857142726, 1.681190476190463, 1.6766666666666534, 1.6721428571428438, 1.6676190476190342, 1.6630952380952246, 1.658571428571415, 1.6540476190476054, 1.6495238095237958, 1.6449999999999863, 1.6404761904761767, 1.635952380952367, 1.6314285714285575, 1.6269047619047479, 1.6223809523809383, 1.6178571428571287, 1.613333333333319, 1.6088095238095095, 1.6042857142856999, 1.5997619047618903, 1.5952380952380807, 1.590714285714271, 1.5861904761904615, 1.581666666666652, 1.5771428571428423, 1.5726190476190327, 1.568095238095223, 1.5635714285714135, 1.559047619047604, 1.5545238095237943, 1.5499999999999847, 1.5454761904761751, 1.5409523809523655, 1.536428571428556, 1.5319047619047463, 1.5273809523809367, 1.5228571428571271, 1.5183333333333175, 1.513809523809508, 1.5092857142856984, 1.5047619047618888, 1.5002380952380792, 1.4957142857142696, 1.49119047619046, 1.4866666666666504, 1.4821428571428408, 1.4776190476190312, 1.4730952380952216, 1.468571428571412, 1.4640476190476024, 1.4595238095237928, 1.4549999999999832, 1.4504761904761736, 1.445952380952364, 1.4414285714285544, 1.4369047619047448, 1.4323809523809352, 1.4278571428571256, 1.423333333333316, 1.4188095238095064, 1.4142857142856968, 1.4097619047618872, 1.4052380952380776, 1.400714285714268, 1.3961904761904584, 1.3916666666666488, 1.3871428571428392, 1.3826190476190297, 1.37809523809522, 1.3735714285714105, 1.3690476190476009, 1.3645238095237913, 1.3599999999999817, 1.355476190476172, 1.3509523809523625, 1.3464285714285529, 1.3419047619047433, 1.3373809523809337, 1.332857142857124, 1.3283333333333145, 1.323809523809505, 1.3192857142856953, 1.3147619047618857, 1.310238095238076, 1.3057142857142665, 1.301190476190457, 1.2966666666666473, 1.2921428571428377, 1.2876190476190281, 1.2830952380952185, 1.278571428571409, 1.2740476190475993, 1.2695238095237897, 1.2649999999999801, 1.2604761904761705, 1.255952380952361, 1.2514285714285514, 1.2469047619047418, 1.2423809523809322, 1.2378571428571226, 1.233333333333313, 1.2288095238095034, 1.2242857142856938, 1.2197619047618842, 1.2152380952380746, 1.210714285714265, 1.2061904761904554, 1.2016666666666458, 1.1971428571428362, 1.1926190476190266, 1.188095238095217, 1.1835714285714074, 1.1790476190475978, 1.1745238095237882, 1.1699999999999786, 1.165476190476169, 1.1609523809523594, 1.1564285714285498, 1.1519047619047402, 1.1473809523809306, 1.142857142857121, 1.1383333333333114, 1.1338095238095018, 1.1292857142856922, 1.1247619047618826, 1.120238095238073, 1.1157142857142635, 1.1111904761904539, 1.1066666666666443, 1.1021428571428347, 1.097619047619025, 1.0930952380952155, 1.0885714285714059, 1.0840476190475963, 1.0795238095237867, 1.074999999999977, 1.0704761904761675, 1.065952380952358, 1.0614285714285483, 1.0569047619047387, 1.052380952380929, 1.0478571428571195, 1.04333333333331, 1.0388095238095003, 1.0342857142856907, 1.0297619047618811, 1.0252380952380715, 1.020714285714262, 1.0161904761904523, 1.0116666666666427, 1.0071428571428331, 1.0026190476190235, 0.998095238095214, 0.9935714285714046, 0.9890476190475951, 0.9845238095237856, 0.9799999999999761, 0.9754761904761666, 0.9709523809523571, 0.9664285714285477, 0.9619047619047382, 0.9573809523809287, 0.9528571428571192, 0.9483333333333097, 0.9438095238095002, 0.9392857142856907, 0.9347619047618813, 0.9302380952380718, 0.9257142857142623, 0.9211904761904528, 0.9166666666666433, 0.9121428571428338, 0.9076190476190243, 0.9030952380952149, 0.8985714285714054, 0.8940476190475959, 0.8895238095237864, 0.8849999999999769, 0.8804761904761674, 0.875952380952358, 0.8714285714285485, 0.866904761904739, 0.8623809523809295, 0.85785714285712, 0.8533333333333105, 0.848809523809501, 0.8442857142856915, 0.8397619047618821, 0.8352380952380726, 0.8307142857142631, 0.8261904761904536, 0.8216666666666441, 0.8171428571428346, 0.8126190476190251, 0.8080952380952157, 0.8035714285714062, 0.7990476190475967, 0.7945238095237872, 0.7899999999999777, 0.7854761904761682, 0.7809523809523587, 0.7764285714285493, 0.7719047619047398, 0.7673809523809303, 0.7628571428571208, 0.7583333333333113, 0.7538095238095018, 0.7492857142856924, 0.7447619047618829, 0.7402380952380734, 0.7357142857142639, 0.7311904761904544, 0.7266666666666449, 0.7221428571428354, 0.717619047619026, 0.7130952380952165, 0.708571428571407, 0.7040476190475975, 0.699523809523788, 0.6949999999999785, 0.690476190476169, 0.6859523809523596, 0.6814285714285501, 0.6769047619047406, 0.6723809523809311, 0.6678571428571216, 0.6633333333333121, 0.6588095238095026, 0.6542857142856932, 0.6497619047618837, 0.6452380952380742, 0.6407142857142647, 0.6361904761904552, 0.6316666666666457, 0.6271428571428362, 0.6226190476190268, 0.6180952380952173, 0.6135714285714078, 0.6090476190475983, 0.6045238095237888]
    time = len(const.R0)
    citizens = list(const.CITIZENS_CR)
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

        counts = odeint(model, counts0, tspan, args=(const.R0[i//10], 0))

        hosp_recovered.append(const.HOSP_RECOVERY * util.inverse_probability_vector(const.HOSP_DEATH_PROB) * Hosp[i-1])
        hosp_dead.append(const.HOSP_DEATH_RATE * const.HOSP_DEATH_PROB * Hosp[i-1])
        icu_recovered.append(const.ICU_RECOVERY*util.inverse_probability_vector(const.ICU_DEATH_PROB)*Icu[i-1])
        # store solutions for plotting
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
        for i in range(3):
            out = [g[i] for g in group]
            plt.plot(t, out, label=str(i*3)+"-"+str((i+1)*3))

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
        for i in range(3):
            out = [g[i] for g in proportional]
            plt.plot(t, out, label=str(i)+"-"+str((i+1)*10))

    elif sys.argv[2] == "CzechHosp":
        out_dead = [sum(i) for i in hosp_dead]
        out_recovered = [sum(i) for i in hosp_recovered]
        icu_rec = [sum(i) for i in icu_recovered]
        #plt.plot(t, out_dead, label="Hospital deaths")
        #plt.plot(t, out_recovered, label="Hospital recoverings")
        plt.plot(t, icu_recovered, label="Icu Recover")

    else:

        groups_to_display = sys.argv[2:]
        for group in groups_to_display:
            plt.plot(t, group_sums[group], label=group)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('count')
    #plt.savefig(sys.argv[2] + "_" + sys.argv[3] + ".png", dpi=200)
    
    plt.show()

if __name__ == "__main__":
    main()
