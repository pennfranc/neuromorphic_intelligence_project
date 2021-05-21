import samna.dynapse1 as dyn1
import numpy as np

def gaussian(x, mu, sigma = 1.0):
    '''
    Calculate the Gaussian value given a position x, the Guassian center mu and sigma.
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def gaussian_recurrent_excitation(size, baseWeight, sigma=3.0, self_exc=True):
    '''
    Author: Jingyue
    For a pre neuron, use its neuron id as the mean of Gaussian, calculate the value at
    every other neuron id which is the weight of the synapse.
    
    size: number of neurons
    baseWeight: maximum weight at the center
    sigma: Gaussian parameter
    self_exc: if you want to have self-excitation or not
    
    return:
        output_i: pre neuron ids
        output_j: post neuron ids
    '''
    output_i = []
    output_j = []
    floatWeightMr = np.zeros((size,size))
    
    for pre in range(size):
        mu = pre
        for post in range(size):
            if pre == post and self_exc == False:
                continue
            else:
                # for each neuron id, calculate the synapse weight at every other neurons
                weight = int(baseWeight * gaussian(float(post), float(mu), float(sigma)) )
                floatWeightMr[pre][post] = weight

                # add weight connections between pre and post
                for k in range(weight):
                    output_i.append(pre)
                    output_j.append(post)
            
    print(floatWeightMr[16])
        
    return output_i, output_j  

def gen_param_group_1core():
    paramGroup = dyn1.Dynapse1ParameterGroup()
    # THR, gain factor of neurons
    paramGroup.param_map["IF_THR_N"].coarse_value = 5
    paramGroup.param_map["IF_THR_N"].fine_value = 80

    # refactory period of neurons
    paramGroup.param_map["IF_RFR_N"].coarse_value = 4
    paramGroup.param_map["IF_RFR_N"].fine_value = 128

    # leakage of neurons
    paramGroup.param_map["IF_TAU1_N"].coarse_value = 4
    paramGroup.param_map["IF_TAU1_N"].fine_value = 80

    # turn off tau2
    paramGroup.param_map["IF_TAU2_N"].coarse_value = 7
    paramGroup.param_map["IF_TAU2_N"].fine_value = 255

    # turn off DC
    paramGroup.param_map["IF_DC_P"].coarse_value = 0
    paramGroup.param_map["IF_DC_P"].fine_value = 0

    # leakage of AMPA
    paramGroup.param_map["NPDPIE_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_F_P"].fine_value = 80

    # gain of AMPA
    paramGroup.param_map["NPDPIE_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_F_P"].fine_value = 80

    # weight of AMPA
    # NOTE: !!!!!!!!!!!!!!!!!!!!!!! remember to set the weight of AMPA !!!!!!!!!!!!!!!!!!!!!!!
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 6 
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 40

    # leakage of NMDA
    paramGroup.param_map["NPDPIE_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_S_P"].fine_value = 80

    # gain of NMDA
    paramGroup.param_map["NPDPIE_THR_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_S_P"].fine_value = 80

    # weight of NMDA
    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 0

    # leakage of GABA_A (shunting)
    paramGroup.param_map["NPDPII_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_F_P"].fine_value = 80

    # gain of GABA_A (shunting)
    paramGroup.param_map["NPDPII_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_THR_F_P"].fine_value = 80

    # weight of GABA_A (shunting)
    paramGroup.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_INH_F_N"].fine_value = 0

    # leakage of GABA_B
    paramGroup.param_map["NPDPII_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_S_P"].fine_value = 80

    # gain of GABA_B
    paramGroup.param_map["NPDPII_THR_S_P"].coarse_value = 5
    paramGroup.param_map["NPDPII_THR_S_P"].fine_value = 255

    # weight of GABA_B
    paramGroup.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 7
    paramGroup.param_map["PS_WEIGHT_INH_S_N"].fine_value = 255

    # other advanced parameters
    paramGroup.param_map["IF_NMDA_N"].coarse_value = 0
    paramGroup.param_map["IF_NMDA_N"].fine_value = 0

    paramGroup.param_map["IF_AHTAU_N"].coarse_value = 4
    paramGroup.param_map["IF_AHTAU_N"].fine_value = 80

    paramGroup.param_map["IF_AHTHR_N"].coarse_value = 0
    paramGroup.param_map["IF_AHTHR_N"].fine_value = 0

    paramGroup.param_map["IF_AHW_P"].coarse_value = 0
    paramGroup.param_map["IF_AHW_P"].fine_value = 0

    paramGroup.param_map["IF_CASC_N"].coarse_value = 0
    paramGroup.param_map["IF_CASC_N"].fine_value = 0

    paramGroup.param_map["PULSE_PWLK_P"].coarse_value = 4
    paramGroup.param_map["PULSE_PWLK_P"].fine_value = 106

    paramGroup.param_map["R2R_P"].coarse_value = 3
    paramGroup.param_map["R2R_P"].fine_value = 85

    paramGroup.param_map["IF_BUF_P"].coarse_value = 3
    paramGroup.param_map["IF_BUF_P"].fine_value = 80

    return paramGroup

