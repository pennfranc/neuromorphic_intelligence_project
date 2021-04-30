import samna
import samna.dynapse1 as dyn1
import time

import sys
sys.path.insert(1, '/home/class_NI2021/ctxctl_contrib')
from Dynapse1Constants import *
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron
import numpy as np
import random

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
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 100

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
    paramGroup.param_map["NPDPII_TAU_S_P"].fine_value = 255

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


# open DYNAP-SE1 board to get Dynapse1Model
device_name = "my_dynapse1"
# change the port numbers to not have conflicts with other groups
store = ut.open_dynapse1(device_name, gui=False, sender_port=11111, receiver_port=12222)
model = getattr(store, device_name)

# chip and core ids
chip_id = 0
core_id_exc = 0
core_id_inh = 1


# set parameters of excitatory population
paramGroupExc = gen_param_group_1core()
paramGroupExc.param_map["NPDPIE_THR_F_P"].coarse_value = 3 # ampa gain
paramGroupExc.param_map["NPDPIE_THR_F_P"].fine_value = 80 # ampa gain
paramGroupExc.param_map["NPDPIE_TAU_F_P"].coarse_value = 4 # ampa leakage
paramGroupExc.param_map["NPDPIE_TAU_F_P"].fine_value = 80 # ampa leakage

# set parameters of inhibitory population
paramGroupInh = gen_param_group_1core()
paramGroupInh.param_map["NPDPIE_THR_F_P"].coarse_value = 4 # ampa gain
paramGroupInh.param_map["NPDPIE_THR_F_P"].fine_value = 80 # ampa gain
paramGroupInh.param_map["NPDPIE_TAU_F_P"].coarse_value = 4 # ampa leakage
paramGroupInh.param_map["NPDPIE_TAU_F_P"].fine_value = 80 # ampa leakage

# set parameters of chips and cores
for chip in range(4):
    for core in range(4):
        model.update_parameter_group(gen_param_group_1core(), chip, core)
model.update_parameter_group(paramGroupExc, chip_id, core_id_exc)
model.update_parameter_group(paramGroupInh, chip_id, core_id_inh)

# define the network parameters
exc_offset = 5 # lowest excitatory neuron id
num_exc = 32
num_inh = 8
input_rate = 100
duration = 2
spike_gen_id = 50
ei_rate = 1 # probability of an excitatory synapse between any pair of excitatory and inhibitory neurons
ie_rate = 1 # probability of an inhibitory synapse between any pair of excitatory and inhibitory neurons
nids_exc = range(exc_offset, num_exc + exc_offset)
nids_inh = range(1, num_inh + 1)


# init a network generator
net_gen = n.NetworkGenerator()

# create spikegen
spikegen = Neuron(chip_id, core_id_exc, spike_gen_id, True)

# create excitatory population, connect them to spikegen
exc_neurons = []
for nid_exc in nids_exc:
    exc_neuron = Neuron(chip_id, core_id_exc, nid_exc)
    net_gen.add_connection(spikegen, exc_neuron, dyn1.Dynapse1SynType.AMPA)
    exc_neurons.append(exc_neuron)

# create inhibitory population and ei connections
inh_neurons =[]
for nid_inh in nids_inh:
    inh_neuron = Neuron(chip_id, core_id_inh, nid_inh)
    for exc_neuron in exc_neurons:
        if random.uniform(0, 1) <= ei_rate:
            net_gen.add_connection(exc_neuron, inh_neuron, dyn1.Dynapse1SynType.AMPA)
    inh_neurons.append(inh_neuron)

# create ee connections
output_i, output_j = gaussian_recurrent_excitation(num_exc, 6, 1, self_exc=False) 
for i, j in zip(output_i, output_j):
    net_gen.add_connection(exc_neurons[i], exc_neurons[j], dyn1.Dynapse1SynType.AMPA)


# create ie connections
for inh_neuron in inh_neurons:
    for exc_neuron in exc_neurons:
        if random.uniform(0, 1) <= ie_rate:
            net_gen.add_connection(inh_neuron, exc_neuron, dyn1.Dynapse1SynType.GABA_B)


# print network
net_gen.print_network()
# make a dynapse1config using the network
new_config = net_gen.make_dynapse1_configuration()
# apply the configuration
model.apply_configuration(new_config)





# set up Poisson spike generator
spikegen_global_id = ut.get_global_id(chip_id, core_id_exc, spike_gen_id)
poisson_gen = model.get_poisson_gen()
poisson_gen.set_chip_id(chip_id)
poisson_gen.write_poisson_rate_hz(spikegen_global_id, input_rate)

# get the global neuron IDs of the neurons
monitored_global_nids_exc = [ut.get_global_id(chip_id, core_id_exc, nid_exc) for nid_exc in nids_exc]
monitored_global_nids_inh = [ut.get_global_id(chip_id, core_id_inh, nid_inh) for nid_inh in nids_inh]
monitored_global_nids = monitored_global_nids_exc + monitored_global_nids_inh

# create a graph to monitor the spikes of this neuron
graph, filter_node, sink_node = ut.create_neuron_select_graph(model, monitored_global_nids)

# start graph
graph.start()

# start the stimulus
poisson_gen.start()

# ------------ get events -----------
# clear the output buffer
sink_node.get_buf()
# sleep
time.sleep(duration)
# get the events accumulated during the past 2 sec
events = sink_node.get_buf()
# ------------ get events -----------

# stop the stimulus
poisson_gen.stop()
# stop graph
graph.stop()

# Add counter for every neuron spiked during recording
print(monitored_global_nids)
len_events = len(events)
holder = np.zeros([len_events,2])
counter = np.zeros(num_exc + num_inh + exc_offset)
for i, evt in enumerate(events):
    holder[i,:] = [evt.core_id * max(nids_exc) + evt.neuron_id, (evt.timestamp - events[0].timestamp)]
    counter[evt.core_id * max(nids_exc) + evt.neuron_id] += 1

np.save('./oscillator_holder', holder)
np.save('./oscillator_counter', counter)

# close Dynapse1
ut.close_dynapse1(store, device_name)

