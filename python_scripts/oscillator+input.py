import samna
import samna.dynapse1 as dyn1
import time

import sys
import random
sys.path.insert(1, '/home/class_NI2021/ctxctl_contrib')
from Dynapse1Constants import *
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron
import numpy as np
import matplotlib.pyplot as plt

from utils import gen_param_group_1core, gaussian_recurrent_excitation
from ADM import ADM

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
paramGroupExc.param_map["NPDPIE_THR_F_P"].fine_value = 15 # ampa gain
paramGroupExc.param_map["NPDPIE_TAU_F_P"].coarse_value = 3 # ampa leakage
paramGroupExc.param_map["NPDPIE_TAU_F_P"].fine_value = 80 # ampa leakage

# set parameters of inhibitory population
paramGroupInh = gen_param_group_1core()
paramGroupInh.param_map["NPDPIE_THR_F_P"].coarse_value = 2 # ampa gain
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
exc_offset = 1 # lowest excitatory neuron id
num_exc = 32
num_inh = 8
input_rates = np.linspace(1, 4, 20)
input_phase = np.pi
osc_input_rate = 60
duration = 5
spike_gen_osc_id = 50
spike_gen_in_up_id = 51
spike_gen_in_dn_id = 52
ei_rate = 1 # probability of an excitatory synapse between any pair of excitatory and inhibitory neurons
ie_rate = 1 # probability of an inhibitory synapse between any pair of excitatory and inhibitory neurons
nids_exc = range(exc_offset, num_exc + exc_offset)
nids_inh = range(1, num_inh + 1)


# init a network generator
net_gen = n.NetworkGenerator()

# create spikegens
spikegen_osc = Neuron(chip_id, core_id_exc, spike_gen_osc_id, True)
spikegen_input_up = Neuron(chip_id, core_id_exc, spike_gen_in_up_id, True)
spikegen_input_dn = Neuron(chip_id, core_id_exc, spike_gen_in_dn_id, True)

# create excitatory population, connect them to spikegens
exc_neurons = []
for nid_exc in nids_exc:
    exc_neuron = Neuron(chip_id, core_id_exc, nid_exc)
    net_gen.add_connection(spikegen_osc, exc_neuron, dyn1.Dynapse1SynType.AMPA)
    net_gen.add_connection(spikegen_input_up, exc_neuron, dyn1.Dynapse1SynType.AMPA)
    net_gen.add_connection(spikegen_input_dn, exc_neuron, dyn1.Dynapse1SynType.GABA_B)
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


# get the global neuron IDs of the neurons
monitored_global_nids_exc = [ut.get_global_id(chip_id, core_id_exc, nid_exc) for nid_exc in nids_exc]
monitored_global_nids_inh = [ut.get_global_id(chip_id, core_id_inh, nid_inh) for nid_inh in nids_inh]
monitored_global_nids = monitored_global_nids_exc + monitored_global_nids_inh


# set up oscillator spike generator
spikegen_osc_global_id = ut.get_global_id(chip_id, core_id_exc, spike_gen_osc_id)
fpga_spike_gen = model.get_fpga_spike_gen()
isi_base = 900
repeat_mode=False

spike_count = osc_input_rate * duration 
spike_times_osc = np.linspace(0, duration, spike_count)
indices_osc = [spikegen_osc_global_id]*len(spike_times_osc)
target_chips_osc = [chip_id]*len(indices_osc)

output_rates = []
for input_rate in input_rates:
    print('input signal frequency:', input_rate)
    # set up input spike generators
    input_signal = np.sin(np.arange(0, duration, 1 / 64) * input_rate * 2 * np.pi + input_phase)
    up_spikes, down_spikes = ADM(
            input_signal,
            up_threshold=0.1,
            down_threshold=0.1,
            sampling_rate=64,
            refractory_period=0
    )

    # up
    spikegen_in_up_global_id = ut.get_global_id(chip_id, core_id_exc, spike_gen_in_up_id)
    indices_up = [spikegen_in_up_global_id]*len(up_spikes)
    target_chips_up = [chip_id]*len(indices_up)

    # down
    spikegen_in_dn_global_id = ut.get_global_id(chip_id, core_id_exc, spike_gen_in_dn_id)
    indices_dn = [spikegen_in_dn_global_id]*len(down_spikes)
    target_chips_dn = [chip_id]*len(indices_dn)


    spike_times = np.concatenate([spike_times_osc, up_spikes, down_spikes])
    indices = np.concatenate([indices_osc, indices_up, indices_dn])
    target_chips = np.concatenate([target_chips_osc, target_chips_up, target_chips_dn])
    order = np.argsort(spike_times)
    ut.set_fpga_spike_gen(fpga_spike_gen, spike_times[order], indices[order], target_chips[order], isi_base, repeat_mode)

    # create a graph to monitor the spikes of this neuron
    graph, filter_node, sink_node = ut.create_neuron_select_graph(model, monitored_global_nids)

    # start graph
    graph.start()

    # start spikegens
    fpga_spike_gen.start()

    # ------------ get events -----------
    # clear the output buffer
    sink_node.get_buf()
    # sleep
    time.sleep(duration)
    # get the events accumulated during the past 2 sec
    events = sink_node.get_buf()
    # ------------ get events -----------

    # stop spikegens
    fpga_spike_gen.stop()

    # stop graph
    graph.stop()

    # Add counter for every neuron spiked during recording
    len_events = len(events)
    holder = np.zeros([len_events,2])
    counter = np.zeros(num_exc + num_inh + exc_offset)

    for i, evt in enumerate(events):
        holder[i,:] = [evt.core_id * max(nids_exc) + evt.neuron_id, (evt.timestamp - events[0].timestamp)]
        counter[evt.core_id * max(nids_exc) + evt.neuron_id] += 1
    rates = counter / duration
    output_rates.append(rates[-1])
    print('rates', rates)


    plt.scatter(holder[:,1]/1e3, holder[:,0], marker='|',color='k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title('Oscillator fixed input: {} Hz, Input sine wave: {} Hz'.format(osc_input_rate, round(input_rate, 2), round(input_phase, 2)))
    plt.savefig('./plots/oscillator+input_spikes_' + str(round(osc_input_rate, 2))+ 'Hz_' + str(round(input_rate, 2))+ 'Hz.png')
    plt.close()

plt.plot(input_rates, output_rates)
plt.xlabel('Frequency of input sinusoid [Hz]')
plt.ylabel('Spiking frequency of inhibitory neuron [Hz]')
plt.title('Frequency response of {} Hz oscillator, phase = {}'.format(2.4, input_phase))
plt.savefig('./plots/oscillator+input_spikes+across_frequencies+phase={}.png'.format(round(input_phase, 2)))

# close Dynapse1
ut.close_dynapse1(store, device_name)

