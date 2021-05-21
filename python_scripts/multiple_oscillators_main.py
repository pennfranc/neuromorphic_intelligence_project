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

from utils import gen_param_group_1core
from oscillator import Oscillator, aggregate_osc_input_spikes, get_all_nids
from ADM import ADM

# open DYNAP-SE1 board to get Dynapse1Model
device_name = "my_dynapse1"
# change the port numbers to not have conflicts with other groups
store = ut.open_dynapse1(device_name, gui=False, sender_port=10111, receiver_port=10222)
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
exc_offset = 3 # lowest excitatory neuron id
inh_offset = 1 # lowest inhibitory neuron id
num_exc = 32
num_inh = 8
input_rates = np.linspace(0, 4, 20)
input_phase = np.pi
osc_input_rates = [60, 80, 100, 120]
duration = 5
spike_gen_osc_id = 50
spike_gen_in_up_id = 1
spike_gen_in_dn_id = 2


# init a network generator
net_gen = n.NetworkGenerator()

# create spikegens
spikegen_input_up = Neuron(chip_id, core_id_exc, spike_gen_in_up_id, True)
spikegen_input_dn = Neuron(chip_id, core_id_exc, spike_gen_in_dn_id, True)

Oscillator.net_gen = net_gen

oscillators = [Oscillator(
    spikegen_input_up,
    spikegen_input_dn,
    osc_input_rate=osc_input_rate
) for osc_input_rate in osc_input_rates]
osc_spike_times, osc_indices, osc_target_chips = aggregate_osc_input_spikes(oscillators, duration)

net_gen = Oscillator.net_gen

# print network
net_gen.print_network()
# make a dynapse1config using the network
new_config = net_gen.make_dynapse1_configuration()
# apply the configuration
model.apply_configuration(new_config)


# get the global neuron IDs of the neurons
nids_exc, nids_inh = get_all_nids(oscillators)
monitored_global_nids_exc = [ut.get_global_id(chip_id, core_id_exc, nid_exc) for nid_exc in nids_exc]
monitored_global_nids_inh = [ut.get_global_id(chip_id, core_id_inh, nid_inh) for nid_inh in nids_inh]
monitored_global_nids = monitored_global_nids_exc + monitored_global_nids_inh


# set up spike generator
fpga_spike_gen = model.get_fpga_spike_gen()
isi_base = 900
repeat_mode=False

output_rates = np.zeros((len(oscillators), len(input_rates)))
for input_rate_idx, input_rate in enumerate(input_rates):
    print('input signal frequency:', input_rate)
    # set up input spike generators

    if input_rate > 0:
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

    
        spike_times = np.concatenate([osc_spike_times, up_spikes, down_spikes])
        indices = np.concatenate([osc_indices, indices_up, indices_dn])
        target_chips = np.concatenate([osc_target_chips, target_chips_up, target_chips_dn])

        order = np.argsort(spike_times)
        ut.set_fpga_spike_gen(fpga_spike_gen, spike_times[order], indices[order], target_chips[order], isi_base, repeat_mode)
    else:
        order = np.argsort(osc_spike_times)
        ut.set_fpga_spike_gen(fpga_spike_gen, osc_spike_times[order], osc_indices[order], osc_target_chips[order], isi_base, repeat_mode)

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
    max_exc_index = max(oscillators[-1].nids_exc)
    max_inh_index = max(oscillators[-1].nids_inh)
    len_events = len(events)
    holder = np.zeros([len_events,2])
    counter_exc = np.zeros(max_exc_index + 1)
    counter_inh = np.zeros(max_inh_index + 1)

    for i, evt in enumerate(events):
        holder[i,:] = [evt.core_id * max_exc_index + evt.neuron_id, (evt.timestamp - events[0].timestamp)]
        if evt.core_id == 0:
            counter_exc[evt.neuron_id] += 1
        else:
            counter_inh[evt.neuron_id] += 1
    rates_exc = counter_exc / duration
    rates_inh = counter_inh / duration
    for oscillator_idx, oscillator in enumerate(oscillators):
        print(oscillator.nids_inh)
        output_rates[oscillator_idx, input_rate_idx] = np.mean(rates_inh[oscillator.nids_inh])
        print('oscillator {} - output rate = {}'.format(oscillator_idx, np.mean(rates_inh[oscillator.nids_inh])))


for oscillator_idx, oscillator in enumerate(oscillators):
    plt.plot(input_rates, output_rates[oscillator_idx, :])
    plt.xlabel('Frequency of input sinusoid [Hz]')
    plt.ylabel('Spiking frequency of inhibitory neuron [Hz]')
    plt.title('Frequency response of oscillator, phase = {}'.format(input_phase))
    plt.savefig('../plots/oscillator-{}+input_spikes+across_frequencies+phase={}.png'.format(oscillator_idx, round(input_phase, 2)))
    plt.close()

# close Dynapse1
ut.close_dynapse1(store, device_name)

