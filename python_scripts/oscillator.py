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

from utils import gen_param_group_1core, gaussian_recurrent_excitation

class Oscillator:

    exc_offset = 3
    inh_offset = 1
    net_gen = None

    def __init__(
        self,
        spikegen_input_up,
        spikegen_input_dn,
        osc_input_rate=120,
        num_exc=32, num_inh=8,
        ei_rate=1, ie_rate=1
    ):
        self.chip_id = 0
        self.core_id_exc = 0
        self.core_id_inh = 1
        self.osc_input_rate = osc_input_rate

        exc_offset, inh_offset = Oscillator.get_offsets()
        print(exc_offset, inh_offset)

        self.nids_exc = range(exc_offset, num_exc + exc_offset)
        self.nids_inh = range(inh_offset, num_inh + inh_offset)

        # set up input driving oscillator
        self.spike_gen_osc_id = exc_offset + num_exc
        self.osc_input_rate = osc_input_rate
        spikegen_osc = Neuron(self.chip_id, self.core_id_exc, self.spike_gen_osc_id, True)

        # create excitatory population, connect them to spikegens
        exc_neurons = []
        for nid_exc in self.nids_exc:
            exc_neuron = Neuron(self.chip_id, self.core_id_exc, nid_exc)
            Oscillator.net_gen.add_connection(spikegen_osc, exc_neuron, dyn1.Dynapse1SynType.AMPA)
            Oscillator.net_gen.add_connection(spikegen_input_up, exc_neuron, dyn1.Dynapse1SynType.AMPA)
            Oscillator.net_gen.add_connection(spikegen_input_dn, exc_neuron, dyn1.Dynapse1SynType.GABA_B)
            exc_neurons.append(exc_neuron)

        # create inhibitory population and ei connections
        inh_neurons =[]
        for nid_inh in self.nids_inh:
            inh_neuron = Neuron(self.chip_id, self.core_id_inh, nid_inh)
            for exc_neuron in exc_neurons:
                if random.uniform(0, 1) <= ei_rate:
                    Oscillator.net_gen.add_connection(exc_neuron, inh_neuron, dyn1.Dynapse1SynType.AMPA)
            inh_neurons.append(inh_neuron)

        # create ee connections
        output_i, output_j = gaussian_recurrent_excitation(num_exc, 6, 1, self_exc=False) 
        for i, j in zip(output_i, output_j):
            Oscillator.net_gen.add_connection(exc_neurons[i], exc_neurons[j], dyn1.Dynapse1SynType.AMPA)


        # create ie connections
        for inh_neuron in inh_neurons:
            for exc_neuron in exc_neurons:
                if random.uniform(0, 1) <= ie_rate:
                    Oscillator.net_gen.add_connection(inh_neuron, exc_neuron, dyn1.Dynapse1SynType.GABA_B)

        Oscillator.update_offsets(num_exc, num_inh)

    @classmethod
    def update_offsets(cls, num_exc, num_inh):
        cls.exc_offset += (num_exc + 1)
        cls.inh_offset += num_inh

    @classmethod 
    def get_offsets(cls):
        return cls.exc_offset, cls.inh_offset
        
    def get_ids(self):
        return self.nids_exc, self.nids_inh, self.spike_gen_osc_id

    def get_osc_input_spikes(self, duration):
        spikegen_osc_global_id = ut.get_global_id(self.chip_id, self.core_id_exc, self.spike_gen_osc_id)
        spike_count = self.osc_input_rate * duration 
        spike_times_osc = np.linspace(0, duration, spike_count)
        indices_osc = [spikegen_osc_global_id]*len(spike_times_osc)
        target_chips_osc = [self.chip_id]*len(indices_osc)

        return spike_times_osc, indices_osc, target_chips_osc


def aggregate_osc_input_spikes(oscillators, duration):
    osc_spike_times = []
    osc_indices = []
    osc_target_chips = []

    for oscillator in oscillators:
        spike_times, indices, target_chips = oscillator.get_osc_input_spikes(duration)
        osc_spike_times.append(spike_times),
        osc_indices.append(indices)
        osc_target_chips.append(target_chips)

    return np.concatenate(osc_spike_times), np.concatenate(osc_indices), np.concatenate(osc_target_chips)

def get_all_nids(oscillators):
    nids_exc = []
    nids_inh = []
    for oscillator in oscillators:
        nids_exc += oscillator.nids_exc
        nids_inh += oscillator.nids_inh
    return nids_exc, nids_inh