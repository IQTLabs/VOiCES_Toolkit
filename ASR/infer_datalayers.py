import nemo
import nemo_asr
from nemo_asr.helpers import post_process_predictions
import numpy as np
from nemo_asr.parts.features import WaveformFeaturizer
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
import librosa
import pandas as pd
import torch
from ruamel.yaml import YAML
import os

class AudioInferDataLayer(DataLayerNM):
    """
    A very simple data layer class for programmatic interaction.

    The self.set_signal method sets a value of the waveform that will be output
    by the datalayer in the nemo DAG.
    """
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "audio_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "a_sig_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self, sample_rate):
        """

        Inputs:
            sample_rate - The signal's sampling rate in hz
        """
        super().__init__()
        self._sample_rate = sample_rate
        self.output = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signals):
        """
        This sets the value of the waveforms to transcribe.  It must be updated
        before calling the infer method of the neural_factory.

        Inputs:
            signals - list of arrays where each array contains a waveform to
                transcribe.
        """
        batch_size = len(signals)
        self.signal_shape = np.array([len(signal) for signal in signals])
        max_length = np.max(self.signal_shape)
        self.signal = np.zeros((batch_size,max_length))
        for i in range(batch_size):
            self.signal[i,:self.signal_shape[i]] = np.array(signals[i])
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self
