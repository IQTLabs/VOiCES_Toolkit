import nemo
import nemo_asr
from nemo_asr.helpers import post_process_predictions, word_error_rate
import numpy as np
from nemo_asr.parts.features import WaveformFeaturizer
from .infer_datalayers import AudioInferDataLayer
from nemo.core.neural_modules import NeuralModule
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
import librosa
import pandas as pd
import torch

class JasperInference:
    def __init__(self,model_definition,use_cpu=True,encoder_module=None,decoder_module=None):
        if use_cpu:
            self.neural_factory = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.CPU,
                                                       backend=nemo.core.Backend.PyTorch)
        else:
            self.neural_factory = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch)
        self.model_definition = model_definition
        self.vocab = self.model_definition['labels']
        self.build_components(encoder_module=encoder_module,decoder_module=decoder_module)
        self.build_dag()


    def build_components(self,encoder_module=None,decoder_module=None):
        self.data_layer = AudioInferDataLayer(sample_rate=self.model_definition['sample_rate'])
        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate=self.model_definition['sample_rate'],
            **self.model_definition["AudioToMelSpectrogramPreprocessor"])
        if encoder_module is not None:
            # Pass in an already instantiated neural module for the encoder
            assert isinstance(encoder_module,NeuralModule), 'encoder is not a neural module'
            self.encoder = encoder_module
        else:
            self.encoder = nemo_asr.JasperEncoder(
                feat_in=self.model_definition['AudioToMelSpectrogramPreprocessor']['features'],
                **self.model_definition['JasperEncoder'])
        if decoder_module is not None:
            # Pass in an already instantiated neural module for the decoder
            assert isinstance(decoder_module,NeuralModule), 'decoder is not a neural module'
            self.decoder = decoder_module
        else:
            self.decoder = nemo_asr.JasperDecoderForCTC(
                feat_in=self.model_definition["JasperEncoder"]["jasper"][-1]["filters"],num_classes=len(self.vocab))
        self.greedy_decoder = nemo_asr.GreedyCTCDecoder()
        # TODO:  Add support for N-gram LM model


    def build_dag(self):
        self.audio_signal, self.audio_signal_len = self.data_layer()
        self.processed_signal, self.p_length = self.data_preprocessor(input_signal=self.audio_signal,
                                                                      length=self.audio_signal_len)
        self.encoded, self.encoded_len = self.encoder(audio_signal=self.processed_signal,length=self.p_length)
        self.log_probs = self.decoder(encoder_output=self.encoded)
        self.predictions = self.greedy_decoder(log_probs=self.log_probs)
        # TODO:  Add support for N-gram LM model

    def restore_weights(self,encoder_weight_path=None,decoder_weight_path=None):
        if encoder_weight_path:
            self.encoder_weight_path = encoder_weight_path
            self.encoder.restore_from(encoder_weight_path)
        if decoder_weight_path:
            self.decoder_weight_path = decoder_weight_path
            self.decoder.restore_from(decoder_weight_path)
    def infer(self,filepaths=None,waveforms=None,return_logits=False):
        if filepaths is not None:
            waveforms = []
            for filepath in filepaths:
                waveform,sr = librosa.core.load(filepath,sr=self.model_definition['sample_rate'])
                waveforms.append(waveform)
            self.data_layer.set_signal(waveforms)
        elif waveforms is not None:
            self.data_layer.set_signal(waveforms)
        tensors_to_evaluate = [self.predictions]
        if return_logits:
            tensors_to_evaluate.append(self.log_probs)
        evaluated_tensors = self.neural_factory.infer(tensors_to_evaluate,verbose=False)
        greedy_transcript = post_process_predictions(evaluated_tensors[0],self.vocab)
        result_dict = {'greedy prediction':evaluated_tensors[0]}
        result_dict['greedy transcript']=greedy_transcript
        if return_logits:
            result_dict['logits']=evaluated_tensors[1]
        return result_dict
