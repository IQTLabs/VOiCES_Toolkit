import numpy as np
import os
import argparse
import pandas as pd
from .JasperModels import JasperInference
from ruamel.yaml import YAML
import pesq
import librosa
from nemo_asr.helpers import post_process_predictions, word_error_rate
import tqdm

def process_item(item,dataset_root,jasper_model,sample_rate=16000):
    ground_truth = [item['transcript']]
    result_dict = {'query_name':item['query_name']}
    result_dict['ground_truth']=item['transcript']

    noisy_filepath = os.path.join(dataset_root,item['filename'])
    clean_filepath = os.path.join(dataset_root,item['source'])

    noisy_waveform,_ = librosa.load(noisy_filepath,sr=sample_rate)
    clean_waveform,_ = librosa.load(clean_filepath,sr=sample_rate)

    pesq_nb = pesq.pesq(16000,clean_waveform,noisy_waveform,'nb')
    pesq_wb = pesq.pesq(16000,clean_waveform,noisy_waveform,'wb')
    result_dict['pesq nb'] = pesq_nb
    result_dict['pesq wb'] = pesq_wb

    noisy_result = jasper_model.infer(waveforms=[noisy_waveform])
    clean_result = jasper_model.infer(waveforms=[clean_waveform])

    result_dict['noisy transcript'] = noisy_result['greedy transcript'][0]
    result_dict['clean transcript'] = clean_result['greedy transcript'][0]
    result_dict['noisy wer'] = word_error_rate([result_dict['noisy transcript']],[result_dict['ground_truth']])
    result_dict['clean wer'] = word_error_rate([result_dict['clean transcript']],[result_dict['ground_truth']])

    return result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='DATASET_ROOT',help='VOiCES dataset root',
                        default='none',type=str)
    parser.add_argument('-i',dest='INDEX_PATH',help='Target directory for index files',
                        default='none',type=str)
    parser.add_argument('-e',dest='ENCODER_PATH',help='path to encoder weights',
                        default='none',type=str)
    parser.add_argument('-d',dest='DECODER_PATH',help='path to decoder weights',
                        default='none',type=str)
    parser.add_argument('-c',dest='CONFIG',help='path to config yaml',
                        default='none',type=str)
    parser.add_argument('-o',dest='OUTPUT',help='out filepath',
                        default='none',type=str)
    args = parser.parse_args()

    #load up the dataset
    df = pd.read_csv(args.INDEX_PATH)
    df = df[df['source_length']==df['noisy_length']]

    #load up the model configuration
    yaml = YAML(typ="safe")
    with open(args.CONFIG) as f:
        model_definition = yaml.load(f)
    vocab = model_definition['labels']

    #build the jasper inference model
    jasper = JasperInference(model_definition)
    jasper.restore_weights(encoder_weight_path=args.ENCODER_PATH,decoder_weight_path=args.DECODER_PATH)
    result_list = []
    for i in tqdm.tqdm(range(len(df))):
        item = df.iloc[i]
        result = process_item(item,args.DATASET_ROOT,jasper)
        result_list.append(result)
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(args.OUTPUT)
