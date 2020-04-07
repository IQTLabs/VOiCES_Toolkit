import numpy as np
import os
import argparse
import pandas as pd
from ASR.JasperModels import JasperInference
from ruamel.yaml import YAML
import pesq
import librosa
from nemo_asr.helpers import post_process_predictions, word_error_rate
import tqdm

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def process_batch(item_batch,dataset_root,jasper_model,sample_rate=16000):
    result_batch = []
    noisy_waveform_list = []
    clean_waveform_list = []

    for item in item_batch:
        result_dict = {'query_name':item['query_name']}
        result_dict['ground_truth']=item['transcript']

        noisy_filepath = os.path.join(dataset_root,item['filename'])
        clean_filepath = os.path.join(dataset_root,item['source'])

        noisy_waveform,_ = librosa.load(noisy_filepath,sr=sample_rate)
        noisy_waveform_list.append(noisy_waveform)
        clean_waveform,_ = librosa.load(clean_filepath,sr=sample_rate)
        clean_waveform_list.append(clean_waveform)

        #pesq_nb = pesq.pesq(16000,clean_waveform,noisy_waveform,'nb')
        #pesq_wb = pesq.pesq(16000,clean_waveform,noisy_waveform,'wb')
        #result_dict['pesq nb'] = pesq_nb
        #result_dict['pesq wb'] = pesq_wb
        result_batch.append(result_dict)

    noisy_result = jasper_model.infer(waveforms=noisy_waveform_list)
    clean_result = jasper_model.infer(waveforms=clean_waveform_list)

    for i in range(len(item_batch)):
        result_batch[i]['noisy transcript'] = noisy_result['greedy transcript'][i]
        result_batch[i]['clean transcript'] = clean_result['greedy transcript'][i]
        result_batch[i]['noisy wer'] = word_error_rate([result_batch[i]['noisy transcript']],[result_batch[i]['ground_truth']])
        result_batch[i]['clean wer'] = word_error_rate([result_batch[i]['clean transcript']],[result_batch[i]['ground_truth']])

    return result_batch

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
    parser.add_argument('-b',dest='BATCH_SIZE',help='batch size',
                        default='8',type=int)
    parser.add_argument('--use_cpu',dest='USE_CPU',action='store_true',
                        help='use the cpu')
    args = parser.parse_args()

    #load up the dataset
    df = pd.read_csv(args.INDEX_PATH)
    df = df[df['source_length']==df['noisy_length']]

    #load up the model configuration
    yaml = YAML(typ="safe")
    with open(args.CONFIG) as f:
        model_definition = yaml.load(f)
    vocab = model_definition['labels']

    if args.USE_CPU:
        use_cpu=True
    else:
        use_cpu=False

    #build the jasper inference model
    jasper = JasperInference(model_definition,use_cpu=use_cpu)
    jasper.restore_weights(encoder_weight_path=args.ENCODER_PATH,decoder_weight_path=args.DECODER_PATH)

    #convert the dataframe to a list of dicts
    records = df.to_dict('records')

    #this will hold the processed items
    result_list = []

    for item_batch in tqdm.tqdm(batch(records,n=args.BATCH_SIZE)):
        result_batch = process_batch(item_batch,args.DATASET_ROOT,jasper)
        result_list+=result_batch
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(args.OUTPUT)
