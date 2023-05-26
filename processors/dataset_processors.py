from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random, os, re, json, pickle
from flax import io
import pandas as pd
from dataset_config import VGGSoundConfig, ACAV2MConfig, AudioCapsConfig, AudioSetConfig

def json_load(json_path):
    data = json.loads(tf.io.read_file(json_path).numpy())
    return data

@dataclass
class DatasetProcessor(ABC):

    @abstractmethod
    def get_filepaths_and_descriptions(self, blacklist) -> Tuple[List[str], List[List[str]]]:
        pass

class VGGSoundProcessor(DatasetProcessor):
    # paired wav-json file
    config = VGGSoundConfig()
    
    def get_filepaths_and_descriptions(self, blacklist=None, current_split='full'):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        
        # load audio filepaths
        existing_audiopaths = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')

        # load meta json file
        vgg_meta_file = os.path.join(self.config.data_dir, 'vggsound_full.json')
        with tf.io.gfile.GFile(vgg_meta_file, 'r') as f:
            vgg_meta_dict = json.load(f)
        
        for audiofile in tqdm(existing_audiopaths[:]):

            # get list of text captions
            audio_name = audiofile.split('/')[-1].split('.wav')[0]                
            audio_filepath_list.append(audiofile)

            # obtain description item # tags and title+text
            text_captions = {}
            text_captions['description'] = vgg_meta_dict[audio_name]
            text_dict[audio_name] = text_captions
        
        return audio_filepath_list, text_dict

class AudioCapsProcessor(DatasetProcessor):
    #  AudioCaps uses a master cvs for each datasplit
    config = AudioCapsConfig()
    
    def get_filepaths_and_descriptions(self, blacklist=None, current_split=''):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        meta_info_dict = json_load(os.path.join(self.config.data_dir, 'meta_info.json'))
        vggsound_test_dict = json_load(os.path.join(self.config.data_dir, 'vggsound_test.json'))

        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
            
            # load audio filepaths
            audio_name = audio_filepath.split('/')[-1].split('.wav')[0]
            if audio_name in vggsound_test_dict:
                continue
            audio_filepath_list.append(audio_filepath)
            
            # get list of text captions
            audio_filename = audio_filepath.split('/')[-1]

            # collecting captions
            text_captions = {}
            text_captions['description'] = meta_info_dict[audio_name]
            text_dict[audio_name] = text_captions
            
            # obtain computer description item
        return audio_filepath_list, text_dict

class AudioSetProcessor(DatasetProcessor):
    config = AudioSetConfig()
    
    def get_filepaths_and_descriptions(self, blacklist, current_split=''):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        
        # load meta dict
        class_label_df = pd.read_csv(self.config.data_dir + '/class_labels_indices.csv', 
                                     usecols=["mid", "display_name"])
        class_label_dict = class_label_df.set_index('mid').T.to_dict('list')
        
        # get list of text captions
        caption_filename = current_split + '.csv'
        caption_path = os.path.join(self.config.data_dir, caption_filename)
        caption_list_df = pd.read_csv(caption_path, skiprows=3, sep=', ', engine='python',
                                      header=None,
                                      names=['YTID', 'start_s', 'end_s', 'positive_labels'])
        caption_dict = caption_list_df.set_index('YTID')['positive_labels'].to_dict()
        
        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
                
            audio_name = audio_filepath.split('/')[-1].split('.wav')[0]
            ytb_id = audio_name[1:]
            if ytb_id in blacklist:
                continue
            
            # load audio filepaths
            audio_filepath_list.append(audio_filepath)
            data_slice_labels = caption_dict[ytb_id].replace('"', '').split(',')

            # collecting captions
            text_captions = {}
            text_captions['tags'] = []
            for label in data_slice_labels:
                text_captions['tags'] += class_label_dict[label]
            text_dict[audio_name] = text_captions

        return audio_filepath_list, text_dict
    
class ACAV2MProcessor(DatasetProcessor):
    config = ACAV2MConfig()
    
    def get_filepaths_and_descriptions(self, blacklist, current_split=''):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        
        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
            
            # get wav split
            audio_filename = audio_filepath.split('/')[-1]
            
            ytb_id = audio_filename.split('+')[0]
            if ytb_id in blacklist:
                continue
            
            # load audio filepaths
            audio_filepath_list.append(audio_filepath)

        return audio_filepath_list, text_dict
    
if __name__ == "__main__":
    import time
    
    black_list_json = "/storageHDD/ge/audio_sfx_wav/blacklist.json"
    with open(black_list_json, 'r') as f:
        blacklist_dict = json.load(f)
        
    dataprocessor = AudioSetProcessor()
    filepaths, descriptions = dataprocessor.get_filepaths_and_descriptions(set(blacklist_dict['audioset']), 'unbalanced_train_segments')
    print(len(filepaths))
    
    dataprocessor = ACAV2MProcessor()
    filepaths, descriptions = dataprocessor.get_filepaths_and_descriptions(set(blacklist_dict['audioset']), 'audio')
    print(len(filepaths))
    
        