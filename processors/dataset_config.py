import sys
sys.path.append('../')
from dataclasses import dataclass, field
from typing import Optional
from params import *

@dataclass
class VGGSoundConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'vggsound'
    dataset_split: str = AUDIOTEXT_DATA_PATH + 'vggsound.csv'
    data_split: list = field(default_factory=lambda: ['train', 'test'])

@dataclass
class AudioCapsConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'audiocaps'
    data_split: list = field(default_factory=lambda: ['train', 'test', 'val'])
        
@dataclass
class AudioSetConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'audioset'
    data_split: list = field(default_factory=lambda: ['balanced_train_segments', 'eval_segments', 'unbalanced_train_segments'])
        
@dataclass
class ACAV2MConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'acav2m'