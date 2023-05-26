###
# helper functions for 
# #1 find overlaps between freesound and {clotho test/val, fsd50k eval, us8k, esc50k} 
# #2 find overlaps between audioset unbalanced train and {audiocaps test/val, vgg test}
###
from glob import glob
from tqdm import tqdm
import csv, os, json, shutil
from collections import defaultdict

black_list_json = "/storageHDD/ge/audio_sfx_wav/blacklist.json"
raw_audiopath = '/storageHDD/ge/audio_sfx_raw'

# audioset family
vgg_filelist = os.path.join(raw_audiopath, 'vggsound', 'vggsound_test.json')
audiocaps_folder = '/home/ge/dataset/audiocaps/dataset'
audiocaps_split = ['test.csv', 'val.csv']

# freesound family
clotho_folder = os.path.join(raw_audiopath, 'clothov2') 
clotho_split = ['clotho_metadata_evaluation.csv', 'clotho_metadata_validation.csv']
us8k_filelist =  os.path.join(raw_audiopath, 'us8k/UrbanSound8K/metadata/UrbanSound8K.csv')
esc50_filelist = os.path.join(raw_audiopath, 'esc50.csv')
fsd50keval_filelist = os.path.join(raw_audiopath, 'eval_clips_info_FSD50K.json')

def load_item_from_meta(meta_file, item_order, heading_row=0):
    
    overlap_list = []
    assert meta_file.endswith('.csv')
    with open(meta_file, newline='', errors='ignore') as csvfile:
        dataset_rows = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(dataset_rows):
            if i <= heading_row:
                continue

            if row[item_order] == 'Not found':
                continue

            overlap_list.append(row[item_order])
                
    return overlap_list

if __name__ == '__main__':
    
    overlap_ids = defaultdict(list)
    
    # 1. Audioset overlaps
    ## a. audiocaps
    for split in audiocaps_split:
        audiocaps_split_csv = os.path.join(audiocaps_folder, split)
        audiocaps_split_list = load_item_from_meta(audiocaps_split_csv, item_order=1)
        overlap_ids['audioset'] += audiocaps_split_list
        
    ## b. vggsound test
    with open(vgg_filelist, 'r') as f:
        vgg_dict = json.load(f)
    
    vgglist = [vggfile.split('+')[0] for vggfile in list(vgg_dict.keys())]
    overlap_ids['audioset'] += vgglist
    
    # 2. Freesound overlaps
    ## a. clotho eval val file list with freesound
    for split in clotho_split:
        clotho_split_csv = os.path.join(clotho_folder, split)
        clotho_split_list = load_item_from_meta(clotho_split_csv, item_order=2)
        overlap_ids['freesound'] += clotho_split_list

    ## b. us8k 
    us8k_list = load_item_from_meta(us8k_filelist, item_order=1)
    overlap_ids['freesound'] += us8k_list

    ## c. esc50
    esc50_list = load_item_from_meta(esc50_filelist, item_order=-2)
    overlap_ids['freesound'] += esc50_list

    ## d. fsd50k eval
    with open(fsd50keval_filelist, 'r') as f:
        fsd50keval_dict = json.load(f)
        
    overlap_ids['freesound'] += list(fsd50keval_dict.keys())
    
    # remove duplicates
    overlap_ids['freesound'] = list(set(overlap_ids['freesound']))
    overlap_ids['audioset'] = list(set(overlap_ids['audioset']))
    
    print(len(overlap_ids['freesound']))
    print(len(overlap_ids['audioset']))
    
    with open(black_list_json, 'w') as f:
        json.dump(overlap_ids, f)
    f.close()
    
    