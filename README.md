# openSFX-TFShard
A codebase for open source sound effects (SFX) TFrecord sharding data preparation. It is initially used for CLAP model training on TPUs. 

# Datasets

## Training dataset
| Name    |Duration |Number of Samples   | Caption | Shards |
|---------|--------|--------------------|--------- |--------- |
| Audioset  |X hrs  | XX  | synthetic text|512|
| ACAV2M  |X hrs  | XX  |synthetic text |512|
| Freesound (AIR) |X hrs  | 513061 |(raw text, synthetic text)|512|
| Epidemic Sound  |X hrs   |75645 |text| 128|
| BBC Sound Effects<br />(Full, SoundDescriptions)|X hrs|33042|text|256|
| Audiostock  |X hrs  | 9837|text  |32|
| Free To Use Sounds |X hrs   | 8348   |label  |64|
| ClothoV2-train |X hrs  | 5929  |5 text  |32|
| MACS  |X hrs  | 3930  |5 text |32|
| AudioCaps-train  |X hrs  | XX  |1 text |32|

## Evaluation dataset
| Name    |Duration |Number of Samples   | Caption | Type|
|---------|--------|--------------------|--------- |--------- |
| VGGSound |X hrs  | 167338   |label | ZS|
| ESC50 |X hrs  | X  |label | ZS|
| US8k |X hrs  | X   |label | ZS|
| ClothoV2-test  |X hrs  | 5929  |5 text |  Retrievel|
| AudioCaps-test  |X hrs  | XX  |X text | Retrievel|
|WavText5K| X hrs| XX  |1 text | Retrievel|
|HEAR|  | | label | Probing|


# Preprocessing Code

## Pipeline:
### 1. Audio preprocess (local)
Convert diverse audio formats into mono 16kHz audio in wav format

### 2. Overlap record (local)
* AudioCaps is a subset of AudioSet, VGGSound has overlaps with Audioset.
* Freesound has overlaps with clothov2, US8k, ESC50 and FSD50k.
(to add clotho overlaps)

### 3. TFrecordize