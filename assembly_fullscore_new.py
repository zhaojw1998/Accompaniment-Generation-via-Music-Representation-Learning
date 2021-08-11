import sys
sys.path.append('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement')
from model import DisentangleVAE
from ptvae import PtvaeDecoder
sys.path.append('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/code')
from EC2model import VAE

import pandas as pd
import pretty_midi as pyd 
from pretty_midi.utilities import program_to_instrument_class
import numpy as np
import os
from tqdm import tqdm
from jingwei_dataProcessor import songDataProcessor, melody_processor
import converter
import torch
torch.rand(1).cuda()
from torch.utils.data import Dataset, DataLoader

from assembly_full_score import QueryProcessor, piano_roll_dataset, midiRender, split_phrases


#def merge(seg_list, lenT):
#    new_list = []
#    for item in seg_list:
        
sys.path.append('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement')
from jingwei_midi_interface_mono_and_chord import midi_interface_mono_and_chord
from jingwei_midi_interface_polyphony import midi_interface_polyphony


def traverse(query_segmentation='A8A8B8B8\n', require_accChord=False):
    """load query"""
    song_name, segmentation, note_shift = 'Falling About.mid', 'A8A8B8B8\n', 0
    #song_name, segmentation, note_shift = 'Flapjack.mid', 'A8A8B8B8\n', 1
    #song_name, segmentation, note_shift = 'Hopwas Hornpipe.mid', 'A8A8B8C8\n', 1
    #song_name, segmentation, note_shift = 'Beaux of Oakhill.mid', 'A8A8B8B8\n', 1
    #song_name, segmentation, note_shift = 'Lord Moira.mid', 'A4A4B8\n', 0.5
    song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_Dual-track-Direct'
    #song_root='C:/Users/lenovo/Desktop/updated_data/new_midi_chord'
    #get query phrase
    songData = QueryProcessor(song_name=song_name, song_root=song_root, note_shift=note_shift)
    pianoRoll, _, TIV = songData.melody2pianoRoll() #pianoRoll: time_span * 142; TIV: time_span * 12 (time in 16th note)
    chord_table = songData.chord2chordTable()   #chord_table: time_span * 36 (time in 4th note)

    """load and rank reference"""
    segmentation_root = 'D:/Computer Music Research/03_Shuqi/hierarchical-structure-analysis/POP909'
    data_root = 'D:/Computer Music Research/03_Shuqi/POP909-full-phrase'
    df = pd.read_excel('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/index.xlsx')
    #for song_id in range(1, 910):
    for song_id in range(1, 910):
        #print(song_id)
        meta_data = df[df.song_id == song_id]
        num_beats = meta_data.num_beats_per_measure.values[0]
        num_quavers = meta_data.num_quavers_per_beat.values[0]
        if int(num_beats) == 3 or int(num_quavers) == 3:
            continue
        try:
            with open(os.path.join(segmentation_root, str(song_id).zfill(3)+'/segsections.txt'), 'r') as f:
                info = f.readlines()[0]
        except:
            continue
        reference = split_phrases(info)

        song_root = os.path.join(data_root, str(song_id).zfill(3))
        ec2vae_format = np.load(os.path.join(song_root, 'ec2vae_format_full.npy'))  #time_span * 142 (time in 16th note)
        pr_matrix = np.load(os.path.join(song_root, 'pr_matrix_full.npy'))  #128 * time_span (time in 16th note)
        pr_chord = np.load(os.path.join(song_root, 'pr_chord_full.npy'))    #time_span * 14 (time in 4th note)
        #mix = np.load(os.path.join(song_root, 'pr_matrix_mix.npy')) #128 * time_span (time in 16th note)

        melodySet = []
        accompanySet = []
        chordSet = []
        for item in reference:
            label = item[0]
            length = item[1]
            start = item[2]
            if length == 8:
                save_root = 'C:/Users/lenovo/Desktop/for will/' + str(song_id).zfill(3)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                melody_chord = ec2vae_format[start*16: (start+length)*16, :]
                melodySet.append(melody_chord)

                accompaniment_track = pr_matrix[:, start*16: (start+length)*16]
                accompanySet.append(accompaniment_track)

                chord = pr_chord[start*4: (start+length)*4, :]
                np.save(os.path.join(save_root, 'chord.npy'), chord.shape)
                print(chord.shape)

            #melodySet.append(ec2vae_format[start*16: (start+length)*16, :])
            #accompanySet.append(pr_matrix[:, start*16: (start+length)*16])
            #chordSet.append(pr_chord[start*4: (start+length)*4, :])
        

        
if __name__ == '__main__':
    traverse()
        