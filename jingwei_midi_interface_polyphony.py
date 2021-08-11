
import pretty_midi as pyd
import numpy as np
import os
from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys
import pickle
from joblib import delayed
from joblib import Parallel
import time
import platform

class midi_interface_polyphony(object):
    def __init__(self, recogLevel='Mm'):
        self.cl = Chord_Loader(recogLevel = recogLevel)

    def load_single(self, file_path):
        midi_data = pyd.PrettyMIDI(file_path) 
        tempo = midi_data.get_tempo_changes()[-1][0]
        
    def Midi2PrMatrix_byBeats(self, polyphony_track, downbeats):
        time_stamp_sixteenth_reso = []
        delta_set = []
        for i in range(len(downbeats)-1):
            s_curr = downbeats[i]
            s_next = downbeats[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                time_stamp_sixteenth_reso.append(s_curr + delta * i)
                delta_set.append(delta)
        time_stamp_sixteenth_reso = np.array(time_stamp_sixteenth_reso)

        pr_matrix = np.zeros((time_stamp_sixteenth_reso.shape[0], 128))
        for note in polyphony_track.notes:
            onset = note.start
            t = np.argmin(np.abs(time_stamp_sixteenth_reso - onset))
            p = note.pitch
            duration = int(round((note.end - onset) / delta_set[t]))
            pr_matrix[t, p] = duration
        return pr_matrix

    def numpySplit(self, matrix, WINDOWSIZE=32, HOPSIZE=16):
        splittedMatrix = np.empty((0, WINDOWSIZE, 128))
        #print(matrix.shape[0])
        for idx_T in range(0, matrix.shape[0]-WINDOWSIZE, HOPSIZE):
            sample = matrix[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
        return splittedMatrix

    def PrMatrix2Midi(self, PrMatrix, ts=None, ks=None, tempo=120, fourthNote_reso=4):
        midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        instrument = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        delta = 60 / tempo / fourthNote_reso
        idx_onset = list(1 * (np.sum(PrMatrix, axis=0) != 0))
        #print(idx_onset)
        for t in range(len(idx_onset)):
            if idx_onset[t] == 1:
                pitch_onset = list(1 * (PrMatrix[:, t] != 0))
                for p in range(len(pitch_onset)):
                    if pitch_onset[p] == 1:
                        duration = PrMatrix[p, t]
                        start = delta * t
                        end = delta * (t + duration)
                        new_note = pyd.Note(velocity=100, pitch=p, start=start, end=end)
                        instrument.notes.append(new_note)
        midiReGen.instruments.append(instrument)
        if not ts == None:
            midiReGen.time_signature_changes = ts
        if not ks == None:
            midiReGen.key_signature_changes = ks
        return midiReGen

    
if __name__ == "__main__":
    #debug script
    """
    midi_data = pyd.PrettyMIDI('D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi_processed/0025.mid')
    ts = midi_data.time_signature_changes
    ks = midi_data.key_signature_changes
    tempo = midi_data.get_tempo_changes()[-1][0]
    print(ts, ks, tempo)
    processor = midi_interface_polyphony()
    pr_matrix = processor.Midi2PrMatrix_byBeats(midi_data)
    midi = processor.PrMatrix2Midi(pr_matrix, ts=ts, ks=ks, tempo=tempo)
    midi.write('poly_test.mid')"""
    #print(pr_matrix[:, 16])
    root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi_processed'
    triple = 0
    mal = 0
    for item in os.listdir(root):
        file = os.path.join(root, item)
        try:
            midi_data = pyd.PrettyMIDI(file)
        except:
            mal += 1
            continue
        ts = midi_data.time_signature_changes
        for ts_item in ts:
            if ts_item.numerator % 2 != 0:
                triple += 1
                print(item)
                break
    print('triple:', triple, 'mal:', mal)