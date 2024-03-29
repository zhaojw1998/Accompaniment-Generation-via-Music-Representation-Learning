import sys

from pretty_midi.utilities import program_to_instrument_class

sys.path.append('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement')
from model import DisentangleVAE
from ptvae import PtvaeDecoder

sys.path.append('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/code')
from EC2model import VAE

import pandas as pd
import pretty_midi as pyd 
import numpy as np
import os
from tqdm import tqdm
from jingwei_dataProcessor import songDataProcessor, melody_processor
import converter
import torch
torch.rand(1).cuda()
from torch.utils.data import Dataset, DataLoader

class QueryProcessor(object):
    def __init__(self, song_name, 
                        song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat_withChord', 
                        note_shift = 0):
        self.song = pyd.PrettyMIDI(os.path.join(song_root, song_name))
        self.downbeats = list(self.song.get_downbeats())
        self.downbeats.append(self.downbeats[-1] + (self.downbeats[-1] - self.downbeats[-2]))
        if not note_shift == 0:
            tempo = int(round(self.song.get_tempo_changes()[-1][0]))
            #print(tempo)
            shift_const = (60 / tempo) * note_shift
            self.downbeats = [item+shift_const for item in self.downbeats]
        #print(self.downbeats[-100:])
        self.melody_track = self.song.instruments[0]
        self.chord_track = self.song.instruments[-1]

    def getDownbeats(self):
        return self.downbeats

    def getToneKey(self):
        return self.song.key_signature_changes[0].key_number

    def computeTIV(self, chroma):
        #inpute size: Time*12
        #chroma = chroma.reshape((chroma.shape[0], -1, 12))
        #print('chroma', chroma.shape)
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        #print(TIV.shape)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
        return TIV #Time * 12

    def melody2pianoRoll(self):
        processor = melody_processor(recogLevel='Seven')
        melodySeq = processor.getMelodySeq_byBeats(self.melody_track, self.downbeats)
        #print('1-')
        chordSeq = processor.getChordSeq_byBeats(self.chord_track, self.downbeats)
        #print(chordSeq)
        #print('2-')
        pianoRoll, chordMatrix = processor.seq2Numpy(melodySeq, chordSeq)
        #print('3-')
        #EC2VAE_format = processor.numpySplit(pianoRoll, WINDOWSIZE=32, HOPSIZE=32)
        TIV = self.computeTIV(chordMatrix)
        #print('4-')
        return pianoRoll, chordSeq, TIV

    def chord2chordTable(self):
        return self.chordConverter(self.chord_track, self.downbeats)

    def chordConverter(self, chord_track, downbeats):
        """only applicable to triple chords"""
        chromas = {
            #           1     2     3     4  5     6     7
            'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        }
        distr2quality = {'(4, 3)': 'maj/0', 
                         '(3, 5)': 'maj/1', 
                         '(5, 4)': 'maj/2', 

                         '(3, 4)': 'min/0', 
                         '(4, 5)': 'min/1', 
                         '(5, 3)': 'min/2',

                         '(4, 4)': 'aug/0',
                         
                         '(3, 3)': 'dim/0',
                         '(3, 6)': 'dim/1',
                         '(6, 3)': 'dim/2',

                         '(4, 3, 3)': '7/0',
                         '(3, 3, 2)': '7/1',
                         '(3, 2, 4)': '7/2',
                         '(2, 4, 3)': '7/3',

                         '(4, 3, 4)': 'maj7/0',
                         '(3, 4, 1)': 'maj7/1',
                         '(4, 1, 4)': 'maj7/2',
                         '(1, 4, 2)': 'maj7/3',

                         '(3, 4, 3)': 'min7/0',
                         '(4, 3, 2)': 'min7/1',
                         '(3, 2, 3)': 'min7/2',
                         '(2, 3, 4)': 'min7/3',

                         '(3, 4, 4)': 'minmaj7/0',
                         '(4, 4, 1)': 'minmaj7/1',
                         '(4, 1, 3)': 'minmaj7/2',
                         '(1, 3, 4)': 'minmaj7/3',

                         '(3, 3, 3)': 'dim7/0',

                         '(3, 3, 4)': 'hdim7/0',
                         '(3, 4, 2)': 'hdim7/1',
                         '(4, 2, 3)': 'hdim7/2',
                         '(2, 3, 3)': 'hdim7/3',
                         }
        NC = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
        last_time = 0
        chord_set = []
        chord_time = [0.0, 0.0]
        chordsRecord = []
        for note in chord_track.notes:
            if len(chord_set) == 0:
                chord_set.append(note.pitch)
                chord_time[0] = note.start
                chord_time[1] = note.end
            else:
                if note.start == chord_time[0] and note.end == chord_time[1]:
                    chord_set.append(note.pitch)
                else:
                    if last_time < chord_time[0]:
                        chordsRecord.append({"start":last_time,"end": chord_time[0], "chord" : NC})
                    chord_set.sort()
                    assert(len(chord_set) == 3 or len(chord_set) == 4)
                    if len(chord_set) == 3:
                        quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1])))]
                    elif len(chord_set) == 4:
                        quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1]), (chord_set[3]-chord_set[2])))]
                    root = chord_set[-int(quality.split('/')[-1])] % 12
                    chroma = chromas[quality.split('/')[0]]
                    chroma = chroma[-root:] + chroma[:-root]
                    bass = (chord_set[0]%12-root) % 12
                    
                    #concatenate
                    chordsRecord.append({"start": chord_time[0],"end": chord_time[1],"chord": [root]+chroma+[bass]})
                    last_time = chord_time[1]
                    chord_set = []
                    chord_set.append(note.pitch)
                    chord_time[0] = note.start
                    chord_time[1] = note.end 
        if len(chord_set) > 0:
            if last_time < chord_time[0]:
                chordsRecord.append({"start":last_time ,"end": chord_time[0], "chord" : NC})
            chord_set.sort()
            assert(len(chord_set) == 3 or len(chord_set) == 4)
            if len(chord_set) == 3:
                quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1])))]
            elif len(chord_set) == 4:
                quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1]), (chord_set[3]-chord_set[2])))]
            root = chord_set[-int(quality.split('/')[-1])] % 12
            chroma = chromas[quality.split('/')[0]]
            chroma = chroma[-root:] + chroma[:-root]
            bass = (chord_set[0] % 12 - root) % 12
            chordsRecord.append({"start": chord_time[0],"end": chord_time[1],"chord": [root]+chroma+[bass]})
            last_time = chord_time[1]
        ChordTable = []
        anchor = 0
        chord = chordsRecord[anchor]
        start = chord['start']
        for i in range(len(downbeats)-1):
            s_curr = downbeats[i]
            s_next = downbeats[i+1]
            delta = (s_next - s_curr) / 4
            for i in range(4):  # one-beat resolution
                while chord['end'] <= (s_curr + i * delta) and anchor < len(chordsRecord)-1:
                    anchor += 1
                    chord = chordsRecord[anchor]
                    start = chord['start']
                if s_curr + i * delta < start:
                    ChordTable.append(converter.expand_chord(chord=NC, shift=0))
                else:
                    ChordTable.append(converter.expand_chord(chord=chord['chord'], shift=0))
        return np.array(ChordTable)

    @classmethod
    def numpySplit(self, matrix, WINDOWSIZE=32, HOPSIZE=16, VECTORSIZE=142):
        start_downbeat = 0
        end_downbeat = matrix.shape[0]//16
        assert(end_downbeat - start_downbeat >= 2)
        splittedMatrix = np.empty((0, WINDOWSIZE, VECTORSIZE))
        #print(matrix.shape[0])
        #print(matrix.shape[0])
        for idx_T in range(start_downbeat*16, (end_downbeat-1)*16, HOPSIZE):
            if idx_T > matrix.shape[0]-32:
                break
            sample = matrix[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
        return splittedMatrix
    @classmethod
    def chordSplit(self, chord, WINDOWSIZE=8, HOPSIZE=8):
        start_downbeat = 0
        end_downbeat = chord.shape[0]//4
        splittedChord = np.empty((0, 8, 36))
        #print(matrix.shape[0])
        for idx_T in range(start_downbeat*4, (end_downbeat-1)*4, HOPSIZE):
            if idx_T > chord.shape[0]-8:
                break
            sample = chord[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedChord = np.concatenate((splittedChord, sample), axis=0)
        return splittedChord

def numpy2melodyAndChord(matrix, tempo=120, start_time=0.0):
    processor = melody_processor(recogLevel='Seven')
    if len(matrix.shape) == 2:
        return processor.midiReconFromNumpy(matrix=matrix, tempo=tempo, start_time=start_time)
    else:
        assert(len(matrix.shape) == 3)
        start = start_time
        melody_notes = []
        chord_notes = []
        for i in range(0, matrix.shape[0], 2):
            note_m, note_c = processor.midiReconFromNumpy(matrix=matrix[i], tempo=tempo, start_time=start)
            start += 8 / (tempo / 60)
            melody_notes += note_m
            chord_notes += note_c
        return melody_notes, chord_notes
    
def numpy2textureTrack(pr_matrix, tempo=120, start_time=0.0):
    alpha = 0.25 * 60 / tempo
    notes = []
    for t in range(pr_matrix.shape[0]):
        for p in range(128):
            if pr_matrix[t, p] >= 1:
                s = alpha * t + start_time
                e = alpha * (t + pr_matrix[t, p]) + start_time
                notes.append(pyd.Note(100, int(p), s, e))
    return notes
#get a lead sheet query

def tone_shift(original_batchData, i):
    #shift_tone i ranges from -6, 5
    #print(idx)
    tmpP = original_batchData[:, :, :128]
    #print(tmp.shape)
    tmpP = np.concatenate((tmpP[:, :, i:], tmpP[:, :, :i]), axis=-1)
    tmpC = original_batchData[:, :, 130:]
    tmpC = np.concatenate((tmpC[:, :, i:], tmpC[:, :, :i]), axis=-1)
    shifted_batchData = np.concatenate((tmpP, original_batchData[:, :, 128: 130], tmpC), axis=-1)
    return shifted_batchData

class piano_roll_dataset(Dataset):
    def __init__(self, diric='./scrape_musescore/data_to_be_used/piano_roll_all.npy'):
        super(piano_roll_dataset, self).__init__()
        self.batched_piano = np.load(diric)
        print(self.batched_piano.shape)

    def __getitem__(self, idx):
        return self.batched_piano[idx]
        
    def __getlen__(self):
        return self.batched_piano.shape[0]

class midiRender(melody_processor):
    def __init__(self):
        super(midiRender, self).__init__()

    def numpy2textureTrack(self, pr_matrix, tempo=120, start_time=0.0):
        alpha = 0.25 * 60 / tempo
        notes = []
        for t in range(pr_matrix.shape[0]):
            for p in range(128):
                if pr_matrix[t, p] >= 1:
                    s = alpha * t + start_time
                    e = alpha * (t + pr_matrix[t, p]) + start_time
                    notes.append(pyd.Note(100, int(p), s, e))
        return notes
    
    def numpy2melodyAndChord(self, matrix, tempo=120, start_time=0.0):
        processor = melody_processor(recogLevel='Seven')
        return processor.midiReconFromNumpy(matrix=matrix, tempo=tempo, start_time=start_time)
    
    def leadSheet_recon(self, matrix, tempo=120, start_time=0.0):
        melody_notes, chord_notes = self.numpy2melodyAndChord(matrix, tempo, start_time)
        midi_ReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        melody_track = pyd.Instrument(program=pyd.instrument_name_to_program('Violin'))
        chord_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        melody_track.notes += melody_notes
        chord_track.notes += chord_notes
        midi_ReGen.instruments.append(melody_track)
        midi_ReGen.instruments.append(chord_track)
        return midi_ReGen

    def pr_matrix2note(self, pr_matrix, tempo=120):
        pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
        start = 0
        tempo = tempo
        midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        acc_track_recon = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        for idx in range(0, pr_matrix.shape[0], 2):
            notes = pt_decoder.pr_matrix_to_note(pr_matrix[idx], bpm=tempo, start=start)
            acc_track_recon.notes += notes
            start += 60 / tempo * 8
        midiReGen.instruments.append(acc_track_recon)
        return midiReGen

    
    def accomapnimentGeneration(self, piano_roll, pr_matrix, tempo=120):
        pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
        start = 0
        tempo = tempo
        midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        melody_track = pyd.Instrument(program=pyd.instrument_name_to_program('Violin'))
        texture_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        for idx in range(0, pr_matrix.shape[0], 2):
            melody_notes, chord_notes = self.numpy2melodyAndChord(matrix=piano_roll[idx], tempo=tempo, start_time=start)
            pr, _ = pt_decoder.grid_to_pr_and_notes(grid=pr_matrix[idx], bpm=tempo, start=0)
            #print(pr.shape)
            texture_notes = self.numpy2textureTrack(pr_matrix=pr, tempo=tempo, start_time=start)
            melody_track.notes += melody_notes
            texture_track.notes += texture_notes
            start += 60 / tempo * 8
        midiReGen.instruments.append(melody_track)
        midiReGen.instruments.append(texture_track)
        #midiReGen.write('test.mid')
        return midiReGen
    
def get_queryPhrase(song_name='2nd part for B music.mid',
                    start_downbeat=0, 
                    end_downbeat=8, 
                    note_shift= 0,
                    song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat_withChord'):
    #startdownbeat -content- middownbeat -content- middownbeat -content- end_downbeat
    #input should be a double-track midi
    songData = QueryProcessor(song_name=song_name, song_root=song_root, note_shift=note_shift)
    #get content
    pianoRoll, _, TIV = songData.melody2pianoRoll()
    #toneKey = songData.getToneKey()
    #chordfunc = chord_function_progression(chordSeq, toneKey)
    #get phrtase
    #pianoRoll_phrase = songData.numpySplit(pianoRoll, start_downbeat, end_downbeat, 32, 16, 142)
    #TIV_phrase = songData.numpySplit(TIV, start_downbeat, end_downbeat, 32, 16, 12)
    chord_table = songData.chord2chordTable()
    #pr_chord_phrase = songData.chordSplit(chord_table, start_downbeat, end_downbeat, 8, 4)
    return pianoRoll, chord_table

def split_phrases(segmentation, keepAcc=False):
    phrases = []
    lengths = []
    label_anchor = 0
    upper = 1
    length = 0
    current = 0
    while segmentation[current] != '\n':
        if not segmentation[current].isalpha():
            length += 1
            if segmentation[current+1].isalpha() or segmentation[current+1] == '\n':
                label = segmentation[label_anchor]
                phrase_length = int(segmentation[upper: upper+length])
                if keepAcc == False:
                    if segmentation[label_anchor].isupper():
                        phrases.append(label)
                        lengths.append(phrase_length)
                else:
                    phrases.append(label)
                    lengths.append(phrase_length)
                label_anchor = current+1
                upper = current+2
                length = 0
        current += 1
    return [(phrases[i], lengths[i], sum(lengths[:i])) for i in range(len(phrases))]

def arrangement(query, reference, prgsQ, prgsR):
    #query and reference are tuples (phrase_label, phrase_length, start_idx), e.g., ('A', 8, 0)
    if len(query) == 1:
        return True, [], 0
    if len(reference) == 1:
        return False, [], 0
    if query[0][1] == reference[0][1]:
        if prgsQ != prgsR:
            prgsR += ord(reference[1][0]) - ord(reference[0][0])
            new_reference = reference[1:]
            flag, arr, score = arrangement(query, new_reference, prgsQ, prgsR)
            if flag == True:
                return True, arr, score
            else:
                selection = reference[0]
                if query[0][0] == query[1][0] and (reference[0][0] != reference[1][0]):
                    #pointR += 0
                    prgsR = 0
                else:
                    prgsR = ord(reference[1][0]) - ord(reference[0][0])
                    reference = reference[1:]
                prgsQ = ord(query[1][0]) - ord(query[0][0])
                query = query[1:]
                flag, arr, score = arrangement(query, reference, prgsQ, prgsR)
                return flag, [selection] + arr, score - 1
        else:
            selection = reference[0]
            if query[0][0] == query[1][0] and (reference[0][0] != reference[0][0]):
                #pointR += 0
                prgsR = 0
            else:
                prgsR = ord(reference[1][0]) - ord(reference[0][0])
                reference = reference[1:]
            prgsQ = ord(query[1][0]) - ord(query[0][0])
            query = query[1:]
            flag, arr, score = arrangement(query, reference, prgsQ, prgsR)
            return flag, [selection] + arr, score
    elif query[0][1] > reference[0][1] and query[0][1]%reference[0][1] == 0:
        new_reference = [(reference[0][0], reference[0][1]+reference[1][1], reference[0][-1])] + reference[2:]
        flag, arr, score = arrangement(query, new_reference, prgsQ, prgsR)
        if flag == True:
            return True, arr, score - 1
        else:
            return arrangement(query, reference[1:], prgsQ, prgsR+ord(reference[1][0])-ord(reference[0][0]))
    elif query[0][1] < reference[0][1] and reference[0][1]%query[0][1] == 0:
        if len(query) > 2:
            new_query = [(query[1][0], query[0][1]+query[1][1], query[0][-1])] + query[2:]
            flag, arr, score = arrangement(new_query, reference, prgsQ, prgsR)
            if flag == True:
                return True, arr, score - 1
            else:
                return arrangement(query, reference[1:], prgsQ, prgsR+ord(reference[1][0])-ord(reference[0][0]))
        else:
            return arrangement(query, reference[1:], prgsQ, prgsR+ord(reference[1][0])-ord(reference[0][0]))
    else:
        return arrangement(query, reference[1:], prgsQ, prgsR+ord(reference[1][0])-ord(reference[0][0]))

def traverse(query_segmentation='A8A8B8B8\n', require_accChord=False):
    segmentation_root = 'D:/Computer Music Research/03_Shuqi/hierarchical-structure-analysis/POP909'
    data_root = 'D:/Computer Music Research/03_Shuqi/POP909-full-phrase'
    df = pd.read_excel('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/index.xlsx')
    melody_collection = []
    accompany_collection = []
    chord_collection = []
    index_collection = []
    config_collection = []
    score_collection = []
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
        query = split_phrases(query_segmentation)
        reference.append(('X', 0))
        query.append(('X', 0))
        flag, arr, score = arrangement(query, reference, 0, 0)
        if flag == True:
            #arr: e.g., (A, 8, 0)
            song_root = os.path.join(data_root, str(song_id).zfill(3))
            ec2vae_format = np.load(os.path.join(song_root, 'ec2vae_format_full.npy'))
            pr_matrix = np.load(os.path.join(song_root, 'pr_matrix_full.npy'))
            pr_chord = np.load(os.path.join(song_root, 'pr_chord_full.npy'))
            ec2vae_selection = np.empty((0, 142)) 
            pr_selection = np.empty((128, 0))
            pr_chord_selection = np.empty((0, 14))
            for item in arr:
                ec2vae_selection = np.concatenate((ec2vae_selection, ec2vae_format[item[-1]*16: (item[-1]+item[-2])*16, :]), axis=0)
                pr_selection = np.concatenate((pr_selection, pr_matrix[:, item[-1]*16: (item[-1]+item[-2])*16]), axis=-1)
                pr_chord_selection = np.concatenate((pr_chord_selection, pr_chord[item[-1]*4: (item[-1]+item[-2])*4, :]), axis=0)
            #print(song_id, arr, np.shape(ec2vae_selection), pr_selection.shape)
            melody_collection.append(ec2vae_selection)
            accompany_collection.append(pr_selection)
            chord_collection.append(pr_chord_selection)
            index_collection.append(song_id)
            config_collection.append([item[0]+str(item[1]) for item in arr])
            score_collection.append(score)
    if not require_accChord:
        return np.array(melody_collection), np.array(accompany_collection), index_collection, config_collection, score_collection
    else:
        return np.array(melody_collection), np.array(accompany_collection), np.array(chord_collection), index_collection, config_collection, score_collection
            
# Task 1: for testing if resfister matters
def register_shift(query_piano_roll_phrase, target_prPianoRoll_whole):
    query_piano_roll_phrase = query_piano_roll_phrase[::2].reshape((-1, 142))[:, :128]
    query = np.argmax(query_piano_roll_phrase, axis=-1)
    pitch = 0
    for idx, note in enumerate(query):
        if note != 0:
            pitch = note
        else:
            query[idx] = pitch
    target_prPianoRoll = target_prPianoRoll_whole[:, :128]
    target = np.argmax(target_prPianoRoll, axis=-1)
    pitch = 0
    for idx, note in enumerate(target):
        if note != 0:
            pitch = note
        else:
            target[idx] = pitch
    shift = int(round(np.mean(query) - np.mean(target)))
    #print(shift)
    #shifted_piano_roll = np.concatenate((np.roll(target_prPianoRoll_whole[:, :128], shift, axis=-1), target_prPianoRoll_whole[:, 128: 130]), axis=-1)
    #shifted_chord = np.roll(target_prPianoRoll_whole[:, 130:], shift, axis=-1)
    #shifted_target_prPianoRoll_whole = np.concatenate((shifted_piano_roll, shifted_chord), axis=-1)
    return shift

# Task 2: for testing if chord similarity matters
def chord_comparison(pr_chord_phrase, shifted_ensemble, segmentation, num_candidate=10):
    #pr_chord_phrase: batch*8*36; prChordSet: batch*32*14
    if len(pr_chord_phrase.shape) == 3:
        pr_chord_phrase = pr_chord_phrase[::2].reshape((-1, 36))[:, 12:24]
    else:
        pr_chord_phrase = pr_chord_phrase[:, 12:24] #duration * size
    #print(pr_chord_phrase)
    """use TIV"""
    pr_chord_phrase = computeTIV(pr_chord_phrase)
    shifted_ensemble = computeTIV(shifted_ensemble)
    #print(pr_chord_phrase.shape, shifted_ensemble.shape)
    """use chroma"""
    #pr_chord_phrase = pr_chord_phrase
    #shifted_ensemble = shifted_ensemble
    candidates, shifts, scores, recorder = cosine(pr_chord_phrase, shifted_ensemble, segmentation, num_candidate = num_candidate)
    #print(pr_chord_phrase.shape, shifted_ensemble.shape)
    #print(candidates, scores)
    return candidates, shifts, scores, recorder

def chord_shift(prChordSet):
    prChordSet = prChordSet[:, :, 1: -1]
    num_total = prChordSet.shape[0]
    shift_const = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    shifted_ensemble = []
    for i in shift_const:
        shifted_term = np.roll(prChordSet, i, axis=-1)
        shifted_ensemble.append(shifted_term)
    shifted_ensemble = np.array(shifted_ensemble)   #num_pitches * num_pieces * duration * size   #.reshape((-1, prChordSet.shape[1], prChordSet.shape[2]))
    return shifted_ensemble, num_total, shift_const
def computeTIV(chroma):
    #inpute size: Time*12
    #chroma = chroma.reshape((chroma.shape[0], -1, 12))
    #print('chroma', chroma.shape)
    if (len(chroma.shape)) == 4:
        num_pitch = chroma.shape[0]
        num_pieces = chroma.shape[1]
        chroma = chroma.reshape((-1, 12))
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        #print(TIV.shape)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
        TIV = TIV.reshape((num_pitch, num_pieces, -1, 12))
    else:
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        #print(TIV.shape)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
    return TIV #Time * 12
def cosine(query, instance_space, segmentation, threshold=0.8, num_candidate = 10):
    #query: T * 12
    #instance space: Batch * T * 12
    #instance_space: batch * vectorLength
    #print(instance_space[0, 6])
    #print(query)
    final_result = np.ones((instance_space.shape[1]))
    whole_shifts = []
    recorder = []
    start = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 4
            result = np.dot(instance_space[:, :, start: end, :], np.transpose(query[start: end, :]))/(np.linalg.norm(instance_space[:, :, start: end, :], axis=-1, keepdims=True) * np.transpose(np.linalg.norm(query[start: end, :], axis=-1, keepdims=True)) + 1e-10)
            #print(result.shape)
            #result = (result >= threshold) * 1
            result = np.trace(result, axis1=-2, axis2=-1)

            #
            shifts = np.argmax(result, axis=0)
            score = np.max(result, axis=0) / (end-start)
            whole_shifts.append(shifts)
            final_result = np.multiply(final_result, score)
            recorder.append(score)
            #print(result.shape)
            #final_result = np.multiply(final_result, result)    #element-wise product
            start = end
    #shifts = np.argmax(final_result, axis=0)
    #final_result = np.max(final_result, axis=0)

    whole_shifts = np.array(whole_shifts).transpose()
    #print(whole_shifts)

    candidates = np.array(range(0, final_result.shape[0]))#final_result.argsort()[::-1][:num_candidate]
    #shifts = shifts[candidates]
    scores = final_result[candidates]
    #print(shifts)
    #names = [os.listdir('./scrape_musescore/data_to_be_used/8')[i] for i in candidates]
    #sort by edit distance over melody
    #candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
    return candidates, whole_shifts, scores, recorder#, query[::4], instance_space[candidates][:, ::4]

def cosine_1d(query, instance_space, segmentation, num_candidate = 10):
    #query: T
    #instance space: Batch * T
    #instance_space: batch * vectorLength

    
    final_result = np.ones((instance_space.shape[0]))
    recorder = []
    start = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 16
            result = np.abs(np.dot(instance_space[:, start: end], query[start: end])/(np.linalg.norm(instance_space[:, start: end], axis=-1) * np.linalg.norm(query[start: end]) + 1e-10))
            recorder.append(result)
            final_result = np.multiply(final_result, result)    #element-wise product
            start = end
    #print(result.shape)
    #result = (result >= threshold) * 1
    #result = np.trace(result, axis1=-2, axis2=-1)
    #print(result.shape)
    candidates = final_result.argsort()[::-1][:num_candidate]
    scores = final_result[candidates]
    #names = [os.listdir('./scrape_musescore/data_to_be_used/8')[i] for i in candidates]
    #sort by edit distance over melody
    #candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
    return candidates, scores, recorder#, query[::4], instance_space[candidates][:, ::4]

def cosine_2d(query, instance_space, segmentation, record_chord=None, num_candidate = 10):
    final_result = np.ones((instance_space.shape[0]))
    recorder = []
    start = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 4
            result = np.dot(np.transpose(instance_space[:, start: end, :], (0, 2, 1)), query[start: end, :])/(np.linalg.norm(np.transpose(instance_space[:, start: end, :], (0, 2, 1)), axis=-1, keepdims=True) * np.linalg.norm(query[start: end, :], axis=0, keepdims=True) + 1e-10)
            #print(result.shape)
            #result = (result >= threshold) * 1
            #result = 0.6 * result[:, 0, 0] + 0.4 * result[:, 1, 1]
            result = np.trace(result, axis1=-2, axis2=-1) /2
            recorder.append(result)

            final_result = np.multiply(final_result, result)
            start = end
    if not record_chord == None:
        record_chord = np.array(record_chord)
        recorder = np.array(recorder)
        assert np.shape(record_chord) == np.shape(recorder)
        final_result = np.array([(np.product(recorder[:, i]) * np.product(record_chord[:, i])) * (2 *recorder.shape[0]) for i in range(recorder.shape[1])])
 
    candidates = final_result.argsort()[::-1]#[:num_candidate]
    scores = final_result[candidates]

    return candidates, scores, recorder




# Task 3:  for testing if melody contour matters
def appearanceMatch(query, candidate_idx, batchTarget):
    #query:  T X 142
    #batchTarget: numBatch X T X 142
    newScore = []
    #print(query.shape)
    for i in range(len(candidate_idx)):
        #print(search[i])
        candidate = batchTarget[candidate_idx[i]]
        #print(candidate.shape)
        assert(query.shape == candidate.shape)
        score = 0
        register = 0
        registerMelody = 0
        for idxT in range(candidate.shape[0]):
            MmChord = query[idxT][130:]
            #print(MmChord)
            #print(MmChord.shape, candidate[idxT].shape)
            note = np.argmax(candidate[idxT][:130])
            if note == 129:
                if not np.argmax(query[max(0, idxT)][:130]) == 129:
                    score -= 1
                continue
            elif note == 128:
                note = register
            else:
                note = note % 12
                register = note
            continueFlag = 0
            if MmChord[note] == 1:
                continue
            else:
                score_to_add_pre = 0
                score_to_add_post = 0
                for idxt in range(-3, 1, 1):
                    melodyNote = np.argmax(query[max(0, idxT+idxt)][:130])
                    if melodyNote == 129:
                        continue
                    elif melodyNote == 128:
                        melodyNote = registerMelody
                    else:
                        melodyNote = melodyNote % 12
                        registerMelody = melodyNote
                    #print(melodyNote, note)
                    if melodyNote == note:
                        score_to_add_pre = (3+idxt)
                        continueFlag = 1
                for idxt in range(1, 4, 1):
                    melodyNote = np.argmax(query[min(idxT+idxt, candidate.shape[0]-1)][:130])
                    if melodyNote == 129:
                        continue
                    elif melodyNote == 128:
                        melodyNote = registerMelody
                    else:
                        melodyNote = melodyNote % 12
                        registerMelody = melodyNote
                    #print(melodyNote, note)
                    if melodyNote == note:
                        score_to_add_post = (3-idxt)
                        continueFlag = 1
                        #print('add')
                        #break
                score += max(score_to_add_pre, score_to_add_post)
                if continueFlag:
                    continue
            score -= 1
            #print(score)
        newScore.append((candidate_idx[i], score))
    #print(sorted(newScore, reverse=True, key=lambda score: score[1]))
    return [item[0] for item in sorted(newScore, reverse=True, key=lambda score: score[1])], [item[1] for item in sorted(newScore, reverse=True, key=lambda score: score[1])]
def piano_roll_shift(prpiano_rollSet):
    num_total, timeRes, piano_shape = prpiano_rollSet.shape
    shift_const = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    shifted_ensemble = []
    for i in shift_const:
        piano = prpiano_rollSet[:, :, :128]
        rhythm = prpiano_rollSet[:, :, 128:130]
        chord = prpiano_rollSet[:, :, 130:]
        shifted_piano = np.roll(piano, i, axis=-1)
        shifted_chord = np.roll(chord, i, axis=-1)
        shifted_piano_roll_set = np.concatenate((shifted_piano, rhythm, shifted_chord), axis=-1)
        shifted_ensemble.append(shifted_piano_roll_set)
    shifted_ensemble = np.array(shifted_ensemble).reshape((-1, timeRes, piano_shape))
    return shifted_ensemble, num_total, shift_const

def rhythmCompare(query, instance_space, num_candidate=10):
    #print(query)
    for batch in range(instance_space.shape[0]):
        #print(instance_space.shape)
        assert(instance_space[batch].shape == query.shape)
        instance_space[batch] = instance_space[batch]-query
    result = np.sum(np.abs(instance_space), axis=-1)
    #print(result.shape)
    candidates = result.argsort()[:num_candidate]
    scores = result[candidates]
    #print(instance_space[1712])
    return candidates, scores#, query[::4], instance_space[candidates][:, ::4]

def search_random_for_leadSheet():
    #define query
    #song_name, segmentation, note_shift = "Morning Star.mid", 'A4A4B8B8\n', 1
    #song_name, segmentation, note_shift = "The 29th of May.mid", 'A4A4B8\n', 0
    #song_name, segmentation, note_shift = "Barry's Favourite.mid", 'A8A8B8C8\n', 1
    song_name, segmentation, note_shift = 'Boggy Brays.mid', 'A8A8B8B8\n', 0
    song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_Dual-track-Direct'
    #get query phrase
    songData = QueryProcessor(song_name=song_name, song_root=song_root, note_shift=note_shift)
    pianoRoll, _, TIV = songData.melody2pianoRoll()
    chord_table = songData.chord2chordTable()
    #print(chord_table[0])
    #print(chord_table[-1])

    prPianoRollSet, prMatrixSet, config_collection, score_collection = traverse(segmentation)
    #piano_roll_phrase, pr_chord_phrase = get_queryPhrase(song_name, start_bar, end_bar, note_shift)
    #phrase_length = end_bar - start_bar
    #load arrangement set
    #pick one randomly
    random_number = np.random.randint(prMatrixSet.shape[0])
    print('random number:', random_number)
    pr_matrix = prMatrixSet[random_number]
    pr_matrix = np.transpose(pr_matrix)
    prPianoRoll = prPianoRollSet[random_number]
    config = config_collection[random_number]
    score = score_collection[random_number]
    #print(prPianoRoll.shape)
    #chord_comparison(pr_chord_phrase, prChordSet)
    shift = register_shift(pianoRoll, prPianoRoll)
    print(shift, config, score)
    shift= np.sign(shift) * (abs(shift)%12)
    pr_matrix_shifted = np.roll(pr_matrix, shift, axis=-1)


    pr_matrix = QueryProcessor.numpySplit(pr_matrix, 32, 16, 128)
    pr_matrix_shifted = QueryProcessor.numpySplit(pr_matrix_shifted, 32, 16, 128)
    pianoRoll = QueryProcessor.numpySplit(pianoRoll, 32, 16, 142)
    gt_chord = QueryProcessor.chordSplit(chord_table, 8, 4)
    #print(gt_chord.shape)
    #init midi render
    midi_render = midiRender()
    #load corpus

    query_midi = midi_render.leadSheet_recon(pianoRoll[::2].reshape((-1, 142)), tempo=120, start_time=0)
    
    #retrieved acc
    acc_original = midi_render.pr_matrix2note(pr_matrix, 120)
    acc_original_shifted = midi_render.pr_matrix2note(pr_matrix_shifted, 120)
    #poly-disentangle
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/model_master_final.pt')
    model.load_state_dict(checkpoint)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    pr_matrix = torch.from_numpy(pr_matrix).float().cuda()
    pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
    gt_chord = torch.from_numpy(gt_chord).float().cuda()
    est_x = model.inference(pr_matrix, gt_chord, sample=False)
    est_x_shifted = model.inference(pr_matrix_shifted, gt_chord, sample=False)
    #Generate accompaniment
    midiReGen = midi_render.accomapnimentGeneration(pianoRoll, est_x, 120)
    midiReGen_shifted = midi_render.accomapnimentGeneration(pianoRoll, est_x_shifted, 120)
    midiReGen.write('accompaniment_test_1.mid')
    midiReGen_shifted.write('accompaniment_test_1_shifted.mid')
    acc_original.write('accompaniment_origional_1.mid')
    acc_original_shifted.write('accompaniment_origional_1_shifted.mid')
    query_midi.write('query_leadsheet.mid')
    target_midi = midi_render.leadSheet_recon(prPianoRoll, tempo=120, start_time=0)
    target_midi.write('target_leadsheet.mid')

def search_withChord_for_leadSheet():
    #define query
    #song_name, start_bar, end_bar, note_shift = 'Abram Circle.mid', 0, 8, 1
    #song_name, start_bar, end_bar, note_shift = '2nd part for B music.mid', 0, 8, 0
    song_name, start_bar, end_bar, note_shift = 'Boggy Brays.mid', 16, 24, 0
    #get query phrase
    piano_roll_phrase, pr_chord_phrase = get_queryPhrase(song_name, start_bar, end_bar, note_shift)
    phrase_length = end_bar - start_bar
    #load arrangement set
    phrase_root = os.path.join('../POP909-phrase', str(phrase_length))
    prMatrixSet = np.load(os.path.join(phrase_root, 'pr_matrix.npy'))
    prChordSet = np.load(os.path.join(phrase_root, 'pr_chord.npy'))
    prPianoRollSet = np.load(os.path.join(phrase_root, 'ec2vae_format.npy'))
    #pick one randomly
    
    shifted_prChordSet, scale, table = chord_shift(prChordSet)
    #pr_chord_phrase = shifted_prChordSet[20000]
    candidates, scores = chord_comparison(pr_chord_phrase, shifted_prChordSet)
    print(candidates, scores)
    shift = table[candidates[6] // scale]
    idx = candidates[6] % scale
    print(shift, idx)

    pr_matrix = prMatrixSet[idx]
    pr_matrix = np.transpose(pr_matrix)
    prPianoRoll = prPianoRollSet[idx]
    #print(prPianoRoll.shape)
    
    #shift = register_shift(piano_roll_phrase, prPianoRoll)
    #print(shift)
    pr_matrix_shifted = np.roll(pr_matrix, shift, axis=-1)
    pr_matrix = QueryProcessor.numpySplit(pr_matrix, 0, 8, 32, 16, 128)
    pr_matrix_shifted = QueryProcessor.numpySplit(pr_matrix_shifted, 0, 8, 32, 16, 128)

    piano = prPianoRoll[:, :128]
    rhythm = prPianoRoll[:, 128:130]
    chord = prPianoRoll[:, 130:]
    shifted_piano = np.roll(piano, shift, axis=-1)
    shifted_chord = np.roll(chord, shift, axis=-1)
    shifted_prPianoRoll = np.concatenate((shifted_piano, rhythm, shifted_chord), axis=-1)
    #init midi render
    midi_render = midiRender()
    #load corpus

    query_midi = midi_render.leadSheet_recon(piano_roll_phrase[::2].reshape((-1, 142)), tempo=120, start_time=0)
    
    #retrieved acc
    acc_original = midi_render.pr_matrix2note(pr_matrix, 120)
    acc_original_shifted = midi_render.pr_matrix2note(pr_matrix_shifted, 120)
    #poly-disentangle
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/model_master_final.pt')
    model.load_state_dict(checkpoint)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    pr_matrix = torch.from_numpy(pr_matrix).float().cuda()
    pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
    gt_chord_phrase = torch.from_numpy(pr_chord_phrase).float().cuda()
    est_x = model.inference(pr_matrix, gt_chord_phrase, sample=False)
    est_x_shifted = model.inference(pr_matrix_shifted, gt_chord_phrase, sample=False)
    #Generate accompaniment
    midiReGen = midi_render.accomapnimentGeneration(piano_roll_phrase, est_x, 120)
    midiReGen_shifted = midi_render.accomapnimentGeneration(piano_roll_phrase, est_x_shifted, 120)
    midiReGen.write('accompaniment_test_1.mid')
    midiReGen_shifted.write('accompaniment_test_1_shifted.mid')
    acc_original.write('accompaniment_origional_1.mid')
    acc_original_shifted.write('accompaniment_origional_1_shifted.mid')
    query_midi.write('query_leadsheet.mid')
    target_midi = midi_render.leadSheet_recon(shifted_prPianoRoll, tempo=120, start_time=0)
    target_midi.write('target_leadsheet.mid')

def search_withMeldoy_for_leadSheet():
    #define query
    #song_name, start_bar, end_bar, note_shift = 'Abram Circle.mid', 0, 8, 1
    #song_name, start_bar, end_bar, note_shift = '2nd part for B music.mid', 0, 8, 0
    song_name, start_bar, end_bar, note_shift = 'Boggy Brays.mid', 16, 24, 0
    #get query phrase
    piano_roll_phrase, pr_chord_phrase = get_queryPhrase(song_name, start_bar, end_bar, note_shift)
    phrase_length = end_bar - start_bar
    #load arrangement set
    phrase_root = os.path.join('../POP909-phrase', str(phrase_length))
    prMatrixSet = np.load(os.path.join(phrase_root, 'pr_matrix.npy'))
    prChordSet = np.load(os.path.join(phrase_root, 'pr_chord.npy'))
    prPianoRollSet = np.load(os.path.join(phrase_root, 'ec2vae_format.npy'))
    #pick one randomly
    shifted_prChordSet, scale_1, table = chord_shift(prChordSet)
    shifted_prPianoRollSet, scale_2, table = piano_roll_shift(prPianoRollSet)
    assert(scale_1 == scale_2)
    candidates, scores = chord_comparison(pr_chord_phrase, shifted_prChordSet, num_candidate=100)
    pr = piano_roll_phrase[::2].reshape((-1, piano_roll_phrase.shape[2]))
    print(pr.shape)
    candidates, scores = appearanceMatch(pr, candidates, shifted_prPianoRollSet)
        
    print(candidates[:10], scores[:10])
    shift = table[candidates[3] // scale_2]
    idx = candidates[3] % scale_2
    print(shift, idx)

    pr_matrix = prMatrixSet[idx]
    pr_matrix = np.transpose(pr_matrix)
    prPianoRoll = prPianoRollSet[idx]
    #print(prPianoRoll.shape)
    piano = prPianoRoll[:, :128]
    rhythm = prPianoRoll[:, 128:130]
    chord = prPianoRoll[:, 130:]
    shifted_piano = np.roll(piano, shift, axis=-1)
    shifted_chord = np.roll(chord, shift, axis=-1)
    shifted_prPianoRoll = np.concatenate((shifted_piano, rhythm, shifted_chord), axis=-1)
    
    #shift = register_shift(piano_roll_phrase, prPianoRoll)
    #print(shift)
    pr_matrix_shifted = np.roll(pr_matrix, shift, axis=-1)

    pr_matrix = QueryProcessor.numpySplit(pr_matrix, 0, 8, 32, 16, 128)
    pr_matrix_shifted = QueryProcessor.numpySplit(pr_matrix_shifted, 0, 8, 32, 16, 128)
    #init midi render
    midi_render = midiRender()
    #load corpus

    query_midi = midi_render.leadSheet_recon(piano_roll_phrase[::2].reshape((-1, 142)), tempo=120, start_time=0)
    
    #retrieved acc
    acc_original = midi_render.pr_matrix2note(pr_matrix, 120)
    acc_original_shifted = midi_render.pr_matrix2note(pr_matrix_shifted, 120)
    #poly-disentangle
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/model_master_final.pt')
    model.load_state_dict(checkpoint)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    pr_matrix = torch.from_numpy(pr_matrix).float().cuda()
    pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
    gt_chord_phrase = torch.from_numpy(pr_chord_phrase).float().cuda()
    est_x = model.inference(pr_matrix, gt_chord_phrase, sample=False)
    est_x_shifted = model.inference(pr_matrix_shifted, gt_chord_phrase, sample=False)
    #Generate accompaniment
    midiReGen = midi_render.accomapnimentGeneration(piano_roll_phrase, est_x, 120)
    midiReGen_shifted = midi_render.accomapnimentGeneration(piano_roll_phrase, est_x_shifted, 120)
    midiReGen.write('accompaniment_test_1.mid')
    midiReGen_shifted.write('accompaniment_test_1_shifted.mid')
    acc_original.write('accompaniment_origional_1.mid')
    acc_original_shifted.write('accompaniment_origional_1_shifted.mid')
    query_midi.write('query_leadsheet.mid')
    target_midi = midi_render.leadSheet_recon(shifted_prPianoRoll, tempo=120, start_time=0)
    target_midi.write('target_leadsheet.mid')

def search_withRhythm_for_leadSheet():
    #define query
    #song_name, segmentation, note_shift = "Morning Star.mid", 'A4A4B8B8\n', 1
    #song_name, segmentation, note_shift = "The 29th of May.mid", 'A4A4B8\n', 0
    #song_name, segmentation, note_shift = "Barry's Favourite.mid", 'A8A8B8C8\n', 1
    #song_name, segmentation, note_shift = 'Boggy Brays.mid', 'A8A8B8B8\n', 0
    #song_name, segmentation, note_shift = 'Bobbin Mill Reel.mid', 'A8A8B8C8\n', 0.5
    #song_name, segmentation, note_shift = 'Chestnut Reel.mid', 'A8A8B8C8\n', 1.5
    #song_name, segmentation, note_shift = 'Cuillin Reel.mid', 'A4A4B8B8\n', 1
    #song_name, segmentation, note_shift = 'The Dance of the Polygon.mid', 'A8B8A8B8\n', 0
    #song_name, segmentation, note_shift = 'Espresso Polka.mid', 'A8A8B8B8\n', 0.5
    song_name, segmentation, note_shift = 'Falling About.mid', 'A8A8B8B8\n', 0
    #song_name, segmentation, note_shift = 'Flapjack.mid', 'A8A8B8B8\n', 1
    #song_name, segmentation, note_shift = 'Hopwas Hornpipe.mid', 'A8A8B8C8\n', 1
    #song_name, segmentation, note_shift = 'Beaux of Oakhill.mid', 'A8A8B8B8\n', 1
    #song_name, segmentation, note_shift = 'Lord Moira.mid', 'A4A4B8\n', 0.5
    song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_Dual-track-Direct'
    #song_root='C:/Users/lenovo/Desktop/updated_data/new_midi_chord'
    #get query phrase
    songData = QueryProcessor(song_name=song_name, song_root=song_root, note_shift=note_shift)
    pianoRoll, _, TIV = songData.melody2pianoRoll()
    chord_table = songData.chord2chordTable()
    #print(chord_table[0])
    #print(chord_table[-1])

    prPianoRollSet, prMatrixSet, prChordSet, index_collection, config_collection, score_collection = traverse(segmentation, require_accChord=True)
    if prPianoRollSet.shape[1] < pianoRoll.shape[0]:
        pianoRoll = pianoRoll[:prPianoRollSet.shape[1], :]
        chord_table = chord_table[:prPianoRollSet.shape[1]//4, :]
    #print(prPianoRollSet.shape, pianoRoll.shape, chord_table.shape)
    print(len(index_collection))
    SCORE_Threshold = -1
    prPianoRollSet = prPianoRollSet[np.array(score_collection) >= SCORE_Threshold]
    prMatrixSet = prMatrixSet[np.array(score_collection) >= SCORE_Threshold]
    prChordSet = prChordSet[np.array(score_collection) >= SCORE_Threshold]
    #print(prPianoRollSet.shape, prMatrixSet.shape, prChordSet.shape)
    index_collection = np.array(index_collection)[np.array(score_collection) >= SCORE_Threshold].tolist()
    config_collection = np.array(config_collection)[np.array(score_collection) >= SCORE_Threshold].tolist()
    #print(config_collection)
    
    shifted_prChordSet, scale, table = chord_shift(prChordSet)
    #print(scale, table)
    #shifted_prPianoRollSet, scale, table = piano_roll_shift(prPianoRollSet)

    candidates, shifts, scores_chord, recorder_chord = chord_comparison(chord_table, shifted_prChordSet, segmentation, num_candidate = min(100, scale))
    sub_prPianoRollSet = prPianoRollSet[candidates]

    pr_rhythmSet = np.concatenate((np.sum(sub_prPianoRollSet[ :, :, :128], axis=-1, keepdims=True), sub_prPianoRollSet[ :, :, 128: 129]), axis=-1)#num_total*n_step
    #pr_rhythmSet = np.diff(sub_prPianoRollSet[ :, :, 128])
    #print(prPianoRollSet[1712, 125])
    """for batch in range(pr_rhythmSet.shape[0]):
        for i in range(pr_rhythmSet.shape[1]):
            if pr_rhythmSet[batch][i] == 1:
                j = i + 1
                while j < prPianoRollSet.shape[1] and prPianoRollSet[batch, j, 128] != 0:
                    pr_rhythmSet[batch][i] += 1 
                    j += 1"""
    #pr_rhythmSet = np.concatenate((pr_rhythmSet, prPianoRollSet[:, :, 128: 130]), -1)
    ##pick one randomly
    #shifted_prChordSet, scale_1, table = chord_shift(prChordSet)
    #shifted_prPianoRollSet, scale_2, table = piano_roll_shift(prPianoRollSet)
    #assert(scale_1 == scale_2)
    #candidates, scores = chord_comparison(pr_chord_phrase, shifted_prChordSet, num_candidate=100)
    pr_rhythm = np.concatenate((np.sum(pianoRoll[ :, :128], axis=-1, keepdims=True), pianoRoll[ :, 128: 129]), axis=-1)#n_step
    #pr_rhythm = np.diff(pianoRoll[ :, 128])
    """for i in range(pr_rhythm.shape[0]):
        if pr_rhythm[i] == 1:
            j = i + 1
            while j < pr.shape[0] and pr[j, 128] != 0:
               pr_rhythm[i] += 1 
               j += 1"""

    #pr_rhythm = np.concatenate((pr_rhythm, pr[:, 128: 130]), -1)
    #print(pr.shape)
    #candidates, scores = appearanceMatch(pr, candidates, shifted_prPianoRollSet)
    
    candidates_idx, scores_rhythm, recorder_rhythm = cosine_2d(pr_rhythm, pr_rhythmSet, segmentation, record_chord=recorder_chord, num_candidate = 100)

    indices = [index_collection[candidates[term]] for term in candidates_idx]
    print(len(indices), indices)
    #print('candidates:', candidates)
    #print(candidates_idx, scores_chord)
    #shift = table[candidates[3] // scale_2]
    #idx = candidates[3] % scale_2
    #print(shift, idx)
    SELECTION =0 #int(len(indices)*0.15)
    idx = candidates[candidates_idx[SELECTION]]
    shift = [table[shifts[idx][i]] for i in range(shifts.shape[1])]
    index = index_collection[idx]

    record_chord = [rec[idx] for rec in recorder_chord]
    recorde_rhythm = [rec[candidates_idx[SELECTION]] for rec in recorder_rhythm]

    
    print(shift, index, SELECTION)
    print('record_chord:', record_chord)
    print('record_rhythm:', recorde_rhythm)
    

    pr_matrix = prMatrixSet[idx]
    pr_matrix = np.transpose(pr_matrix)
    prPianoRoll = prPianoRollSet[idx]

    shifted_prPianoRoll = np.empty((0, prPianoRoll.shape[-1]))
    pr_matrix_shifted = np.empty((0, pr_matrix.shape[-1]))
    #print(prPianoRoll.shape)
    start = 0
    count = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 16
            piano = prPianoRoll[start: end, :128]
            rhythm = prPianoRoll[start: end, 128:130]
            chord = prPianoRoll[start: end, 130:]
            shifted_piano = np.roll(piano, shift[count], axis=-1)
            shifted_chord = np.roll(chord, shift[count], axis=-1)
            _shifted_prPianoRoll = np.concatenate((shifted_piano, rhythm, shifted_chord), axis=-1)
            _pr_matrix_shifted = np.roll(pr_matrix[start: end], shift[count], axis=-1)
            shifted_prPianoRoll = np.concatenate((shifted_prPianoRoll, _shifted_prPianoRoll), axis=0)
            pr_matrix_shifted = np.concatenate((pr_matrix_shifted, _pr_matrix_shifted), axis=0)
            start = end
            count += 1
    
    #shift = register_shift(piano_roll_phrase, prPianoRoll)
    #print(shift)

    #pianoRoll = QueryProcessor.numpySplit(pianoRoll, 32, 16, 142)
    
    
    #init midi render
    midi_render = midiRender()
    query_midi = midi_render.leadSheet_recon(pianoRoll, tempo=120, start_time=0)
    
    pr_matrix = QueryProcessor.numpySplit(pr_matrix, 32, 16, 128)
    pr_matrix_shifted = QueryProcessor.numpySplit(pr_matrix_shifted, 32, 16, 128)
    pianoRoll = QueryProcessor.numpySplit(pianoRoll, 32, 16, 142)
    gt_chord = QueryProcessor.chordSplit(chord_table, 8, 4)
    
    #retrieved acc
    acc_original = midi_render.pr_matrix2note(pr_matrix, 120)
    acc_original_shifted = midi_render.pr_matrix2note(pr_matrix_shifted, 120)
    #poly-disentangle
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/model_master_final.pt')
    model.load_state_dict(checkpoint)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    pr_matrix = torch.from_numpy(pr_matrix).float().cuda()
    pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
    gt_chord = torch.from_numpy(gt_chord).float().cuda()
    est_x = model.inference(pr_matrix, gt_chord, sample=False)
    est_x_shifted = model.inference(pr_matrix_shifted, gt_chord, sample=False)
    #Generate accompaniment
    midiReGen = midi_render.accomapnimentGeneration(pianoRoll, est_x, 120)
    midiReGen_shifted = midi_render.accomapnimentGeneration(pianoRoll, est_x_shifted, 120)
    midiReGen.write('accompaniment_test_1.mid')
    midiReGen_shifted.write('accompaniment_test_1_shifted.mid')
    acc_original.write('accompaniment_origional_1.mid')
    acc_original_shifted.write('accompaniment_origional_1_shifted.mid')
    query_midi.write('query_leadsheet.mid')
    target_midi = midi_render.leadSheet_recon(shifted_prPianoRoll, tempo=120, start_time=0)
    #target_midi = midi_render.leadSheet_recon(prPianoRoll, tempo=120, start_time=0)
    target_midi.write('target_leadsheet.mid')

def statistics():
    statistics = {}
    #song_name, segmentation, note_shift = 'Lord Moira.mid', 'A4A4B8\n', 0.5
    #song_root='../produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_Dual-track-Direct'
    song_root='C:/Users/lenovo/Desktop/updated_data/new_midi_chord'
    #get query phrase
    with open('C:/Users/lenovo/Desktop/updated_data/phrase.txt', 'r') as f:
        song_collections = f.readlines()
    for line in tqdm(song_collections):
        try:
            song_name, segmentation, note_shift = line.split('\t')
            song_name = song_name+'.mid'
            segmentation = segmentation+'\n'
            note_shift = float(note_shift.strip('\n'))

            songData = QueryProcessor(song_name=song_name, song_root=song_root, note_shift=note_shift)
            pianoRoll, _, TIV = songData.melody2pianoRoll()
            chord_table = songData.chord2chordTable()
        

            prPianoRollSet, prMatrixSet, prChordSet, index_collection, config_collection, score_collection = traverse(segmentation, require_accChord=True)
            if prPianoRollSet.shape[1] < pianoRoll.shape[0]:
                pianoRoll = pianoRoll[:prPianoRollSet.shape[1], :]
                chord_table = chord_table[:prPianoRollSet.shape[1]//4, :]
            #print(prPianoRollSet.shape, pianoRoll.shape, chord_table.shape)
            print(len(index_collection))
            SCORE_Threshold = -1
            prPianoRollSet = prPianoRollSet[np.array(score_collection) >= SCORE_Threshold]
            prMatrixSet = prMatrixSet[np.array(score_collection) >= SCORE_Threshold]
            prChordSet = prChordSet[np.array(score_collection) >= SCORE_Threshold]
            #print(prPianoRollSet.shape, prMatrixSet.shape, prChordSet.shape)
            index_collection = np.array(index_collection)[np.array(score_collection) >= SCORE_Threshold].tolist()
            config_collection = np.array(config_collection)[np.array(score_collection) >= SCORE_Threshold].tolist()
            
            shifted_prChordSet, scale, table = chord_shift(prChordSet)

            candidates, shifts, scores_chord, recorder_chord = chord_comparison(chord_table, shifted_prChordSet, segmentation, num_candidate = min(100, scale))
            sub_prPianoRollSet = prPianoRollSet[candidates]

            pr_rhythmSet = np.concatenate((np.sum(sub_prPianoRollSet[ :, :, :128], axis=-1, keepdims=True), sub_prPianoRollSet[ :, :, 128: 129]), axis=-1)#num_total*n_step

            pr_rhythm = np.concatenate((np.sum(pianoRoll[ :, :128], axis=-1, keepdims=True), pianoRoll[ :, 128: 129]), axis=-1)#n_step
            
            candidates_idx, scores_rhythm, recorder_rhythm = cosine_2d(pr_rhythm, pr_rhythmSet, segmentation, record_chord=recorder_chord, num_candidate = 100)

            #indices = [index_collection[candidates[term]] for term in candidates_idx]

            #print(candidates[candidates_idx], scores)

            for i in range(min(int(len(candidates_idx)*0.15), scale)):

                idx = candidates[candidates_idx[i]]
                index = index_collection[idx]

                record_chord = [rec[idx] for rec in recorder_chord]
                recorde_rhythm = [rec[candidates_idx[i]] for rec in recorder_rhythm]
                if not index in statistics:
                    statistics[index] = {}
                    statistics[index]['hit'] = 0
                    statistics[index]['score'] = 0
                statistics[index]['hit'] += 1
                statistics[index]['score'] += (np.mean(record_chord) + np.mean(recorde_rhythm) ) / 2
        except:
            print('bad')
            continue
    for key in statistics:
        print(key, statistics[key]['hit'], statistics[key]['score'], statistics[key]['score']/statistics[key]['hit'])
    

if __name__ == '__main__':
    #assembly_chordfunc()
    #song = pyd.PrettyMIDI('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_midi/2nd part for B music.mid')
    #print(song.get_downbeats())
    #root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat_withChord'
    #for item in os.listdir(root):
    #    song = pyd.PrettyMIDI(os.path.join(root, item))
    #    print(song.key_signature_changes)
    #search_for_leadSheet(num_candidate=0)
    #assembly_zp()
    """
    #test get_query and leadsheet recon
    piano_roll, pr_chord_phrase = get_queryPhrase(song_name='Abram Circle.mid', start_downbeat=0, end_downbeat=8, note_shift=1)
    print(pr_chord_phrase.shape)
    melody_notes, chord_notes = numpy2melodyAndChord(piano_roll)
    recon = pyd.PrettyMIDI(initial_tempo=120)
    melody_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    chord_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    melody_track.notes = melody_notes
    chord_track.notes = chord_notes
    recon.instruments.append(melody_track)
    recon.instruments.append(chord_track)
    recon.write('pop909_test.mid')
    """
    #search_random_for_leadSheet()
    #search_withChord_for_leadSheet()
    #search_withMeldoy_for_leadSheet()
    search_withRhythm_for_leadSheet()
    #statistics()
    #traverse(query_segmentation='A8A8B8B8\n')

    
