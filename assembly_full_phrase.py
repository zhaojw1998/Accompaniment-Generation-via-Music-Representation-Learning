import sys

sys.path.append('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement')
from model import DisentangleVAE
from ptvae import PtvaeDecoder

sys.path.append('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/code')
from EC2model import VAE

import pretty_midi as pyd 
import numpy as np
import os
from tqdm import tqdm
from jingwei_dataProcessor import songDataProcessor, melody_processor
import converter
import torch
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
        processor = melody_processor()
        melodySeq = processor.getMelodySeq_byBeats(self.melody_track, self.downbeats)
        #print('1-')
        chordSeq = processor.getChordSeq_byBeats(self.chord_track, self.downbeats)
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
        chromas = {'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]}
        distr2quality = {'(4, 3)': 'maj/0', 
                         '(3, 5)': 'maj/1', 
                         '(5, 4)': 'maj/2', 
                         '(3, 4)': 'min/0', 
                         '(4, 5)': 'min/1', 
                         '(5, 3)': 'min/2'}
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
                    assert(len(chord_set) == 3)
                    quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1])))]
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
            assert(len(chord_set) == 3)
            quality = distr2quality[str(((chord_set[1]-chord_set[0]), (chord_set[2]-chord_set[1])))]
            chroma = chromas[quality.split('/')[0]]
            root = chord_set[-int(quality.split('/')[-1])] % 12
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
    def numpySplit(self, matrix, start_downbeat, end_downbeat, WINDOWSIZE=32, HOPSIZE=16, VECTORSIZE=142):
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

    def chordSplit(self, chord, start_downbeat, end_downbeat, WINDOWSIZE=8, HOPSIZE=8):
        splittedChord = np.empty((0, 8, 36))
        #print(matrix.shape[0])
        for idx_T in range(start_downbeat*4, (end_downbeat-1)*4, HOPSIZE):
            if idx_T > chord.shape[0]-8:
                break
            sample = chord[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedChord = np.concatenate((splittedChord, sample), axis=0)
        return splittedChord

def numpy2melodyAndChord(matrix, tempo=120, start_time=0.0):
    processor = melody_processor()
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
        processor = melody_processor()
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
        for idx in range(0, piano_roll.shape[0], 2):
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
    pianoRoll_phrase = songData.numpySplit(pianoRoll, start_downbeat, end_downbeat, 32, 16, 142)
    #TIV_phrase = songData.numpySplit(TIV, start_downbeat, end_downbeat, 32, 16, 12)
    chord_table = songData.chord2chordTable()
    pr_chord_phrase = songData.chordSplit(chord_table, start_downbeat, end_downbeat, 8, 4)
    return pianoRoll_phrase, pr_chord_phrase

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
def chord_comparison(pr_chord_phrase, shifted_ensemble, num_candidate=10):
    #pr_chord_phrase: batch*8*36; prChordSet: batch*32*14
    if len(pr_chord_phrase.shape) == 3:
        pr_chord_phrase = pr_chord_phrase[::2].reshape((-1, 36))[:, 12:24]
    #print(pr_chord_phrase)
    """use TIV"""
    pr_chord_phrase = computeTIV(pr_chord_phrase)
    shifted_ensemble = computeTIV(shifted_ensemble)
    #print(pr_chord_phrase.shape, shifted_ensemble.shape)
    """use chroma"""
    #pr_chord_phrase = pr_chord_phrase
    #shifted_ensemble = shifted_ensemble
    candidates, scores = cosine(pr_chord_phrase, shifted_ensemble, num_candidate = num_candidate)
    #print(pr_chord_phrase.shape, shifted_ensemble.shape)
    #print(candidates, scores)
    return candidates, scores
def chord_shift(prChordSet):
    prChordSet = prChordSet[:, :, 1: -1]
    num_total = prChordSet.shape[0]
    shift_const = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    shifted_ensemble = []
    for i in shift_const:
        shifted_term = np.roll(prChordSet, i, axis=-1)
        shifted_ensemble.append(shifted_term)
    shifted_ensemble = np.array(shifted_ensemble).reshape((-1, prChordSet.shape[1], prChordSet.shape[2]))
    return shifted_ensemble, num_total, shift_const
def computeTIV(chroma):
    #inpute size: Time*12
    #chroma = chroma.reshape((chroma.shape[0], -1, 12))
    #print('chroma', chroma.shape)
    if (len(chroma.shape)) == 3:
        num_batch = chroma.shape[0]
        chroma = chroma.reshape((-1, 12))
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        #print(TIV.shape)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
        TIV = TIV.reshape((num_batch, -1, 12))
    else:
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        #print(TIV.shape)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
    return TIV #Time * 12
def cosine(query, instance_space, threshold=0.8, num_candidate = 10):
    #query: T * 12
    #instance space: Batch * T * 12
    #instance_space: batch * vectorLength
    result = np.dot(instance_space, np.transpose(query))/(np.linalg.norm(instance_space, axis=-1, keepdims=True) * np.transpose(np.linalg.norm(query, axis=-1, keepdims=True)) + 1e-10)
    #print(result.shape)
    result = (result >= threshold) * 1
    result = np.trace(result, axis1=-2, axis2=-1)
    #print(result.shape)
    candidates = result.argsort()[::-1][:num_candidate]
    scores = result[candidates]
    #names = [os.listdir('./scrape_musescore/data_to_be_used/8')[i] for i in candidates]
    #sort by edit distance over melody
    #candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
    return candidates, scores#, query[::4], instance_space[candidates][:, ::4]

def cosine_1d(query, instance_space, num_candidate = 10):
    #query: T
    #instance space: Batch * T
    #instance_space: batch * vectorLength
    result = np.dot(instance_space, query)/(np.linalg.norm(instance_space, axis=-1) * np.linalg.norm(query) + 1e-10)
    print(result.shape)
    #result = (result >= threshold) * 1
    #result = np.trace(result, axis1=-2, axis2=-1)
    #print(result.shape)
    candidates = result.argsort()[::-1][:num_candidate]
    scores = result[candidates]
    #names = [os.listdir('./scrape_musescore/data_to_be_used/8')[i] for i in candidates]
    #sort by edit distance over melody
    #candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
    return candidates, scores#, query[::4], instance_space[candidates][:, ::4]

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
    random_number = np.random.randint(prMatrixSet.shape[0])
    print('random number:', random_number)
    pr_matrix = prMatrixSet[random_number]
    pr_matrix = np.transpose(pr_matrix)
    prPianoRoll = prPianoRollSet[random_number]
    #print(prPianoRoll.shape)
    #chord_comparison(pr_chord_phrase, prChordSet)
    shift = register_shift(piano_roll_phrase, prPianoRoll)
    print(shift)
    shift= np.sign(shift) * (abs(shift)%12)
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
    #song_name, start_bar, end_bar, note_shift = 'Abram Circle.mid', 0, 8, 1
    #song_name, start_bar, end_bar, note_shift = '2nd part for B music.mid', 0, 8, 0
    song_name, start_bar, end_bar, note_shift = 'Boggy Brays.mid', 0, 8, 0
    #get query phrase
    piano_roll_phrase, pr_chord_phrase = get_queryPhrase(song_name, start_bar, end_bar, note_shift)
    phrase_length = end_bar - start_bar
    #load arrangement set
    phrase_root = os.path.join('../POP909-phrase', str(phrase_length))
    prMatrixSet = np.load(os.path.join(phrase_root, 'pr_matrix.npy'))
    prChordSet = np.load(os.path.join(phrase_root, 'pr_chord.npy'))
    prPianoRollSet = np.load(os.path.join(phrase_root, 'ec2vae_format.npy'))
    
    shifted_prChordSet, scale, table = chord_shift(prChordSet)
    shifted_prPianoRollSet, scale, table = piano_roll_shift(prPianoRollSet)
    #pr_chord_phrase = shifted_prChordSet[20000]
    candidates, scores = chord_comparison(pr_chord_phrase, shifted_prChordSet, num_candidate = 100)
    sub_prPianoRollSet =shifted_prPianoRollSet[candidates]
    
    pr_rhythmSet = sub_prPianoRollSet[ :, :, :128].sum(-1) #num_total*n_step
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
    pr = piano_roll_phrase[::2].reshape((-1, piano_roll_phrase.shape[2]))
    pr_rhythm = pr[ :, :128].sum(-1) #n_step
    """for i in range(pr_rhythm.shape[0]):
        if pr_rhythm[i] == 1:
            j = i + 1
            while j < pr.shape[0] and pr[j, 128] != 0:
               pr_rhythm[i] += 1 
               j += 1"""

    #pr_rhythm = np.concatenate((pr_rhythm, pr[:, 128: 130]), -1)
    #print(pr.shape)
    #candidates, scores = appearanceMatch(pr, candidates, shifted_prPianoRollSet)
    
    candidates_idx, scores = cosine_1d(pr_rhythm, pr_rhythmSet, num_candidate = 10)

    print(candidates[candidates_idx], scores)
    #shift = table[candidates[3] // scale_2]
    #idx = candidates[3] % scale_2
    #print(shift, idx)
    idx = candidates[candidates_idx[0]]
    shift = table[idx // scale]
    idx = idx % scale
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
    #target_midi = midi_render.leadSheet_recon(prPianoRoll, tempo=120, start_time=0)
    target_midi.write('target_leadsheet.mid')
    

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
    search_random_for_leadSheet()
    #search_withChord_for_leadSheet()
    #search_withMeldoy_for_leadSheet()
    #search_withRhythm_for_leadSheet()

    
