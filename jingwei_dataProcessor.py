import torch 
from torch.utils.data import Dataset, DataLoader 
import pretty_midi as pyd
from jingwei_midi_interface_mono_and_chord import midi_interface_mono_and_chord
from jingwei_midi_interface_polyphony import midi_interface_polyphony
import os
import converter
import numpy as np
from tqdm import tqdm

class melody_processor(midi_interface_mono_and_chord):
    def __init__(self, recogLevel='Seven'):
        super(melody_processor, self).__init__(recogLevel=recogLevel)

    def getMelodySeq_byBeats(self, melody_track, downbeats):
        melodySequence = []
        anchor = 0
        note = melody_track.notes[anchor]
        start = note.start
        new_note = True
        for i in range(len(downbeats)-1):
            s_curr = downbeats[i]
            s_next = downbeats[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                while note.end <= (s_curr + i * delta) and anchor < len(melody_track.notes)-1:
                    anchor += 1
                    note = melody_track.notes[anchor]
                    start = note.start
                    new_note = True
                if s_curr + i * delta < start:
                    melodySequence.append(self.rest_pitch)
                else:
                    if not new_note:
                        melodySequence.append(self.hold_pitch)
                    else:
                        melodySequence.append(note.pitch)
                        new_note = False
        return melodySequence

    def getChordSeq_byBeats(self, chord_track, downbeats):
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
                        chordsRecord.append({"start":last_time,"end": chord_time[0], "chord" : "NC"})
                    chordsRecord.append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                    last_time = chord_time[1]
                    chord_set = []
                    chord_set.append(note.pitch)
                    chord_time[0] = note.start
                    chord_time[1] = note.end 
        if len(chord_set) > 0:
            if last_time < chord_time[0]:
                chordsRecord.append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
            chordsRecord.append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
            last_time = chord_time[1]
        
        ChordSequence = []
        anchor = 0
        chord = chordsRecord[anchor]
        start = chord['start']
        for i in range(len(downbeats)-1):
            s_curr = downbeats[i]
            s_next = downbeats[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                while chord['end'] <= (s_curr + i * delta) and anchor < len(chordsRecord)-1:
                    anchor += 1
                    chord = chordsRecord[anchor]
                    start = chord['start']
                if s_curr + i * delta < start:
                    ChordSequence.append('NC')
                else:
                    ChordSequence.append(chord['chord'])
        return ChordSequence

class songDataProcessor(object):
    def __init__(self, texture_root, chord_root, song_idx_str):
        self.song = pyd.PrettyMIDI(os.path.join(chord_root, song_idx_str+'.mid'))
        self.downbeats = list(self.song.get_downbeats())
        self.downbeats.append(self.downbeats[-1] + (self.downbeats[-1] - self.downbeats[-2]))
        #print(self.downbeats[-100:])
        self.melody_track = self.song.instruments[0]
        self.chord_track = self.song.instruments[-1]
        self.polyphony_track = pyd.PrettyMIDI(os.path.join(texture_root, song_idx_str+'.mid')).instruments[-1]
        #self.splitted_grids, self.splitted_pr_matrix, self.splittedChord = self.texture2prMatrix()

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

    def texture2prMatrix(self, max_note_count=16):
        processor = midi_interface_polyphony(recogLevel='Seven')
        pr_matrix = processor.Midi2PrMatrix_byBeats(self.polyphony_track, self.downbeats)
        #splitted_pr_matrix = processor.numpySplit(pr_matrix, WINDOWSIZE=32, HOPSIZE=32)
        
        #splitted_grids =np.zeros(((splitted_pr_matrix.shape[0]), 32, max_note_count, 6))
        #for i in range(splitted_grids.shape[0]):
        #    splitted_grids[i] = converter.target_to_3dtarget(splitted_pr_matrix[i], 
        #                                                    max_note_count=max_note_count, 
        #                                                    max_pitch=128,
        #                                                    min_pitch=0,
        #                                                    pitch_pad_ind=130,
        #                                                    pitch_sos_ind=128,
        #                                                    pitch_eos_ind=129)

        chord = self.chordConverter(self.chord_track, self.downbeats)
        #splittedChord = self.chordSplit(chord)
        return pr_matrix, chord #pr_matrix is 16th-note resolution; chord is 4th-note resolution

    def chordConverter(self, chord_track, downbeats):
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
                    bass = chord_set[0] % 12
                    
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
            bass = chord_set[0] % 12
            root = chord_set[-int(quality.split('/')[-1])] % 12
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

    def chordSplit(self, chord, start_downbeat, end_downbeat, WINDOWSIZE=8, HOPSIZE=8):
        splittedChord = np.empty((0, 8, 36))
        #print(matrix.shape[0])
        for idx_T in range(start_downbeat*4, (end_downbeat-1)*4, HOPSIZE):
            if idx_T > chord.shape[0]-8:
                break
            sample = chord[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedChord = np.concatenate((splittedChord, sample), axis=0)
        return splittedChord


    def numpy2melodyAndChord(self, matrix, tempo=120, start_time=0.0):
        processor = melody_processor()
        return processor.midiReconFromNumpy(matrix=matrix, tempo=tempo, start_time=start_time)
    
    def numpy2textureTrack(self, pr_matrix, tempo=120, start_time=0.0):
        alpha = 0.25 * 60 / tempo
        notes = []
        for t in range(pr_matrix.shape[0]):
            for p in range(128):
                if pr_matrix[t, p] >= 1:
                    s = alpha * t + start
                    e = alpha * (t + pr_matrix[t, p]) + start
                    notes.append(pyd.Note(100, int(p), s, e))
        return notes

if __name__ == '__main__':
    test_root = './scrape_musescore/data_to_be_used/8/0016-10'
    piano_roll = np.load(os.path.join(test_root, 'batched_piano_roll.mid.npy'))
    pr_matrix = np.load(os.path.join(test_root, 'batched_pr_matrix.mid.npy'))
    assert(piano_roll.shape[0] == pr_matrix.shape[0])

    processed_midi_root_withChord = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musescore\\musescore_midi_processed_chord'
    processed_midi_root_withTexture = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musescore\\musescore_midi_processed'
    file_index = '0075'
    processor = songDataProcessor(texture_root=processed_midi_root_withTexture,
                                    chord_root=processed_midi_root_withChord, 
                                    song_idx_str=file_index)
    start = 0
    tempo = 120
    midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
    melody_track = pyd.Instrument(program=pyd.instrument_name_to_program('Violin'))
    texture_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    for idx in range(0, piano_roll.shape[0], 2):
        melody_notes, chord_notes = processor.numpy2melodyAndChord(matrix=piano_roll[idx], tempo=tempo, start_time=start)
        texture_notes = processor.numpy2textureTrack(pr_matrix=pr_matrix[idx], tempo=tempo, start_time=start)
        melody_track.notes += melody_notes
        texture_track.notes += texture_notes
        start += 60 / 120 * 8
    midiReGen.instruments.append(melody_track)
    midiReGen.instruments.append(texture_track)
    midiReGen.write('test.mid')



