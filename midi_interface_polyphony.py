
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
        
    def Midi2PrMatrix_byBeats(self, midi_data):
        time_stamps = list(midi_data.get_downbeats())
        time_stamps.append(time_stamps[-1]+(time_stamps[-1]-time_stamps[-2]))
        time_stamp_sixteenth_reso = []
        delta_set = []
        for i in range(len(time_stamps)-1):
            s_curr = time_stamps[i]
            s_next = time_stamps[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                time_stamp_sixteenth_reso.append(s_curr + delta * i)
                delta_set.append(delta)
        time_stamp_sixteenth_reso = np.array(time_stamp_sixteenth_reso)

        pr_matrix = np.zeros((128, time_stamp_sixteenth_reso.shape[0]))
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                onset = note.start
                t = np.argmin(np.abs(time_stamp_sixteenth_reso - onset))
                p = note.pitch
                duration = int(round((note.end - onset) / delta_set[t]))
                #print(t, p, duration)
                pr_matrix[p, t] = duration
        return pr_matrix

    def PrMatrix2Midi(self, PrMatrix, ts, ks, tempo=120, fourthNote_reso=4):
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
        midiReGen.time_signature_changes = ts
        midiReGen.key_signature_changes = ks
        return midiReGen

    def getChordSeq_byBeats(self, midi_data):
        last_time = 0
        chord_set = []
        chord_time = [0.0, 0.0]
        chordsRecord = []
        for note in midi_data.instruments[1].notes:
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
        time_stamps = list(midi_data.get_downbeats())
        time_stamps.append(time_stamps[-1]+(time_stamps[-1]-time_stamps[-2]))
        for i in range(len(time_stamps)-1):
            s_curr = time_stamps[i]
            s_next = time_stamps[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                while chord['end'] <= (s_curr + i * delta) and anchor < len(chordsRecord)-1:
                    #if anchor >= len(midi_data.instruments[0].notes)-1:
                    #    start = 1e5
                    #    break
                    anchor += 1
                    chord = chordsRecord[anchor]
                    start = chord['start']
                if s_curr + i * delta < start:
                    ChordSequence.append('NC')
                else:
                    ChordSequence.append(chord['chord'])
        return ChordSequence
    
    def midiReconFromSeq(self, melodySequence, ChordSequence, tempo):
        minStep = 60 / tempo / 4
        midiRecon = pyd.PrettyMIDI(initial_tempo=tempo)
        program = pyd.instrument_name_to_program('Violin')
        melody = pyd.Instrument(program=program)
        program = pyd.instrument_name_to_program('Acoustic Grand Piano')
        chord = pyd.Instrument(program=program)
        onset_or_rest = [i for i in range(len(melodySequence)) if not melodySequence[i]==self.hold_pitch]
        onset_or_rest.append(len(melodySequence))
        for idx, onset in enumerate(onset_or_rest[:-1]):
            if melodySequence[onset] == self.rest_pitch:
                continue
            else:
                pitch = melodySequence[onset]
                start = onset * minStep
                end = onset_or_rest[idx+1] * minStep
                noteRecon = pyd.Note(velocity=100, pitch=pitch, start=start, end=end)
                melody.notes.append(noteRecon)
        
        onset_or_rest = [0]
        onset_or_rest_ = [i for i in range(1, len(ChordSequence)) if ChordSequence[i] != ChordSequence[i-1] ]
        onset_or_rest = onset_or_rest + onset_or_rest_
        onset_or_rest.append(len(ChordSequence))
        #print(onset_or_rest)
        for idx, onset in enumerate(onset_or_rest[:-1]):
            if ChordSequence[onset] == 'NC':
                continue
            else:
                chordset = self.cl.name2note(ChordSequence[onset])
                if chordset == None:
                    continue
                start = onset * minStep
                end = onset_or_rest[idx+1] * minStep
                for note in chordset:
                    noteRecon = pyd.Note(velocity=100, pitch=note+4*12, start=start, end=end)
                    chord.notes.append(noteRecon)

        midiRecon.instruments.append(melody)
        midiRecon.instruments.append(chord)
        #midiRecon.write('melody_and_chord_reconTest1.mid')
        return midiRecon

    def seq2Numpy(self, melodySequence, chordSequence, ROLL_SIZE=130, CHORD_SIZE=12):
        assert(len(melodySequence) == len(chordSequence))
        melodyMatrix = np.zeros((len(melodySequence), ROLL_SIZE))
        chordMatrix = np.zeros((len(chordSequence), CHORD_SIZE))
        for idx, note in enumerate(melodySequence):
            melodyMatrix[idx, note] = 1
            chordName = chordSequence[idx]
            chordset = self.cl.name2note(chordName)
            if chordset == None:
                continue
            for idxP in chordset:
                chordMatrix[idx, idxP%12] = 1
        return np.concatenate((melodyMatrix, chordMatrix), axis=-1)

    def midiReconFromNumpy(self, matrix, tempo, ROLL_SIZE=130, CHORD_SIZE=12):
        melodyMatrix = matrix[:, :ROLL_SIZE]
        chordMatrix = matrix[:, ROLL_SIZE:]
        melodySequence = [np.argmax(melodyMatrix[i]) for i in range(melodyMatrix.shape[0])]
        chordSequence = []
        for i in range(chordMatrix.shape[0]):
            chordset = [idx for idx in range(CHORD_SIZE) if chordMatrix[i][idx] == 1]
            chordSequence.append(self.cl.note2name(chordset))
            #print(chordset)
        return self.midiReconFromSeq(melodySequence, chordSequence, tempo)
    
    def numpySplit(self, matrix, WINDOWSIZE=32, HOPSIZE=16):
        splittedMatrix = np.empty((0, WINDOWSIZE, 142))
        #print(matrix.shape[0])
        for idx_T in range(0, matrix.shape[0]-WINDOWSIZE, HOPSIZE):
            sample = matrix[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
            splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
        return splittedMatrix
        
class accompanySelection(midi_interface_polyphony):
    def __init__(self, save_root='./'):
        super(accompanySelection, self).__init__()
        self.save_root = save_root
        self.raw_midi_set = []
        self.minStep_set = []
        self.tempo_set = []
        self.start_record = []
        self.melodySequence_set = []
        self.npy_midi_set = []
        self.belonging = []
        self.num_beforeShift = None
    
    def load_dataset(self, dataset_doubleTrack, dataset_makeChord):
        print('begin loading dataset')
        for midi in tqdm(os.listdir(dataset_doubleTrack)):
            #store raw data and relevant infomation
            midi_data = pyd.PrettyMIDI(os.path.join(dataset_doubleTrack, midi))
            self.raw_midi_set.append(midi_data)
            tempo = np.mean(midi_data.get_tempo_changes()[-1])
            self.tempo_set.append(tempo)
            minStep = 60 / tempo / 4
            self.minStep_set.append(minStep)
            start = midi_data.instruments[0].notes[0].start
            self.start_record.append(start)
            #convert to and store numpy matrix
            mididata_withChord = pyd.PrettyMIDI(os.path.join(dataset_makeChord, midi))
            melodySequence = self.getMelodySeq_byBeats(mididata_withChord)
            chordSequence = self.getChordSeq_byBeats(mididata_withChord)
            self.melodySequence_set.append(melodySequence)
            melodyMatrix = self.seq2Numpy(melodySequence, chordSequence)
            self.npy_midi_set.append(melodyMatrix)
    
    def EC2_VAE_batchData(self):
        numTotal = len(self.npy_midi_set)
        print(numTotal)
        NumMiniBatch = numTotal // 10
        print('begin generating batch data for EC2-VAE')
        for part, idx_B in enumerate(range(0, numTotal-NumMiniBatch, NumMiniBatch)):
            batchData = np.empty((0, 32, 142))
            sub_midi_set = self.npy_midi_set[idx_B: idx_B+NumMiniBatch]
            for idx in tqdm(range(len(sub_midi_set))):
                numpyMatrix = sub_midi_set[idx]
                for idxT in range(0, numpyMatrix.shape[0]-32, 16):
                    sample = numpyMatrix[idxT:idxT+32, :][np.newaxis, :, :]
                    if sample[0, 0, 128] == 1:
                        for idx_forward in range(idxT,0, -1):
                            note = self.melodySequence_set[part*NumMiniBatch+idx][idx_forward]
                            if note != 128:
                                break
                        sample[0, 0, 128] = 0
                        sample[0, 0, note] = 1
                    batchData = np.concatenate((batchData, sample), axis=0)
                    self.belonging.append((part*NumMiniBatch+idx, idxT))
            save_name = 'batchData_withChord_part%d.npy'%part
            np.save(os.path.join(self.save_root, save_name), batchData)
            print(batchData.shape)
            #print(len(self.belonging))
        
        batchData = np.empty((0, 32, 142))
        sub_midi_set = self.npy_midi_set[idx_B+NumMiniBatch:]
        for idx in tqdm(range(len(sub_midi_set))):
            numpyMatrix = sub_midi_set[idx]
            for idxT in range(0, numpyMatrix.shape[0]-32, 16):
                sample = numpyMatrix[idxT:idxT+32, :][np.newaxis, :, :]
                if sample[0, 0, 128] == 1:
                    for idx_forward in range(idxT,0, -1):
                        note = self.melodySequence_set[(part+1)*NumMiniBatch+idx][idx_forward]
                        if note != 128:
                            break
                    sample[0, 0, 128] = 0
                    sample[0, 0, note] = 1
                batchData = np.concatenate((batchData, sample), axis=0)
                self.belonging.append(((part+1)*NumMiniBatch+idx, idxT))
        
        save_name = 'batchData_withChord_part%d.npy'%(part+1)
        np.save(os.path.join(self.save_root, save_name), batchData)
        print(batchData.shape)
        print('begin saving auxilary information')
        time1 = time.time()
        with open(os.path.join(self.save_root, 'auxilary_withChord.txt'), 'wb') as f:
            pickle.dump(self.raw_midi_set, f)
            pickle.dump(self.tempo_set, f)
            pickle.dump(self.minStep_set, f)
            pickle.dump(self.start_record, f)
            pickle.dump(self.belonging, f)
        duration = time.time() - time1
        print('finish, using time %.2fs'%duration)
        #return batchData
        
    def loadAuxilary(self, file_name):
        print('begin loading parameters')
        time1=time.time()
        with open(os.path.join(self.save_root, file_name), 'rb') as f:
            self.raw_midi_set = pickle.load(f)
            self.tempo_set = pickle.load(f)
            self.minStep_set = pickle.load(f)
            self.start_record = pickle.load(f)
            self.belonging = pickle.load(f)
            try:
                self.num_beforeShift = pickle.load(f)
                #self.num_beforeShift = 238444
                #print(self.num_beforeShift)
            except EOFError:
                self.num_beforeShift = None
        duration = time.time() - time1
        print('finish loading parameters, using time %.2fs'%duration)
        time.sleep(.5)

    def retriveRawMidi(self, batchIdx, batchFile='batchData_withChord_part0.npy'):
        if self.num_beforeShift == None:
            midiIdx, idxT = self.belonging[batchIdx]
            minStep = self.minStep_set[midiIdx]
            start = idxT*minStep
            end = (idxT+32)*minStep
            tempo = 60 / minStep / 4
            midi_file = self.raw_midi_set[midiIdx]
            midiRetrive = pyd.PrettyMIDI(initial_tempo=tempo)
            melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
            accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
            for note in midi_file.instruments[0].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start, end = min(note.end, end)-start)
                    melody.notes.append(note_recon)
            for note in midi_file.instruments[1].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start, end = min(note.end, end)-start)
                    accompany.notes.append(note_recon)
            midiRetrive.instruments.append(melody)
            midiRetrive.instruments.append(accompany)
        else:
            midiIdx, idxT = self.belonging[batchIdx % self.num_beforeShift]
            minStep = self.minStep_set[midiIdx]
            start = idxT*minStep
            end = (idxT+32)*minStep
            tempo = 60 / minStep / 4
            midi_file = self.raw_midi_set[midiIdx]
            midiRetrive = pyd.PrettyMIDI(initial_tempo=tempo)
            melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
            accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
            shift = 6 - (batchIdx // self.num_beforeShift)
            for note in midi_file.instruments[0].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch + shift, start = max(note.start, start)-start, end = min(note.end, end)-start)
                    melody.notes.append(note_recon)
            for note in midi_file.instruments[1].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch + shift, start = max(note.start, start)-start, end = min(note.end, end)-start)
                    accompany.notes.append(note_recon)
            midiRetrive.instruments.append(melody)
            midiRetrive.instruments.append(accompany)
        #print(accompany.notes)
        #midiRetrive.write('test_retrive.mid')
        #batchData = np.load(os.path.join(self.save_root, batchFile))
        #melodyRecon = self.midiReconFromNumpy(batchData[batchIdx, :, :], tempo)
        #melodyRecon.write('test_recon.mid')
        return midiRetrive

    def tone_shift(self):
        with open(os.path.join(self.save_root, 'auxilary_withChord.txt'), 'rb') as f:
            raw_midi_set = pickle.load(f)
            tempo_set = pickle.load(f)
            minStep_set = pickle.load(f)
            start_record = pickle.load(f)
            belonging = pickle.load(f)
        original_batchData = np.load(os.path.join(self.save_root, 'batchData_withChord.npy'))
        shifted_batchData = np.zeros((original_batchData.shape[0]*12, original_batchData.shape[1], original_batchData.shape[2]))
        for idx, i in enumerate(tqdm(range(-6, 6, 1))):
            #print(idx)
            tmpP = original_batchData[:, :, :128]
            #print(tmp.shape)
            tmpP = np.concatenate((tmpP[:, :, i:], tmpP[:, :, :i]), axis=-1)
            tmpC = original_batchData[:, :, 130:]
            tmpC = np.concatenate((tmpC[:, :, i:], tmpC[:, :, :i]), axis=-1)
            tmp = np.concatenate((tmpP, original_batchData[:, :, 128: 130], tmpC), axis=-1)
            (shifted_batchData[original_batchData.shape[0]*idx: original_batchData.shape[0]*(idx+1), :, :]) = tmp
        np.save(os.path.join(self.save_root, 'batchData_withChord_shifted.npy'), shifted_batchData)
        with open(os.path.join(self.save_root, 'auxilary_withChord_shifted.txt'), 'wb') as f:
            pickle.dump(raw_midi_set, f)
            pickle.dump(tempo_set, f)
            pickle.dump(minStep_set, f)
            pickle.dump(start_record, f)
            pickle.dump(belonging, f)
            pickle.dump(len(belonging), f)
        return len(belonging)

    
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