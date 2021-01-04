import pretty_midi as pyd
import numpy as np
import os
from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys

class midi_processor(object):
    def __int__(self):
        pass
    
    def judgeMono(self, instrument, tempo):
        minOverlap = tempo / 60 #duration of a fourth note
        last_end = 0
        for note in instrument.notes:
            if last_end - note.start > minOverlap:
                return False
            last_end = note.end
        return True

    def judgeSequence(self, instrument):
        last_time = 0
        for note in instrument.notes:
            if note.start >= last_time:
                last_time = note.start
            else:
                return False
        return True

    def noteSort(self, instrument, programType):
        toSort = []
        for note in instrument.notes:
            toSort.append((note.pitch, note.start, note.end))
        afterSort = sorted(toSort, key=lambda note: note[1])
        insRecon = pyd.Instrument(program = pyd.instrument_name_to_program(programType))
        for note in afterSort:
            insRecon.notes.append(pyd.Note(velocity=100, pitch=note[0], start=note[1], end=note[2]))
        return insRecon
    
    def multi2doubleTrack(self, midi_data):
        tempi = midi_data.get_tempo_changes()[-1]
        assert(len(tempi) == 1)
        tempo = tempi[0]
        Melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
        Accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        for instrument in midi_data.instruments:
            if self.judgeMono(instrument, tempo):
                tmp = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
                for note in Melody.notes:
                    tmp.notes.append(note)
                for note in instrument.notes:
                    tmp.notes.append(note)
                tmp = self.noteSort(tmp, 'Violin')
                if self.judgeMono(tmp, tempo):
                    for note in instrument.notes:
                        Melody.notes.append(note)
                else:
                    for note in instrument.notes:
                        Accompany.notes.append(note)
            else:
                for note in instrument.notes:
                    Accompany.notes.append(note)
        midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        midiReGen.instruments.append(Melody)
        midiReGen.instruments.append(Accompany)
        #midiReGen.write('multi2doubleTrackTest.mid')
        return midiReGen
    
    def tempoNormalize(self, midi_data):
        time_stamps, tempi = midi_data.get_tempo_changes()
        #print(time_stamps, tempi)
        time_stamps = list(time_stamps)
        time_stamps.append(10000)
        tempi = list(tempi)
        #print(time_stamps, tempi)
        midiReGen = pyd.PrettyMIDI(initial_tempo=int(round(np.mean(tempi))))
        for instrument in midi_data.instruments:
            if not self.judgeSequence(instrument):
                instrument = self.noteSort(instrument, 'Acoustic Grand Piano')
            insReGen = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
            
            initTime = 0
            idxT = 0
            firstStart = instrument.notes[0].start
            for idx in range(len(time_stamps)):
                if firstStart >= time_stamps[idx+1]:
                    ratio = tempi[idxT] / int(round(np.mean(tempi)))
                    initTime += (time_stamps[idxT+1] - time_stamps[idxT]) * ratio
                    idxT += 1
                else:
                    break

            for note in instrument.notes:
                pitch = note.pitch
                startOri = note.start
                endOri = note.end
                #print(startOri)
                if not (startOri >= time_stamps[idxT] and startOri < time_stamps[idxT+1]):
                    initTime += (time_stamps[idxT+1] - time_stamps[idxT]) *  ratio
                    idxT += 1
                    #print(idxT)
                ratio = tempi[idxT] / int(round(np.mean(tempi)))
                new_start = initTime + ratio * (startOri - time_stamps[idxT])
                new_end = initTime + ratio * (endOri - time_stamps[idxT])
                noteRecon = pyd.Note(velocity=100, pitch=pitch, start=new_start, end=new_end)
                insReGen.notes.append(noteRecon)
            midiReGen.instruments.append(insReGen)
        #midiReGen.write('tempoNormalTest.mid')
        return midiReGen
    
def tempoNormalize_and_trackDoublize():
    midi_root = 'D:/Download/Program/Musicalion/solo+piano/data'
    save_root = 'D:/Download/Program/Musicalion/solo+piano/data_temoNormalized_trackDoublized'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    numBadMidi = 0
    numSingleTrack = 0
    numMessyMidi = 0
    numSuccessful = 0
    processor = midi_processor()
    for mid in tqdm(os.listdir(midi_root)):
        try:
            midi_data = pyd.PrettyMIDI(os.path.join(midi_root, mid))
        except:
            numBadMidi += 1
            continue
        if len(midi_data.instruments) <= 1:
            numSingleTrack += 1
            continue
        if len(midi_data.get_tempo_changes()[-1]) > 1:
            normalized = processor.tempoNormalize(midi_data)
            doublized = processor.multi2doubleTrack(normalized)
        else:
            doublized = processor.multi2doubleTrack(midi_data)
        if len(doublized.instruments[0].notes) == 0:
            numMessyMidi += 1
            continue
        doublized.write(os.path.join(save_root, mid))
        numSuccessful += 1
    print('succesful:', numSuccessful, 'can\'t load:', numBadMidi, 'single track:', numSingleTrack, 'messy track:', numMessyMidi)


if __name__ == '__main__':
    tempoNormalize_and_trackDoublize()