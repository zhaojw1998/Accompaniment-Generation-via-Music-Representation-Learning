import pretty_midi as pyd
import numpy as np
import os
from tqdm import tqdm

root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_midi'
save = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_Dual-track-Direct'
if not os.path.exists(save):
    os.mkdir(save)
for item in tqdm(os.listdir(root)):
    #print(item)
    piece = pyd.PrettyMIDI(os.path.join(root, item))
    if not (len(piece.instruments)) == 3:
        continue
    downsets = [note.start for note in piece.instruments[1].notes]
    downsets.append(piece.instruments[-1].notes[-1].end)
    chordTrack = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    i = 0
    for note in piece.instruments[-1].notes:
        while downsets[i+1] < note.start:
            i += 1
        #if note.start >= downsets[i] and note.start < downsets[i+1]:
        chordTrack.notes.append(pyd.Note(velocity=100, pitch=note.pitch, start=downsets[i], end=downsets[i+1]))
    new_chordTrack = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    
    start = chordTrack.notes[-1].start
    end = chordTrack.notes[-1].end
    for note in chordTrack.notes[::-1]:
        if note.end <=start:
            end = start
            start = note.start
        new_chordTrack.notes.append(pyd.Note(velocity=100, pitch=note.pitch, start=start, end=end))

    recon = pyd.PrettyMIDI(initial_tempo=120)
    recon.instruments.append(piece.instruments[0])
    recon.instruments.append(new_chordTrack)
    recon.key_signature_changes = piece.key_signature_changes
    recon.time_signature_changes = piece.time_signature_changes
    recon.write(os.path.join(save, item))