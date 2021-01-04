import os
import pretty_midi as pyd 

midi_root = '...'
save_root = '.....'
#对文件夹下的每个midi文件做处理
for midi_name in os.listdir(midi_root):
    #读取多轨midi
    midi_data = pyd.PrettyMIDI(os.path.join(midi_root, midi_name))
    #重建一个midi文件
    midi_ReGen = pyd.PrettyMIDI(initial_tempo=120)
    #为重建的midi文件设置一个单一的track
    instrument_ReGen = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    #遍历多轨midi的每个track
    for instrument in midi_data.instruments:
        #遍历每个track的每个note
        for note in instrument.notes:
            #把该note加到重建的单一track中
            instrument_ReGen.notes.append(note)
    #把重建的单一track加入到重建的midi文件
    midi_ReGen.instruments.append(instrument_ReGen)
    #保存新的midi文件
    midi_ReGen.write(os.path.join(save_root, midi_name))
