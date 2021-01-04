import xmltodict
import dicttoxml
import xml.dom.minidom
import collections
import os
from tqdm import tqdm
import sys
import pretty_midi as pyd
import math
from jingwei_midi_interface_mono_and_chord import midi_interface_mono_and_chord
from jingwei_midi_interface_polyphony import midi_interface_polyphony
from jingwei_dataProcessor import songDataProcessor
import numpy as np 

def phrase_split_by_lyric_punctuation(xml_file='D:/Download/Program/xml/musicalion60517-1 (2).xml'):
    file_object = open(xml_file, encoding = 'utf-8')                                                                                                            
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
    #xml To dict 
    convertedDict = xmltodict.parse(all_the_xmlStr)
    #print(convertedDict['score-partwise']['part'][0]['measure'][4]['note'])
    #try:
    measures_to_split = []
    rest_measure_record = []
    print(len(convertedDict['score-partwise']['part']))
    for i in range(len(convertedDict['score-partwise']['part'])):
        #part_rest_measure_record = []
        part = convertedDict['score-partwise']['part'][i]
        #try:
        for j in range(len(part['measure'])):
            measure = part['measure'][j]
            total_time = 0
            rest_time = 0
            if type(measure['note']) == list:
                for note in measure['note']:
                    #print(note)
                    if 'lyric' in note:
                        text = note['lyric']['text']
                        if type(text) == collections.OrderedDict:
                            text = text['#text']
                        #print(text)
                        if (',' in text) or (';' in text) or ('.' in text) or (':' in text) or ('!' in text):
                            measures_to_split.append(j)
                            break
                    total_time += int(note['duration'])
                    #print(int(note['duration']))
                    if 'rest' in note:
                        rest_time += int(note['duration'])
            if total_time != 0 and rest_time / total_time > 0.5 and 'rest' in measure['note'][0]:
                # we consider measures with a lot of rest, while this measure is should not be the end of phrase
                rest_measure_record.append(j)

            else:
                note = measure['note']
                if 'rest' in note:
                    rest_measure_record.append(j)
                    #print(note)
                    if 'lyric' in note:
                        text = note['lyric']['text']
                        if type(text) == collections.OrderedDict:
                                text = text['#text']
                        #print(text)
                        if (',' in text) or (';' in text) or ('.' in text) or (':' in text) or ('!' in text):
                            measures_to_split.append(j)
        #print(part_rest_measure_record)
        #if i == 0:
        #    rest_measure_record = part_rest_measure_record
        #else:
        #    rest_measure_record = list(set(rest_measure_record) & set(part_rest_measure_record))
    print(sorted(list(set(rest_measure_record))))
    print(sorted(list(set(measures_to_split))))

def select_xml_candidates():
    part_list = []
    melody_track_list = []
    xml_root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_xml'
    midi_root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi'
    for item in tqdm(os.listdir(xml_root)):
        xml_dir = os.path.join(xml_root, item)
        file_object = open(xml_dir, encoding = 'utf-8')                                                                                                            
        try:
            all_the_xmlStr = file_object.read()
        finally:
            file_object.close()
        #xml To dict 
        convertedDict = xmltodict.parse(all_the_xmlStr)
        #num_part_in_real = len(convertedDict['score-partwise']['part'])
        if type(convertedDict['score-partwise']['part-list']['score-part']) == collections.OrderedDict:
            continue
        elif len(convertedDict['score-partwise']['part-list']['score-part']) > 5:
            continue 
        else:
            part_config = [part['part-name']['#text'] if type(part['part-name'])==collections.OrderedDict else part['part-name']  for part in convertedDict['score-partwise']['part-list']['score-part']]
            if 'Piano' in part_config or 'Pianoforte' in part_config:
                # One piano track may take two midi channels.
                numChannel_by_xmlParsing =  len(part_config)
                numChannel_by_prettyMidi = len(pyd.PrettyMIDI(os.path.join(midi_root, item.split('.')[0]+'.mid')).instruments)
                if len(part_config) == 2:
                    #piano + voice
                    idx_melodyTrack = -1
                    offset = 0
                    for idx, part in enumerate(convertedDict['score-partwise']['part']):
                        if part_config[idx] == 'Piano' or part_config[idx] == 'Pianoforte':
                            if numChannel_by_xmlParsing + offset < numChannel_by_prettyMidi:
                                offset += 1
                        else:
                            idx_melodyTrack = idx + offset
                            break
                    if idx_melodyTrack == -1:
                        continue
                elif 'Soprano' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Soprano'):
                            idx_melodyTrack = part_config.index('Soprano') + 1
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Soprano'):
                            idx_melodyTrack = part_config.index('Soprano') + 1
                        else:
                            idx_melodyTrack = part_config.index('Soprano')
                    else:
                        idx_melodyTrack = part_config.index('Soprano')
                elif 'Alto' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Alto'):
                            idx_melodyTrack = part_config.index('Alto') + 1
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Alto'):
                            idx_melodyTrack = part_config.index('Alto') + 1
                        else:
                            idx_melodyTrack = part_config.index('Alto')
                    else:
                        idx_melodyTrack = part_config.index('Alto')
                elif 'Tenor' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Tenor'):
                            idx_melodyTrack = part_config.index('Tenor') + 1
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Tenor'):
                            idx_melodyTrack = part_config.index('Tenor') + 1
                        else:
                            idx_melodyTrack = part_config.index('Tenor')
                    else:
                        idx_melodyTrack = part_config.index('Tenor')
                elif 'Baritone' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Baritone'):
                            idx_melodyTrack = part_config.index('Baritone') + 1
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Baritone'):
                            idx_melodyTrack = part_config.index('Baritone') + 1
                        else:
                            idx_melodyTrack = part_config.index('Baritone')
                    else:
                        idx_melodyTrack = part_config.index('Baritone')
                elif 'Bass' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Bass'):
                            idx_melodyTrack = part_config.index('Bass') + 1
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Bass'):
                            idx_melodyTrack = part_config.index('Bass') + 1
                        else:
                            idx_melodyTrack = part_config.index('Bass')
                    else:
                        idx_melodyTrack = part_config.index('Bass')
                else:
                    offset = 0
                    for idx, part in enumerate(convertedDict['score-partwise']['part']):
                        if part_config[idx] == 'Piano' or part_config[idx] == 'Pianoforte':
                            if numChannel_by_xmlParsing + offset < numChannel_by_prettyMidi:
                                offset += 1
                        if (part_config[idx] != 'Piano') and (part_config[idx] != 'Pianoforte'):
                            idx_melodyTrack = idx + offset
                            break
                melody_track_list.append(item.split('.')[0]+':'+str(idx_melodyTrack)+'\n')
                        
                """for idx, part in enumerate(convertedDict['score-partwise']['part']):
                    if part_config[idx] == 'Piano' or part_config[idx] == 'Pianoforte':
                        #ignore the piano track
                        continue
                    #print(part.keys())
                    part_xml = dicttoxml.dicttoxml(part)
                    dom = xml.dom.minidom.parseString(part_xml)
                    document = dom.documentElement
                    eleList = document.getElementsByTagName('lyric')
                    if len(eleList) > 0:
                        part_list.append(item.split('.')[0]+': '+str(part_config)+'\n')
                        break
                """


    with open('melody_track.txt', 'w', encoding='utf-8') as f:
        f.writelines(melody_track_list) 
    #return num_part

def judgeLyric(part):
    part_xml = dicttoxml.dicttoxml(part)
    dom = xml.dom.minidom.parseString(part_xml)
    document = dom.documentElement
    eleList = document.getElementsByTagName('lyric')
    if len(eleList) > 0:
        return True
    else:
        return False

def getLyric(file_index, part):
    lyric =''
    for j in range(len(part['measure'])):
        measure = part['measure'][j]
        if type(measure['note']) == list:
            for note in measure['note']:
                #print(note)
                if 'lyric' in note:
                    if not type(note['lyric']) == list:
                        lyric_list = [note['lyric']]
                    else:
                        lyric_list = note['lyric']
                    for lyric_ele in lyric_list:    
                        text = lyric_ele['text']
                        if type(text) == None:
                            continue
                        if type(text) == collections.OrderedDict:
                            try:
                                text = text['#text']
                            except KeyError:
                                continue
                        #if file_index == '0036':
                        #    print(text)
                        if type(text) == list:
                            tmp = ''
                            for element in text:
                                if type(element) == str:
                                    tmp = tmp + element
                                else:
                                    if ('#text' in element) and (type(element['#text']) == str):
                                        tmp = tmp + element['#text']
                            text = tmp
                        if text == None:
                            continue
                        lyric = lyric + text

        else:
            note = measure['note']
            if 'lyric' in note:
                if not type(note['lyric']) == list:
                    lyric_list = [note['lyric']]
                else:
                    lyric_list = note['lyric']
                for lyric_ele in lyric_list:    
                    text = lyric_ele['text']
                    if type(text) == collections.OrderedDict:
                        try:
                            text = text['#text']
                        except KeyError:
                            continue
                    #if file_index == '0036':
                    #        print(text)
                    if type(text) == list:
                        tmp = ''
                        for element in text:
                            if type(element) == str:
                                tmp = tmp + element
                            else:
                                try:
                                    tmp = tmp + element['#text']
                                except:
                                    continue
                        text = tmp
                    lyric = lyric + text
    return lyric

def split_by_lyrics(file_index, part):
    bar_with_punctuation = []
    num_bars = len(part['measure'])
    for j in range(len(part['measure'])):
        measure = part['measure'][j]
        lyric_in_bar = ''
        if type(measure['note']) == list:
            for note in measure['note']:
                if 'lyric' in note:
                    if not type(note['lyric']) == list:
                        lyric_list = [note['lyric']]
                    else:
                        lyric_list = note['lyric']
                    for lyric_ele in lyric_list:    
                        text = lyric_ele['text']
                        if type(text) == collections.OrderedDict:
                            try:
                                text = text['#text']
                            except KeyError:
                                continue
                        if type(text) == list:
                            tmp = ''
                            for element in text:
                                if type(element) == str:
                                    tmp = tmp + element
                                else:
                                    if ('#text' in element) and (type(element['#text']) == str):
                                        tmp = tmp + element['#text']
                            text = tmp
                        if type(text) == str:
                            lyric_in_bar = lyric_in_bar + text
        else:
            note = measure['note']
            if 'lyric' in note:
                if not type(note['lyric']) == list:
                    lyric_list = [note['lyric']]
                else:
                    lyric_list = note['lyric']
                for lyric_ele in lyric_list:    
                    text = lyric_ele['text']
                    if type(text) == collections.OrderedDict:
                        try:
                            text = text['#text']
                        except KeyError:
                            continue
                    if type(text) == list:
                        tmp = ''
                        for element in text:
                            if type(element) == str:
                                tmp = tmp + element
                            else:
                                if ('#text' in element) and (type(element['#text']) == str):
                                    tmp = tmp + element['#text']
                        text = tmp
                    if type(text) == str:
                        lyric_in_bar = lyric_in_bar + text
        
        if (',' in lyric_in_bar) or ('，' in lyric_in_bar):
            bar_with_punctuation.append(j)
        elif ('!' in lyric_in_bar) or ('！' in lyric_in_bar):
            bar_with_punctuation.append(j)
        elif ('?' in lyric_in_bar) or ('？' in lyric_in_bar):
            bar_with_punctuation.append(j)
        elif (';' in lyric_in_bar) or ('；' in lyric_in_bar):
            bar_with_punctuation.append(j)
        elif ('.' in lyric_in_bar) or ('。' in lyric_in_bar):
            bar_with_punctuation.append(j)
        else:
            pass
    return bar_with_punctuation
    
def find_rest_bars(file_index, part):
    rest = []
    num_bars = len(part['measure'])
    for j in range(len(part['measure'])):
        measure = part['measure'][j]
        if type(measure['note']) == collections.OrderedDict:
            note = measure['note']
            if 'rest' in note:
                #print(j, 'True')
                rest.append(j)
    return rest, num_bars

def post_split_and_merge(file_index, total_bars, punctuation_anchor, rest_bars):
    num_bars = total_bars
    bar_list = list(range(num_bars))
    #step 1, split the whole score by selecting continuous melody periods
    bar_list.append(100000)
    for rest_bar in rest_bars:
        bar_list.remove(rest_bar)
    melody_periods = []
    initial_anchor = bar_list[0]
    for i in range(len(bar_list)-1):
        if bar_list[i] + 1 < bar_list[i+1]:
            melody_period = list(range(initial_anchor, bar_list[i]+1))
            initial_anchor = bar_list[i+1]
            melody_periods.append(melody_period)
    #print(melody_periods)

    #step 2, split each melody period with lyrics to get phrases
    MaxLen = 8
    MinLen = 4
    phrases = []
    float_stamp = 0
    for melody_period in melody_periods:
        phrases_in_melody = []
        definite_stamp = 0
        while (float_stamp < len(punctuation_anchor)) and (punctuation_anchor[float_stamp] <= melody_period[0]):
            float_stamp += 1
        while (float_stamp < len(punctuation_anchor)) and (punctuation_anchor[float_stamp] <= melody_period[-1]):
            length = punctuation_anchor[float_stamp]-melody_period[definite_stamp]
            phrase_candidate = melody_period[definite_stamp: definite_stamp+length]
            if length >= MinLen and length <= MaxLen:
                if len(phrases_in_melody) == 0:
                    phrases_in_melody.append(phrase_candidate)
                    definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                    float_stamp += 1
                else:
                    last = phrases_in_melody.pop()
                    #print(last)
                    #print('*********', len(last), length)
                    if len(last) < MinLen and len(last) + length <= MaxLen:
                        phrase_candidate = last + phrase_candidate
                        phrases_in_melody.append(phrase_candidate)
                        definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                        float_stamp += 1
                    else:
                        phrases_in_melody.append(last)
                        phrases_in_melody.append(phrase_candidate)
                        definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                        float_stamp += 1
            elif length > MaxLen:
                for i in range(math.ceil(length/MaxLen)):
                    phrases_in_melody.append(phrase_candidate[i*MaxLen: min((i+1)*MaxLen, length)])
                definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                float_stamp += 1
            elif length < MinLen:
                if len(phrases_in_melody) == 0:
                    phrases_in_melody.append(phrase_candidate)
                    definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                    float_stamp += 1
                else:
                    last = phrases_in_melody.pop()
                    if len(last) + length <= MaxLen:
                        phrase_candidate = last + phrase_candidate
                        phrases_in_melody.append(phrase_candidate)
                        definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                        float_stamp += 1
                    else:
                        phrases_in_melody.append(last)
                        if (float_stamp+1 < len(punctuation_anchor)) and (punctuation_anchor[float_stamp+1] <= melody_period[-1]):
                            next_length = punctuation_anchor[float_stamp+1] - punctuation_anchor[float_stamp]
                            next_ = melody_period[definite_stamp+length: definite_stamp+length+next_length]
                            if length + next_length <= MaxLen:
                                phrase_candidate = phrase_candidate + next_
                                phrases_in_melody.append(phrase_candidate)
                                definite_stamp = melody_period.index(punctuation_anchor[float_stamp+1])
                                float_stamp += 2
                            else:
                                phrases_in_melody.append(phrase_candidate)
                                definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                                float_stamp += 1
                        else:
                            phrases_in_melody.append(phrase_candidate)
                            definite_stamp = melody_period.index(punctuation_anchor[float_stamp])
                            float_stamp += 1
        if definite_stamp < len(melody_period):
            phrase_candidate = melody_period[definite_stamp: len(melody_period)]
            length = len(phrase_candidate)
            if length >= MinLen and length <= MaxLen:
                if len(phrases_in_melody) == 0:
                    phrases_in_melody.append(phrase_candidate)
                else:
                    last = phrases_in_melody.pop()
                    if len(last) < MinLen and len(last) + length <= MaxLen:
                        phrase_candidate = last + phrase_candidate
                        phrases_in_melody.append(phrase_candidate)
                    else:
                        phrases_in_melody.append(last)
                        phrases_in_melody.append(phrase_candidate)
            elif length > MaxLen:
                for i in range(math.ceil(length/MaxLen)):
                    phrases_in_melody.append(phrase_candidate[i*MaxLen: min((i+1)*MaxLen, length)])
            elif length < MinLen:
                if len(phrases_in_melody) == 0:
                    phrases_in_melody.append(phrase_candidate)
                else:
                    last = phrases_in_melody.pop()
                    if len(last) + length <= MaxLen:
                        phrase_candidate = last + phrase_candidate
                        phrases_in_melody.append(phrase_candidate)
                    else:
                        phrases_in_melody.append(last)
                        phrases_in_melody.append(phrase_candidate)   
        phrases = phrases + phrases_in_melody
    return phrases

def splitMIDI(midi_data, start_time, end_time):
    tempo = midi_data.get_tempo_changes()[-1][0]
    ts = midi_data.time_signature_changes
    midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
    for instrument in midi_data.instruments:
        new_instrument = pyd.Instrument(program=instrument.program)
        for note in instrument.notes:
            if note.start >= start_time and note.end <= end_time:
                new_start = note.start - start_time
                new_end = note.end - start_time
                new_note = pyd.Note(velocity=100, pitch=note.pitch, start=new_start, end=new_end)
                new_instrument.notes.append(new_note)
        midiReGen.instruments.append(new_instrument)
    return midiReGen

class dataRender(midi_interface_mono_and_chord):
    def __init__(self):
        super(dataRender, self).__init__()
    def getData(self, midi_data):
        melodySequence = self.getMelodySeq_byBeats(midi_data)
        chordSequence = self.getChordSeq_byBeats(midi_data)
        #print(melodySequence)
        matrix = self.seq2Numpy(melodySequence, chordSequence)
        return matrix, chordSequence

def chord_function_progression(chordsequence, key_number):
    reference = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    progression = []
    key_number = key_number % 12
    reference = reference[key_number:] + reference[:key_number]
    for item in chordsequence:
        item = item.strip('m')
        if item == 'NC':
            progression.append(-1)
        else:
            progression.append(reference.index(item))
    return progression

def phrase_split():
    #whole_lyric = []
    save_root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/data_to_be_used'
    melody_dict = {}
    with open('./melody_track.txt', 'r') as f:
        readlines = f.readlines()
    for item in readlines:
        melody_dict[item.split(':')[0]] = int(item.strip('\n').split(':')[-1].split(';')[-1])
    xml_root = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musescore\\musescore_xml'
    processed_midi_root_withChord = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musescore\\musescore_midi_processed_chord'
    processed_midi_root_withTexture = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musescore\\musescore_midi_processed'

    for item in tqdm(os.listdir(processed_midi_root_withChord)):
        file_index = item.split('.')[0]
        #file_root = os.path.join(save_root, file_index)
        #midi_file = os.path.join(processed_midi_root, item)
        xml_file = os.path.join(xml_root, file_index+'.xml')
        #read xml file
        file_object = open(xml_file, encoding = 'utf-8')                                                                                                            
        try:
            all_the_xmlStr = file_object.read()
        finally:
            file_object.close()
        #xml To dict 
        convertedDict = xmltodict.parse(all_the_xmlStr)
        part = convertedDict['score-partwise']['part'][melody_dict[file_index]]
        rest_bars, num_bars = find_rest_bars(file_index, part)
        if judgeLyric(part):
            punctuation_anchor= split_by_lyrics(file_index, part)
        else:
            punctuation_anchor = []
        phrases = post_split_and_merge(file_index, num_bars, punctuation_anchor, rest_bars)
        #phrases.append([phrases[-1][-1]+1])
        #print(phrases)
        processor = songDataProcessor(texture_root=processed_midi_root_withTexture,
                                        chord_root=processed_midi_root_withChord, 
                                        song_idx_str=file_index)
        time_downbeats = processor.getDownbeats()
        tone_key = processor.getToneKey()

        piano_roll_withChord, chordSeqence, TIV = processor.melody2pianoRoll()
        pr_matrix, pr_chord = processor.texture2prMatrix()
        chord_funcProgression = chord_function_progression(chordSeqence, tone_key)

        for idx_Ph, phrase in enumerate(phrases):
            idx_start_bar = phrase[0]
            idx_end_bar = phrase[-1] + 1
            num_bars = idx_end_bar - idx_start_bar
            if num_bars <= 1:
                continue
            phrase_root = os.path.join(save_root, str(num_bars))
            if not os.path.exists(phrase_root):
                os.mkdir(phrase_root)
            file_root = os.path.join(phrase_root, file_index+'-'+str(idx_Ph))
            if not os.path.exists(file_root):
                os.mkdir(file_root)
            batched_piano_roll = processor.numpySplit(matrix=piano_roll_withChord, 
                                                        start_downbeat=idx_start_bar, 
                                                        end_downbeat=idx_end_bar, 
                                                        WINDOWSIZE=32, 
                                                        HOPSIZE=16, 
                                                        VECTORSIZE=142)
            batched_TIV = processor.numpySplit(matrix=TIV, 
                                                start_downbeat=idx_start_bar, 
                                                end_downbeat=idx_end_bar, 
                                                WINDOWSIZE=32, 
                                                HOPSIZE=16, 
                                                VECTORSIZE=12)

            chord_func_vector = np.array(chord_funcProgression[idx_start_bar*16: idx_end_bar*16])

            batched_pr_matrix = processor.numpySplit(matrix=pr_matrix, 
                                                    start_downbeat=idx_start_bar, 
                                                    end_downbeat=idx_end_bar, 
                                                    WINDOWSIZE=32, 
                                                    HOPSIZE=16, 
                                                    VECTORSIZE=128)
            batched_pr_chord = processor.chordSplit(chord=pr_chord, 
                                                    start_downbeat=idx_start_bar, 
                                                    end_downbeat=idx_end_bar,  
                                                    WINDOWSIZE=8, 
                                                    HOPSIZE=4)
            np.save(os.path.join(file_root, 'batched_piano_roll.mid'), batched_piano_roll)
            np.save(os.path.join(file_root, 'batched_TIV.mid'), batched_TIV)
            np.save(os.path.join(file_root, 'chord_func_vector.mid'), chord_func_vector)
            np.save(os.path.join(file_root, 'batched_pr_matrix.mid'), batched_pr_matrix)
            np.save(os.path.join(file_root, 'batched_pr_chord.mid'), batched_pr_chord)

            print(batched_piano_roll.shape, batched_TIV.shape, chord_func_vector.shape, batched_pr_matrix.shape, batched_pr_chord.shape)


if __name__ == '__main__':
    #phrase_split_by_lyric_punctuation(xml_file='D:/Download/Program/xml/musicalion50709-1 (6).xml')
    #phrase_split_by_lyric_punctuation()
    """you need to discriminate vocal/solo track"""

    #select_xml_candidates()
    phrase_split()
    """test instance: D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi_processed/0025.mid"""
    
    """statistics = 0
    midi_root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi_processed'
    xml_root = 'D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_xml'
    melody_dict = {}
    with open('./melody_track.txt', 'r') as f:
        readlines = f.readlines()
    for item in readlines:
        melody_dict[item.split(':')[0]] = int(item.strip('\n').split(':')[-1].split(';')[-1])

    for item in os.listdir(midi_root):
        file_index = item.split('.')[0]
        midi_data = pyd.PrettyMIDI(os.path.join(midi_root, item))
        #print(midi_data.get_tempo_changes()[0])
        
        #print(midi_data.time_signature_changes)
        #print(midi_data.key_signature_changes)
        #print(midi_data.time_signature_changes[0].time)
        length_1 = midi_data.get_downbeats()
        print(length_1)
        xml_file = os.path.join(xml_root, file_index+'.xml')
        file_object = open(xml_file, encoding = 'utf-8')                                                                                                            
        try:
            all_the_xmlStr = file_object.read()
        finally:
            file_object.close()
        convertedDict = xmltodict.parse(all_the_xmlStr)
        part = convertedDict['score-partwise']['part'][melody_dict[file_index]]
        length_2 = len(part['measure'])
        #if not length_1 == length_2:
        #    #print(file_index, length_1, length_2)
        #    statistics += 1
    print('number of unmatched:', statistics)"""
    """a = pyd.PrettyMIDI('D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi/0146.mid')
    print(a.time_signature_changes)"""