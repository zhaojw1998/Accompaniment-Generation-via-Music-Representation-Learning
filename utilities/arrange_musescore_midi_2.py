import xmltodict
import dicttoxml
import xml.dom.minidom
import collections
import os
from tqdm import tqdm
import sys
import pretty_midi as pyd
from midi_processor import midi_processor


def select_xml_candidates():
    # select scores for vocal/solo + piano, and discard others
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
                            idx_melodyPart = idx
                            break
                    if idx_melodyTrack == -1:
                        continue
                elif 'Soprano' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Soprano'):
                            idx_melodyTrack = part_config.index('Soprano') + 1
                            idx_melodyPart = part_config.index('Soprano')
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Soprano'):
                            idx_melodyTrack = part_config.index('Soprano') + 1
                            idx_melodyPart = part_config.index('Soprano')
                        else:
                            idx_melodyTrack = part_config.index('Soprano')
                            idx_melodyPart = part_config.index('Soprano')
                    else:
                        idx_melodyTrack = part_config.index('Soprano')
                        idx_melodyPart = part_config.index('Soprano')
                elif 'Alto' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Alto'):
                            idx_melodyTrack = part_config.index('Alto') + 1
                            idx_melodyPart = part_config.index('Alto')
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Alto'):
                            idx_melodyTrack = part_config.index('Alto') + 1
                            idx_melodyPart = part_config.index('Alto')
                        else:
                            idx_melodyTrack = part_config.index('Alto')
                            idx_melodyPart = part_config.index('Alto')
                    else:
                        idx_melodyTrack = part_config.index('Alto')
                        idx_melodyPart = part_config.index('Alto')
                elif 'Tenor' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Tenor'):
                            idx_melodyTrack = part_config.index('Tenor') + 1
                            idx_melodyPart = part_config.index('Tenor')
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Tenor'):
                            idx_melodyTrack = part_config.index('Tenor') + 1
                            idx_melodyPart = part_config.index('Tenor')
                        else:
                            idx_melodyTrack = part_config.index('Tenor')
                            idx_melodyPart = part_config.index('Tenor')
                    else:
                        idx_melodyTrack = part_config.index('Tenor')
                        idx_melodyPart = part_config.index('Tenor')
                elif 'Baritone' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Baritone'):
                            idx_melodyTrack = part_config.index('Baritone') + 1
                            idx_melodyPart = part_config.index('Baritone')
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Baritone'):
                            idx_melodyTrack = part_config.index('Baritone') + 1
                            idx_melodyPart = part_config.index('Baritone')
                        else:
                            idx_melodyTrack = part_config.index('Baritone')
                            idx_melodyPart = part_config.index('Baritone')
                    else:
                        idx_melodyTrack = part_config.index('Baritone')
                        idx_melodyPart = part_config.index('Baritone')
                elif 'Bass' in part_config:
                    if numChannel_by_xmlParsing < numChannel_by_prettyMidi:
                        if 'Piano' in part_config and part_config.index('Piano') < part_config.index('Bass'):
                            idx_melodyTrack = part_config.index('Bass') + 1
                            idx_melodyPart = part_config.index('Bass')
                        elif 'Pianoforte' in part_config and part_config.index('Pianoforte') < part_config.index('Bass'):
                            idx_melodyTrack = part_config.index('Bass') + 1
                            idx_melodyPart = part_config.index('Bass')
                        else:
                            idx_melodyTrack = part_config.index('Bass')
                            idx_melodyPart = part_config.index('Bass')
                    else:
                        idx_melodyTrack = part_config.index('Bass')
                        idx_melodyPart = part_config.index('Bass')
                else:
                    offset = 0
                    for idx, part in enumerate(convertedDict['score-partwise']['part']):
                        if part_config[idx] == 'Piano' or part_config[idx] == 'Pianoforte':
                            if numChannel_by_xmlParsing + offset < numChannel_by_prettyMidi:
                                offset += 1
                        if (part_config[idx] != 'Piano') and (part_config[idx] != 'Pianoforte'):
                            idx_melodyTrack = idx + offset
                            idx_melodyPart = idx
                            break
                melody_track_list.append(item.split('.')[0]+':'+str(idx_melodyTrack)+';'+str(idx_melodyPart)+'\n')
                        
                """#print part configurations for each music score
                for idx, part in enumerate(convertedDict['score-partwise']['part']):
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



class musescore_midi_processor(midi_processor):
    def __init__(self, save_root='D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi_processed'):
        super(musescore_midi_processor, self).__init__()
        self.save_root = save_root
        self.melody_dict = {}
        with open('./melody_track.txt', 'r') as f:
            readlines = f.readlines()
        for item in readlines:
            self.melody_dict[item.split(':')[0]] = int(item.strip('\n').split(':')[-1].split(';')[0])

    def multi2doubleTrack(self, midi_data, idx_melodyTrack):
        tempi = midi_data.get_tempo_changes()[-1]
        ts = midi_data.time_signature_changes
        ks = midi_data.key_signature_changes
        assert(len(tempi) == 1)
        tempo = tempi[0]
        Melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
        Accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        for idx, instrument in enumerate(midi_data.instruments):
            if idx == idx_melodyTrack:
                for note in instrument.notes:
                    Melody.notes.append(note)             
            else:
                for note in instrument.notes:
                    Accompany.notes.append(note)
        midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
        midiReGen.instruments.append(Melody)
        midiReGen.instruments.append(Accompany)
        ts = [pyd.TimeSignature(numerator=4, denominator=4, time=0.0)]
        midiReGen.time_signature_changes = ts
        midiReGen.key_signature_changes = ks
        #midiReGen.write('multi2doubleTrackTest.mid')
        return midiReGen

    def judgeTS(self, ts):
        for ts_item in ts:
            nume = ts_item.numerator
            deno = ts_item.denominator
            if (nume % 2 == 0 and nume % 3 != 0): # only keep liangpaizi 
                continue
            else:
                return False
        return True
       
    def tempoNormalize_and_trackDoublize(self, root='D:/Computer Music Research/score scrape and analysis/scrape_musescore/musescore_midi'):
        numBadMidi = 0
        numSuccessful = 0
        no_accompaniment = 0
        messy_score = 0
        non_dualbeat = 0
        for name in self.melody_dict:
            try:
                midi_data = pyd.PrettyMIDI(os.path.join(root, name+'.mid'))
            except:
                numBadMidi += 1
                continue
            
            ts = midi_data.time_signature_changes
            if not self.judgeTS(ts):
                non_dualbeat += 1
                continue

            if len(midi_data.get_tempo_changes()[-1]) > 1:
                normalized = self.tempoNormalize(midi_data)
                doublized = self.multi2doubleTrack(normalized, self.melody_dict[name])
            else:
                doublized = self.multi2doubleTrack(midi_data, self.melody_dict[name])
            if len(doublized.instruments) < 2:
                continue
            else:
                if len(doublized.instruments[1].notes) <= 10 or len(doublized.instruments[0].notes) <= 10:
                    no_accompaniment += 1
                    continue
                tempo = doublized.get_tempo_changes()[-1][0]
                if not self.judgeMono(doublized.instruments[0], tempo):
                    messy_score += 1
                    continue
                if self.judgeMono(doublized.instruments[1], tempo):
                    messy_score += 1
                    continue
            doublized.write(os.path.join(self.save_root, name+'.mid'))
            numSuccessful += 1
        print('succesful:', numSuccessful, 'can\'t load:', numBadMidi, 'no accompaniment:', no_accompaniment, 'messy_score:', messy_score, 'non_dualbeat:', non_dualbeat)

    def process_nottingham(self, save_root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat'):
        numBadMidi = 0
        numSuccessful = 0
        no_accompaniment = 0
        messy_score = 0
        non_dualbeat = 0
        root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_midi'
        for mid in os.listdir(root):
            midi_data = pyd.PrettyMIDI(os.path.join(root, mid))
            ts = midi_data.time_signature_changes
            if not self.judgeTS(ts):
                non_dualbeat += 1
                continue
            if len(midi_data.get_tempo_changes()[-1]) > 1:
                normalized = self.tempoNormalize(midi_data)
                doublized = self.multi2doubleTrack(normalized, 0)
            else:
                doublized = self.multi2doubleTrack(midi_data, 0)
            if len(doublized.instruments) < 2:
                continue
            else:
                if len(doublized.instruments[1].notes) <= 10 or len(doublized.instruments[0].notes) <= 10:
                    no_accompaniment += 1
                    continue
                tempo = doublized.get_tempo_changes()[-1][0]
                if not self.judgeMono(doublized.instruments[0], tempo):
                    messy_score += 1
                    continue
                if self.judgeMono(doublized.instruments[1], tempo):
                    messy_score += 1
                    continue
            doublized.write(os.path.join(save_root, mid))
            numSuccessful += 1
            print('succesful:', numSuccessful, 'can\'t load:', numBadMidi, 'no accompaniment:', no_accompaniment, 'messy_score:', messy_score, 'non_dualbeat:', non_dualbeat)


if __name__ == '__main__':
    #first, select candidates which contain only vocal/solo and piano
    #select_xml_candidates()
    #process midi files for the scores and render double-track midis with normalized speed
    processor = musescore_midi_processor()
    processor.process_nottingham()