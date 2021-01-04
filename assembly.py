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

def assembly_chordfunc(root='./scrape_musescore/data_to_be_used', phraseLength=8):
    assembly = np.empty((0, 16*phraseLength))
    num_messy = 0
    phrase_root = os.path.join(root, str(phraseLength))
    for item in tqdm(os.listdir(phrase_root)):
        file_root = os.path.join(phrase_root, item)
        chordfunc = np.load(os.path.join(file_root, 'chord_func_vector.mid.npy'))
        if not chordfunc.shape == (128,):
            chordfunc = np.zeros(16*phraseLength)
            num_messy += 1
        assembly = np.concatenate((assembly, chordfunc[np.newaxis, :]), axis=0)
    np.save(os.path.join(root, 'chordfuncs_'+str(phraseLength)), assembly)
    print('num_messy:', num_messy)

def assembly_zr(root='./scrape_musescore/data_to_be_used', phraseLength=8):
    model = VAE(roll_dims=130, hidden_dims=1024, rhythm_dims=3, condition_dims=12, z1_dims=128, z2_dims=128, n_step=32).cuda()
    model.load_state_dict(torch.load('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt')['model_state_dict'])
    assembly = torch.empty(0, 128*(phraseLength-1))
    num_messy = 0
    phrase_root = os.path.join(root, str(phraseLength))
    for item in tqdm(os.listdir(phrase_root)):
        file_root = os.path.join(phrase_root, item)
        batched_piano_roll = np.load(os.path.join(file_root, 'batched_piano_roll.mid.npy'))
        if batched_piano_roll.shape[0] == phraseLength-1:
            batched_piano_roll = torch.from_numpy(batched_piano_roll).float().cuda()
            recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(batched_piano_roll[:, :, :130], batched_piano_roll[:, :, 130:])
            zr = dis2m.detach().cpu().view(1, -1)
        else:
            zr = torch.zeros(1, 128*(phraseLength-1))
            num_messy += 1
        assembly = torch.cat((assembly, zr), dim=0)
    assembly = assembly.numpy()
    #print(assembly.shape)
    np.save(os.path.join(root, 'zr_'+str(phraseLength)), assembly)
    print('num_messy:', num_messy)

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

def assembly_piano_roll(root='./scrape_musescore/data_to_be_used', phraseLength=[2, 3, 4, 5, 6, 7, 8]):
    assembly = np.empty((0, 32, 142))
    num_messy = 0
    for length in phraseLength:
        phrase_root = os.path.join(root, str(length))
        for item in tqdm(os.listdir(phrase_root)):
            file_root = os.path.join(phrase_root, item)
            batched_piano_roll = np.load(os.path.join(file_root, 'batched_piano_roll.mid.npy'))
            #num_batch = batched_piano_roll.shape[0]
            if batched_piano_roll.shape[0] == length-1:
                assembly = np.concatenate((assembly, batched_piano_roll), axis=0)
            else:
                num_messy += 1
    assembly_shifted = np.zeros((0, 32, 142))
    for i in tqdm(range(-6, 6, 1)):
        assembly_shifted = np.concatenate((assembly_shifted, tone_shift(assembly, i)), axis=0)
    np.save(os.path.join(root, 'piano_roll_all'), assembly_shifted)
    print(assembly_shifted.shape, 'num_messy:', num_messy)

class piano_roll_dataset(Dataset):
    def __init__(self, diric='./scrape_musescore/data_to_be_used/piano_roll_all.npy'):
        super(piano_roll_dataset, self).__init__()
        self.batched_piano = np.load(diric)
        print(self.batched_piano.shape)

    def __getitem__(self, idx):
        return self.batched_piano[idx]
        
    def __getlen__(self):
        return self.batched_piano.shape[0]

def assembly_zp(root='./scrape_musescore/data_to_be_used'):
    model = VAE(roll_dims=130, hidden_dims=1024, rhythm_dims=3, condition_dims=12, z1_dims=128, z2_dims=128, n_step=32).cuda()
    model.load_state_dict(torch.load('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt')['model_state_dict'])
    dataloader = DataLoader(piano_roll_dataset(), batch_size=256, num_workers=4, shuffle=False, drop_last=False)
    zp = torch.empty(0, 128)
    for batch in dataloader:
        batch = batch.float().cuda()
        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(batch[:, :, :130], batch[:, :, 130:])
        zp_ = dis1m.detach().cpu()
        zp = torch.cat((zp, zp_), dim=1)
    print(zp.shape)
    np.save(os.path.join(root, 'zp_all'), zp)


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

def get_queryPhrase(start_downbeat, end_downbeat, song_name='2nd part for B music'):
    #startdownbeat -content- middownbeat -content- middownbeat -content- end_downbeat
    #input should be a double-track midi
    texture_root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat'
    chord_root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat_withChord'
    songData = songDataProcessor(texture_root, chord_root, song_name)
    #get content
    pianoRoll, chordSeq, TIV = songData.melody2pianoRoll()
    toneKey = songData.getToneKey()
    chordfunc = chord_function_progression(chordSeq, toneKey)
    #get phrtase
    pianoRoll_phrase = songData.numpySplit(pianoRoll, start_downbeat, end_downbeat, 32, 16, 142)
    TIV_phrase = songData.numpySplit(TIV, start_downbeat, end_downbeat, 32, 16, 12)
    chordfunc_phrase = np.array(chordfunc[start_downbeat*16: end_downbeat*16])

    _, chord = songData.texture2prMatrix()
    chord_phrase = songData.chordSplit(chord, start_downbeat, end_downbeat, 8, 4)


    return pianoRoll_phrase, TIV_phrase, chordfunc_phrase, chord_phrase

def cosine(query, instance_space, num_candidate = 10):
    #instance_space: batch * vectorLength
    if len(query.shape) == 2:
        query = query[:, 0]
    result = np.dot(instance_space, query)/(np.linalg.norm(instance_space, axis=1) * np.linalg.norm(query) + 1e-10)
    candidates = result.argsort()[::-1][-6*num_candidate:-5*num_candidate]
    scores = result[candidates]
    names = [os.listdir('./scrape_musescore/data_to_be_used/8')[i] for i in candidates]
    #sort by edit distance over melody
    #candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
    return candidates, scores, names, query[::4], instance_space[candidates][:, ::4]

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
    
def search_for_leadSheet(num_candidate=0):
    #init midi render
    midi_render = midiRender()
    #load corpus
    instance_root = './scrape_musescore/data_to_be_used/8'
    chordfunc_instanceSpace = np.load('./scrape_musescore/data_to_be_used/chordfuncs_8.npy')
    #get query
    #pianoRoll_phrase, TIV_phrase, chordfunc_phrase, gt_chord_phrase = get_queryPhrase(0, 8)#, song_name='Boggy Brays')
    pianoRoll_phrase, TIV_phrase, chordfunc_phrase, gt_chord_phrase = get_queryPhrase(0, 8, song_name='Boggy Brays')
    query_midi = midi_render.leadSheet_recon(pianoRoll_phrase[::2].reshape((-1, 142)), tempo=120, start_time=0)
    #search for acc candidates
    candidates, results, names, query, matches = cosine(chordfunc_phrase, chordfunc_instanceSpace)
    print(results)
    print(names)
    #print(query)
    #print(matches)
    retrive_dir = os.path.join(instance_root, os.listdir(instance_root)[candidates[num_candidate]])
    pr_matrix = np.load(os.path.join(retrive_dir, 'batched_pr_matrix.mid.npy'))
    #retrieved acc
    acc_original = midi_render.pr_matrix2note(pr_matrix, 120)
    #poly-disentangle
    model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    checkpoint = torch.load('D:/Computer Music Research/produce_polyphonic-chord-texture-disentanglement/data/model_master_final.pt')
    model.load_state_dict(checkpoint)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    pr_matrix = torch.from_numpy(pr_matrix).float().cuda()
    gt_chord_phrase = torch.from_numpy(gt_chord_phrase).float().cuda()
    est_x = model.inference(pr_matrix, gt_chord_phrase, sample=False)
    #Generate accompaniment
    midiReGen = midi_render.accomapnimentGeneration(pianoRoll_phrase, est_x, 120)
    midiReGen.write('accompaniment_test_1.mid')
    acc_original.write('accompaniment_origional_1.mid')
    query_midi.write('query_leadsheet.mid')


if __name__ == '__main__':
    #assembly_chordfunc()
    #song = pyd.PrettyMIDI('D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_midi/2nd part for B music.mid')
    #print(song.get_downbeats())
    #root = 'D:/Computer Music Research/produce_Deep-Music-Analogy-Demos/nottingham_database/nottingham_dual-track_doublebeat_withChord'
    #for item in os.listdir(root):
    #    song = pyd.PrettyMIDI(os.path.join(root, item))
    #    print(song.key_signature_changes)
    search_for_leadSheet(num_candidate=0)
    #assembly_zp()

    


    
