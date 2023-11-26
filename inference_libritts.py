import os.path

import torch
import soundfile
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)


def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)  # [1,14]

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)  # s[1] 14
        text_mask = length_to_mask(input_lengths).to(device)  # [1,14] False

        t_en = model.text_encoder(tokens, input_lengths, text_mask)  # [1,512,14]
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())  # tokens [1,14]    attention_mask [1,14] all 1   bert_dur [1,14,768]
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [1,512,14]

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),  # [1,1,256]
                         embedding=bert_dur,  # [1,14,768]
                         embedding_scale=embedding_scale,  # embedding_scale 1
                         features=ref_s,  # [1,256]  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)  # diffusion_steps = 5  s_pred [1,256]

        s = s_pred[:, 128:]  # s [1,128]
        ref = s_pred[:, :128]  # ref [1,128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]  # alpha = 0.3  ref [1,128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]   # ref = 0.7  s [1, 128]

        d = model.predictor.text_encoder(d_en,  # [1,512,14]
                                         s, input_lengths, text_mask)  # s [1,128] input_lengths=14  text_mask [1,14] all False  d [1,14,640]

        x, _ = model.predictor.lstm(d)  # x [1,14,512]
        duration = model.predictor.duration_proj(x)  # duration [1,14,50]

        duration = torch.sigmoid(duration).sum(axis=-1)  # duration [1,14]
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)  # pred_dur [14]

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))  # int(pred_dur.sum().data)=44  input_lengths = 14
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)  # construct mono align

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))  # [1,14,640] > [1,640,14]  [1,14,44] > en: [1,640,44]
        if model_params.decoder.type == "hifigan":  # yes
            asr_new = torch.zeros_like(en)  # asr_new [1,640,44]
            asr_new[:, :, 0] = en[:, :, 0]  # first frame initialize asr_new first frame
            asr_new[:, :, 1:] = en[:, :, 0:-1]  # 0~-1 frame initialize asr_new 1: frame
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)  # en=asr_new  s [1,128] -> F0_pred [1,88] N_pred [1,88]

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))  # t_en [1,512,14] pred_aln_trg -> [1,14,44] -> [1,512,44]
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)  # asr_new [1,512,44]
            asr_new[:, :, 0] = asr[:, :, 0]  # eg
            asr_new[:, :, 1:] = asr[:, :, 0:-1]  # eg
            asr = asr_new

        out = model.decoder(asr,  # asr [1,512,44]  F0_pred [1, 88]  N_pred [1,88]  ref [1,128]
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))  # out [1,1,26400]

    return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later
reference_dicts = {}
# format: (path, text)
reference_dicts['1221-135767'] = ("Demo/reference_audio/1221-135767-0014.wav", "Hello world")
reference_dicts['5639-40744'] = ("Demo/reference_audio/5639-40744-0020.wav", "Thus did this humane and right minded father comfort his unhappy daughter, and her mother embracing her again, did all she could to soothe her feelings.")
reference_dicts['908-157963'] = ("Demo/reference_audio/908-157963-0027.wav", "And lay me down in my cold bed and leave my shining lot.")
reference_dicts['4077-13754'] = ("Demo/reference_audio/4077-13754-0000.wav", "The army found the people in poverty and left them in comparative wealth.")

noise = torch.randn(1,1,256).to(device)
import shutil
out_dir = "result/base_zero_shot"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for k, v in reference_dicts.items():
    path, text = v
    target_ref_path = os.path.join(out_dir, path.split("/")[-1])
    shutil.copyfile(path,target_ref_path)
    filename = path.split("/")[-1].split(".")[0]
    target_result_path = os.path.join(out_dir, "ref_" + filename + "_result.wav")
    ref_s = compute_style(path)
    start = time.time()
    wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
    soundfile.write(file=target_result_path, data=wav, samplerate=24000)
    rtf = (time.time() - start) / (len(wav) / 24000)
    print(f"RTF = {rtf:5f}")
    print(k + ' Synthesized: ' + text)
    print(k + ' Reference:')


