import yaml
import random
import argparse

import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler

from modules.autoencoder import AutoencoderKL
from modules.mel import LogMelSpectrogram
from model.unet import DiffTSE
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mixture', type=str, default='mixture.wav')
parser.add_argument('--target_sound', type=str, default='Applause')


# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='../ckpts/first_stage.pt')
parser.add_argument('--scale-factor', type=float, default=1.0)

# model configs
parser.add_argument('--autoencoder-config', type=str, default='../ckpts/vae.yaml')
parser.add_argument('--diffusion-config', type=str, default='config/DiffTSE_cls_v_b_1000.yaml')
parser.add_argument('--diffusion-ckpt', type=str, default='../ckpts/base_v_1000.pt')

# log and random seed
parser.add_argument('--random-seed', type=int, default=2023)
args = parser.parse_args()


with open(args.autoencoder_config, 'r') as fp:
    args.vae_config = yaml.safe_load(fp)

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)


args.num_train_steps = args.diff_config["ddim"]["num_train"]
args.num_infer_steps = args.diff_config["ddim"]["num_inference"]
args.beta_start = args.diff_config["ddim"]["beta_start"]
args.beta_end = args.diff_config["ddim"]["beta_end"]
args.v_prediction = args.diff_config["ddim"]["v_predictionn"]


@torch.no_grad()
def sample_diffusion(unet, logmel_val, vocoder, scheduler,
                     mixture, cls, device, ddim_steps=50, eta=1, seed=2023):
    unet.eval()

    scheduler.set_timesteps(ddim_steps)
    mixture_mel = minmax_norm_diff(logmel_val(mixture)).unsqueeze(1)
    # timbre = logmel(timbre)
    cls = cls.long()
    generator = torch.Generator(device=device).manual_seed(seed)

    # init noise
    noise = torch.randn(mixture_mel.shape, generator=generator, device=device)
    pred = noise

    for t in tqdm(scheduler.timesteps):
        pred = scheduler.scale_model_input(pred, t)
        model_output = unet(x=pred, t=t, mixture=mixture_mel, cls=cls,
                            timbre=None, timbre_feature=None, event=None)
        pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                              eta=eta, generator=generator).prev_sample

    # pred = autoencoder.emb2mel(pred)
    pred = reverse_minmax_norm_diff(pred)
    pred_wav = vocoder.mel2wav(pred.squeeze(1))
    return pred_wav



if __name__ == '__main__':
    # logmel = LogMelSpectrogram(mel_length=args.audio_length * 100).to(accelerator.device)
    logmel_val = LogMelSpectrogram(mel_length=1000).to(args.device)

    autoencoder = AutoencoderKL(**args.vae_config['params'])
    checkpoint = torch.load(args.autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    autoencoder.to(args.device)

    unet = DiffTSE(args.diff_config['diffwrap']).to(args.device)
    unet.load_state_dict(torch.load(args.diffusion_ckpt)['model'])

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    if args.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps,
                                        beta_start=args.beta_start, beta_end=args.beta_end,
                                        rescale_betas_zero_snr=True,
                                        timestep_spacing="trailing",
                                        prediction_type='v_prediction')
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps,
                                        beta_start=args.beta_start, beta_end=args.beta_end,
                                        prediction_type='epsilon')

    sound_names = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum",
                   "Burping_or_eructation", "Bus", "Cello", "Chime",
                   "Clarinet", "Computer_keyboard", "Cough", "Cowbell",
                   "Double_bass", "Drawer_open_or_close", "Electric_piano",
                   "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel",
                   "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling",
                   "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone",
                   "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing",
                   "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]

    sound_dict = {name: i for i, name in enumerate(sound_names)}

    sr = 16000

    mixture, sr = librosa.load(args.mixture, sr=sr)
    mixture = torch.tensor(mixture).unsqueeze(0).to(args.device)
    cls = torch.tensor(sound_dict[args.target_sound]).unsqueeze(0).to(args.device)

    pred = sample_diffusion(unet, logmel_val, autoencoder, noise_scheduler,
                            mixture, cls, args.device, ddim_steps=50, eta=1, seed=2023)

    save_audio(f'{args.mixture}_pred.wav', sr, pred)

    print(f'the prediction is save to {args.mixture}_pred.wav')