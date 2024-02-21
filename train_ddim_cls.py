import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time

import torch
import torchaudio
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator
from diffusers import DDIMScheduler

from modules.autoencoder import AutoencoderKL
from modules.mel import LogMelSpectrogram
from model.unet import DiffTSE
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff
from inference import eval_ddim
from dataset import TSEDataset

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--data-path', type=str, default='../data/fsd2018/')
parser.add_argument('--timbre-path', type=str, default=None)
parser.add_argument('--audio-length', type=int, default=4)
parser.add_argument('--use-timbre-feature', type=bool, default=False)

# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='modules/first_stage.pt')
parser.add_argument('--scale-factor', type=float, default=1.0)

# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=2)

# model configs
parser.add_argument('--autoencoder-config', type=str, default='modules/vae.yaml')
parser.add_argument('--diffusion-config', type=str, default='model/DiffTSE_cls_v_l_100.yaml')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# steps for DDIM training
# parser.add_argument("--num-train-steps", type=int, default=1000)
# parser.add_argument("--num-infer-steps", type=int, default=30)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--log-step', type=int, default=100)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


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
args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')

if os.path.exists(args.log_dir + '/pic/gt') is False:
    os.makedirs(args.log_dir + '/pic/gt')

if os.path.exists(args.log_dir + '/audio/gt') is False:
    os.makedirs(args.log_dir + '/audio/gt')

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir)

# n = open(args.log_dir + 'ddim_cls_log.txt', mode='a')
# n.write('diff tse log')
# n.close()

if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    train_set = TSEDataset(meta_dir=args.data_path+'meta.csv', data_dir=args.data_path,
                           subset='train', length=args.audio_length,
                           use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path,
                           mel_length=args.audio_length*100)

    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # val_set = TSEDataset(meta_dir=args.data_path+'meta.csv', data_dir=args.data_path, length=10,
    #                      use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path,
    #                      mel_length=1000)
    # val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.batch_size)
    # test_set = TSEDataset(meta_dir=args.data_path+'meta.csv', data_dir=args.data_path, length=10,
    #                       use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path,
    #                       mel_length=1000)
    # test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=args.batch_size)

    # use this load for check generated audio samples
    eval_set = TSEDataset(meta_dir='eval.csv', data_dir=args.data_path, subset='val', length=10,
                          use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path,
                          mel_length=1000)
    eval_loader = DataLoader(eval_set, num_workers=args.num_workers, batch_size=args.batch_size)
    # use these two loaders for benchmarks

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)

    logmel = LogMelSpectrogram(mel_length=args.audio_length*100).to(accelerator.device)
    logmel_val = LogMelSpectrogram(mel_length=1000).to(accelerator.device)

    autoencoder = AutoencoderKL(**args.vae_config['params'])
    checkpoint = torch.load(args.autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    autoencoder.to(accelerator.device)

    unet = DiffTSE(args.diff_config['diffwrap']).to(accelerator.device)

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # if args.v_prediction:
    #     print('v prediction')
    #     noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps,
    #                                     beta_start=args.beta_start, beta_end=args.beta_end,
    #                                     rescale_betas_zero_snr=True,
    #                                     timestep_spacing="trailing",
    #                                     prediction_type='v_prediction')
    # else:
    #     print('noise prediction')
    #     noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps,
    #                                     beta_start=args.beta_start, beta_end=args.beta_end,
    #                                     prediction_type='epsilon')
    #
    # optimizer = torch.optim.AdamW(unet.parameters(),
    #                               lr=args.learning_rate,
    #                               betas=(args.beta1, args.beta2),
    #                               weight_decay=args.weight_decay,
    #                               eps=args.adam_epsilon,
    #                               )
    # loss_func = torch.nn.MSELoss()
    #
    # # scaler = GradScaler()
    # # put to accelerator
    # unet, autoencoder, logmel, logmel_val, optimizer, train_loader = \
    #     accelerator.prepare(unet, autoencoder, logmel, logmel_val, optimizer, train_loader)
    #
    # global_step = 0
    # losses = 0
    #
    # if accelerator.is_main_process:
    #     eval_ddim(autoencoder, unet, logmel_val, noise_scheduler, eval_loader, args, accelerator.device,
    #               epoch=0, ddim_steps=args.num_infer_steps, eta=1)
    #
    # for epoch in range(args.epochs):
    #     unet.train()
    #     for step, batch in enumerate(tqdm(train_loader)):
    #         # compress by vae
    #         mixture, _, target, _, _, cls, _, _ = batch
    #
    #         # mixture = autoencoder.mel2emb(logmel(mixture))*args.scale_factor
    #         mixture = minmax_norm_diff(logmel(mixture)).unsqueeze(1)
    #         # timbre = logmel(timbre)
    #         cls = cls.long()
    #         # target = autoencoder.mel2emb(logmel(target))*args.scale_factor
    #         target = minmax_norm_diff(logmel(target)).unsqueeze(1)
    #
    #         # adding noise
    #         noise = torch.randn(target.shape).to(accelerator.device)
    #         timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
    #                                   device=accelerator.device,).long()
    #         noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
    #         # v prediction - model output
    #         velocity = noise_scheduler.get_velocity(target, noise, timesteps)
    #
    #         # inference
    #         pred = unet(x=noisy_target, t=timesteps, mixture=mixture, cls=cls,
    #                     timbre=None, timbre_feature=None, event=None)
    #
    #         # backward
    #         if args.v_prediction:
    #             loss = loss_func(pred, velocity)
    #         else:
    #             loss = loss_func(pred, noise)
    #
    #         accelerator.backward(loss)
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #         global_step += 1
    #         losses += loss.item()
    #
    #         if accelerator.is_main_process:
    #             if global_step % args.log_step == 0:
    #                 n = open(args.log_dir + 'ddim_cls_log.txt', mode='a')
    #                 n.write(time.asctime(time.localtime(time.time())))
    #                 n.write('\n')
    #                 n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
    #                     epoch + 1, args.epochs, step+1, len(train_loader), losses / args.log_step))
    #                 n.close()
    #                 losses = 0.0
    #
    #     if accelerator.is_main_process:
    #         eval_ddim(autoencoder, unet, logmel_val, noise_scheduler, eval_loader, args, accelerator.device,
    #                   epoch=epoch+1, ddim_steps=args.num_infer_steps, eta=1)
    #
    #     if (epoch + 1) % args.save_every == 0:
    #         accelerator.wait_for_everyone()
    #         unwrapped_unet = accelerator.unwrap_model(unet)
    #         accelerator.save({
    #             "model": unwrapped_unet.state_dict(),
    #         }, args.save_dir+str(epoch)+'.pt')
