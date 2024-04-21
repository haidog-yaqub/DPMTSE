import torch
import os
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff, get_loss
from tqdm import tqdm


@torch.no_grad()
def eval_ddim(autoencoder, unet, logmel_val, scheduler, eval_loader, args, device, epoch=0, ddim_steps=50, eta=1):
    # noise generator for eval

    generator = torch.Generator(device=device).manual_seed(args.random_seed)
    scheduler.set_timesteps(ddim_steps)
    unet.eval()

    onsets = []
    offsets = []
    mixtures = []
    targets = []
    preds = []

    for step, (mixture, _, target, onset, offset, cls, _, _, file_id) in enumerate(tqdm(eval_loader)):
        # compress by vae
        # mixture = autoencoder.mel2emb(logmel(mixture))*args.scale_factor
        mixture = mixture.to(device)
        target = target.to(device)
        cls = cls.to(device)
        # event_tensor = event_tensor.to(device)

        mixture_mel = minmax_norm_diff(logmel_val(mixture)).unsqueeze(1)
        # timbre = logmel(timbre)
        cls = cls.long()
        # target_mel = autoencoder.mel2emb(logmel_val(target))*args.scale_factor
        target_mel = minmax_norm_diff(logmel_val(target)).unsqueeze(1)

        # init noise
        noise = torch.randn(mixture_mel.shape, generator=generator, device=device)
        pred = noise

        for t in scheduler.timesteps:
            pred = scheduler.scale_model_input(pred, t)
            model_output = unet(x=pred, t=t, mixture=mixture_mel, cls=cls,
                                timbre=None, timbre_feature=None, event=None)
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  eta=eta, generator=generator).prev_sample

        # pred = autoencoder.emb2mel(pred)
        pred = reverse_minmax_norm_diff(pred)
        pred_wav = autoencoder.mel2wav(pred.squeeze(1))

        onsets.append(onset)
        offsets.append(offset)
        mixtures.append(mixture)
        targets.append(target)
        preds.append(pred_wav)

        os.makedirs(f'{args.log_dir}/pic/{epoch}/', exist_ok=True)
        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        target_mel = reverse_minmax_norm_diff(target_mel)
        mixture_mel = reverse_minmax_norm_diff(mixture_mel)

        for j in range(pred.shape[0]):
            save_plot(pred[j].unsqueeze(0), f'{args.log_dir}/pic/{epoch}/pred_{file_id[j]}.png')
            save_audio(f'{args.log_dir}/audio/{epoch}/pred_{file_id[j]}', 16000, pred_wav[j].unsqueeze(0))

            if os.path.exists(f'{args.log_dir}/pic/{file_id[j]}.png') is False:
                # target = autoencoder.emb2mel(target)
                save_plot(target_mel[j].unsqueeze(0), f'{args.log_dir}/pic/gt/{file_id[j]}.png')
                save_audio(f'{args.log_dir}/audio/gt/{file_id[j]}', 16000, target[j].unsqueeze(0))

                save_plot(mixture_mel[j].unsqueeze(0), f'{args.log_dir}/pic/gt/mixture_{file_id[j]}.png')
                save_audio(f'{args.log_dir}/audio/gt/mixture_{file_id[j]}', 16000, mixture[j].unsqueeze(0))

    # onsets = torch.cat(onsets, dim=0)
    # offsets = torch.cat(offsets, dim=0)
    # mixtures = torch.cat(mixtures, dim=0)
    # targets = torch.cat(targets, dim=0)
    # preds = torch.cat(preds, dim=0)
    #
    # loss_sisnr_w, sisnrI_w, loss_sisnr_all, sisnrI_all = get_loss(preds, targets, mixtures, onsets, offsets)
    #
    # n = open(args.log_dir + 'eval.txt', mode='a')
    # n.write('\n')
    # n.write('Epoch: [{}][{}]   '
    #         'loss_sisnr_w: {:.6f}   sisnrI_w: {:.6f}   loss_sisnr_all: {:.6f}   sisnrI_all: {:.6f} \n'.format(
    #     epoch, args.epochs, loss_sisnr_w, sisnrI_w, loss_sisnr_all, sisnrI_all))
    # n.close()
