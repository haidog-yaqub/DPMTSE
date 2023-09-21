import os
import torchaudio
import pandas as pd
from tqdm import tqdm

import torch
from timbre_model import load_model, get_embedding

device = 'cuda'


# @torch.no_grad()
# def extract_timbre(model, audio):
#     # padding_mask = torch.zeros_like(audio).bool()
#     padding_mask = None
#     feature = model.extract_features(audio, padding_mask=padding_mask)[0]
#     feature_mean = torch.mean(feature, dim=1)
#     return feature_mean.cpu().numpy()


if __name__ == "__main__":

    model = load_model('timbre_model/MaskSpec/save/AudioSet_Pretrained_Finetuned.pth', target_size=(128, 1000),
                       norm_file='timbre_model/MaskSpec/audioset/mean_std_128.npy')
    # load the pre-trained checkpoints
    # checkpoint = torch.load('beats/BEATs_iter3_plus_AS2M.pt')
    # cfg = BEATsConfig(checkpoint['cfg'])
    # BEATs_model = BEATs(cfg)
    # BEATs_model.load_state_dict(checkpoint['model'])
    # BEATs_model.eval()
    # BEATs_model.to(device)

    # BEATs_model = torch.compile(BEATs_model)

    folder = 'data/fsd2018/'
    output_folder = 'data/fsd2018/timbre_masked/'
    # file list
    df = pd.read_csv(folder + 'meta.csv')

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        file = row['file']
        subfolder = row['folder']
        timbre_file = file.replace('.wav', '_re.wav')
        # audio, sr = torchaudio.load(folder+subfolder+'/'+timbre_file)
        # audio = audio.to(device)
        # feature = extract_timbre(BEATs_model, audio)
        feature, _, print_out, _ = get_embedding(model, folder+subfolder+'/'+timbre_file, audio_length=10)

        output_path = output_folder + subfolder + '/' + timbre_file.replace('.wav', '.pt')
        if os.path.exists(os.path.dirname(output_path)) is False:
            os.makedirs(os.path.dirname(output_path))

        torch.save(feature, output_path)
        # print(print_out)










