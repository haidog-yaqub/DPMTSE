import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet2DModel
from .blocks import TimbreBlock
import yaml
# from diffusers.models.attention_processor import AttnProcessor2_0


class DiffTSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['fusion'] == 'cross_att':
            raise NotImplementedError
            # self.fusion = 'cross_att'
            # self.unet = UNet2DConditionModel(**self.config['unet'])
        elif config['fusion'] == 'concat':
            pre_hidden = config['pre_hidden']
            in_channel = config['unet']['out_channels'] * 2

            if self.config['use_timbre_feature']:
                self.use_timbre_feature = True
                timbre_feature_dim = config['timbre_feature_dim']
                self.feature_net = nn.Sequential(nn.Linear(timbre_feature_dim, pre_hidden*2), nn.SiLU(),
                                                 nn.Linear(pre_hidden*2, pre_hidden))
                in_channel += pre_hidden
            else:
                self.use_timbre_feature = False

            if self.config['use_timbre_model']:
                self.use_timbre_model = True
                self.timbre_model = TimbreBlock(pre_hidden)
                in_channel += pre_hidden
            else:
                self.use_timbre_model = False

            if self.config['use_event_ppg']:
                self.use_event_ppg = True
                self.ppg_model = nn.Sequential(nn.Linear(1, pre_hidden // 16), nn.SiLU(),
                                               nn.Linear(pre_hidden // 16, pre_hidden // 16)
                                               )
                in_channel += pre_hidden//16
            else:
                self.use_event_ppg = False

            config['unet']['in_channels'] = in_channel
            self.fusion = 'concat'
            self.unet = UNet2DModel(**self.config['unet'])
            self.unet.set_use_memory_efficient_attention_xformers(True)
            # self.unet.set_attn_processor(AttnProcessor2_0())

    def forward(self, x, t, mixture, cls=None, timbre=None, timbre_feature=None, event=None):
        timbre_all = []

        if self.use_timbre_feature:
            timbre_all.append(self.feature_net(timbre_feature))
        if self.use_timbre_model:
            timbre_all.append(self.timbre_model(timbre))

        if self.use_timbre_feature or self.use_timbre_model:
            timbre_all = torch.cat(timbre_all, dim=1).unsqueeze(2).unsqueeze(3)
            timbre_all = torch.cat(x.shape[2]*[timbre_all], 2)
            timbre_all = torch.cat(x.shape[3]*[timbre_all], 3)
            x = torch.cat([x, mixture, timbre_all], dim=1)
        else:
            x = torch.cat([x, mixture], dim=1)

        if self.use_event_ppg:
            event = event.unsqueeze(-1)
            event = self.ppg_model(event)
            event = torch.transpose(event, 1, 2).unsqueeze(2)
            event = torch.cat(x.shape[2]*[event], 2)
            x = torch.cat([x, event], dim=1)

        noise = self.unet(sample=x, timestep=t, class_labels=cls)['sample']

        return noise


if __name__ == "__main__":
    with open('DiffTSE_cls.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    device = 'cuda'

    model = DiffTSE(config['diffwrap']).to(device)

    x = torch.rand((1, 1, 64, 400)).to(device)
    t = torch.randint(0, 1000, (1, )).long().to(device)
    cls = torch.randint(0, 41, (1,)).long().to(device)

    mixture = torch.rand((1, 1, 64, 400)).to(device)
    timbre = None
    timbre_feature = None
    event = torch.rand(1, 400).to(device)
    # timbre = torch.rand(1, 64, 1000).to(device)
    # timbre_feature = torch.rand(1, 768).to(device)

    y = model(x, t, mixture, cls, timbre, timbre_feature, event)
