import torch
import torch.nn as nn
import pdb

from collections import OrderedDict
from .dgcnn import DGCNN
from einops import rearrange, repeat
from .hrnet.hrnet_cls_net_featmaps import get_cls_net
from .hrnet.config import update_config as hrnet_update_config
from .hrnet.config import config as hrnet_config
from .atten_utils import Transformer, Transformer_parallel

class Frame_Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        hrnet_yaml = 'tools/modules/hrnet/config/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        hrnet_checkpoint = 'tools/modules/hrnet/config/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        self.backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        self.hidden_dim = hidden_dim
        self.dim_down = nn.Sequential(
            nn.Conv2d(2048, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )

    def forward(self, img):
        '''
        img: [B,T,3,H,W]
        '''
        B,T,_,H,W = img.shape
        img = img.contiguous().view(B*T, -1, H, W)
        out = self.backbone(img)
        out = self.dim_down(out)
        feats = out.contiguous().view(B, T, self.hidden_dim, -1).permute(0, 1, 3, 2)
        feats = feats.contiguous().view(B, -1, self.hidden_dim)                         #[B, THW, C]
        return feats
    
class Modality_wise_Encoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.f_i = Frame_Encoder(args.MODEL.emb_dim)
        self.f_st = Transformer(args.MODEL.emb_dim, depth = 1, heads = 12, dim_head = 64, mlp_dim = 768, dropout = 0.3)
        self.f_m = nn.Sequential(
            nn.Linear(12, 256),
            SwapAxes(),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
            nn.Linear(256, 768)
        )
        '''
        We have integrated the motion encoder weights into the 
        final .pt ckpt and no longer need them here
        
        motion_ckpt = torch.load(args.MODEL.motion_ckpt, map_location='cpu')
        state_dict = motion_ckpt['state_dict']
        self.f_m.load_state_dict(state_dict, strict=False)
        '''
        self.f_o = DGCNN(device=device, emb_dim=args.MODEL.emb_dim)

    def forward(self, V, M, O):

        frames = V.permute(0,2,1,3,4)
        visual_feats = self.f_i(frames)
        F_V = self.f_st(visual_feats, visual_feats)
        B,THW,C = F_V.shape
        T = int(THW / 49)
        M = M.view(B, T, -1)
        F_M = self.f_m(M)
        F_O = self.f_o(O)

        return F_V, F_M, F_O

class Aff_Contact(nn.Module):
    def __init__(self, time, emb_dim):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.Theta_a = Transformer_parallel(emb_dim, depth = 1, heads = 12, dim_head = 64, mlp_dim = 768, dropout = 0.1)
        self.f_a = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            SwapAxes(),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            SwapAxes()
        )
        self.Theta_c = Transformer_parallel(emb_dim, depth = 1, heads = 12, dim_head = 64, mlp_dim = 768, dropout = 0.1)
        self.f_c = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            SwapAxes(),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            SwapAxes()
        )
        self.f_ca = Transformer(emb_dim, depth = 1, heads = 12, dim_head = 64, mlp_dim = 768, dropout = 0.1)

        self.logits_func = nn.Parameter(torch.zeros(1,1,emb_dim), requires_grad=True)
        self.logits_inten = nn.Parameter(torch.zeros(1,1,emb_dim), requires_grad=True)
        self.tau_v = nn.Parameter(torch.zeros(1,1,emb_dim), requires_grad=True)
        self.tau_o = nn.Parameter(torch.zeros(1,1,emb_dim), requires_grad=True)
        self.tau_m = nn.Parameter(torch.zeros(1,1,emb_dim), requires_grad=True)
        self.pe_t = nn.Parameter(torch.zeros(1,time,1,emb_dim), requires_grad=True)

    def forward(self, F_V, F_M, F_O):
        '''
        feats_3d: [B, N_3d, C]
        feats_video: [B,THW,C]
        '''
        B,THW,C = F_V.shape
        T = int(THW / 49)

        logits_func = repeat(self.logits_func, '() n d -> b n d', b = B)
        logits_inten = repeat(self.logits_inten, '() n d -> b n d', b = B)

        tau_v = repeat(self.tau_v, '() n d -> b n d', b = B)
        tau_o = repeat(self.tau_o, '() n d -> b n d', b = B)
        tau_m = repeat(self.tau_m, '() n d -> b n d', b = B)
        pe_t = repeat(self.pe_t, '() t n d -> b t n d', b = B)

        F_V = F_V.view(B, T, -1, C) + pe_t
        F_V = F_V.contiguous().view(B, -1, C)
        F_M = F_M + pe_t.squeeze(dim=2)

        query_3d = torch.cat((logits_func, F_O), dim=1)
        aff_v, aff_m = self.Theta_a(query_3d, F_V*tau_v, F_M*tau_m)
        F_a = torch.cat((aff_v*tau_v, aff_m*tau_m), dim=2)
        F_a = self.f_a(F_a)
        fun_semantic = F_a[:,0,:]
        F_a = F_a[:,1:,:]

        query_v = torch.cat((logits_inten, F_V), dim=1)
        contact_o, contact_m = self.Theta_c(query_v, F_a*tau_o, F_M*tau_m)                 #[B, 1+THW, C]
        F_c = torch.cat((contact_o*tau_o, contact_m*tau_m), dim=2)
        F_c = self.f_c(F_c)
        inten_semantic = F_c[:,0,:]
        F_c = F_c[:,1:,:]     
        F_a = self.f_ca(F_a, F_c)
        F_c = F_c.view(B,T,-1,C)
        F_s = torch.cat((fun_semantic, inten_semantic), dim=-1)

        return F_c, F_a, F_s
    
class Decoder(nn.Module):
    def __init__(self, emb_dim, logits_class):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
            
        self.contact_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            SwapAxes(),
            nn.BatchNorm1d(256),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.aff_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            SwapAxes(),
            nn.BatchNorm1d(256),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.logits_head = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, logits_class)
        )
        self.spatial_mapping = nn.Sequential(
            nn.Linear(49, 2048),
            SwapAxes(),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(2048, 6890),
            SwapAxes(),
            nn.BatchNorm1d(6890),
            SwapAxes(),
            nn.Sigmoid()
        )
    def forward(self, F_c, F_a, F_s):
        B,T,_,C = F_c.shape
        F_c = F_c.contiguous().view(B*T,-1,C)
        contact_feats = self.contact_head(F_c)
        contact = self.spatial_mapping(contact_feats.permute(0, 2, 1))                          
        contact = contact.contiguous().view(B, T, 6890, -1)
        affordance = self.aff_head(F_a)
        logits = self.logits_head(F_s)

        return contact, affordance, logits
    
class EgoChoir(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder = Modality_wise_Encoder(args, device)
        self.modeling_aff_contact = Aff_Contact(args.DATA.num_frames, args.MODEL.emb_dim)
        self.decoder = Decoder(args.MODEL.emb_dim, len(args.DATA.affordances))

    def forward(self, V, M, O):
        F_V, F_M, F_O = self.encoder(V, M, O)
        F_c, F_a, F_s = self.modeling_aff_contact(F_V, F_M, F_O.permute(0, 2, 1))
        contact, affordance, logits = self.decoder(F_c, F_a, F_s)

        return contact.contiguous(), affordance, logits