import torch
import torch.nn as nn
import torch.nn.functional as F

class L_ca(nn.Module):
    def __init__(self):
        super(L_ca, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):

        temp1 = -(1-self.alpha)*torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -self.alpha*torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)
        return CELoss+1.0*DICELoss
    
def kl_div(x1, x2, get_softmax=True):
    KLD = nn.KLDivLoss(reduction='batchmean')
    B = x1.size(0)

    if get_softmax:
        x1 = F.log_softmax(x1.view(B,-1),dim=-1)
        x2 = F.softmax(x2.view(B,-1),dim=-1)
        
    kl_loss = KLD(x1, x2)

    return kl_loss