from network import REAL
import torch 


def getEDVRSmallModel():
    model_path = './200000.pth'
    net = REAL(
            nbr=2,
            )
    net.eval()
    net.load_state_dict(torch.load(model_path), strict=True)
    print("opt. edvr have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
    return net 


