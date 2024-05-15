# --coding:utf-8--
# sys.path.append("..")
from model.stage_I.UNet import *
from utils.utils_optics import *

def StageI(gt):
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    gt = gt.unsqueeze(dim=0).unsqueeze(dim=0)
    model = UNet()
    model_path = '../model/stage_I/model_0601_dict_0.1TV.pth'
    A = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(A)
    matrix, mask, A, _ = fitting_prepare(1)
    x = torch.linspace(-1, 1, 256)
    [Y, X] = torch.meshgrid(x, x, indexing='ij')
    rou = abs(torch.sqrt(X ** 2 + Y ** 2))
    output = model(gt)
    zer_co, error = wavefront_fitting(output, matrix, mask, 1)
    direct_out = output.detach().numpy().reshape(256, 256)
    return zer_co[0,:,0].detach(),np.where(rou >= 1, 0, direct_out)




