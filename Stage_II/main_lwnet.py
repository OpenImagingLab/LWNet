# --coding:utf-8--
from model.stage_II.StageII_Net import *
from utils_model import *
from tqdm import tqdm
from utils_plot import *
from utils_optics import *

# input
filename, Sheet = 'WFA_65251.xlsx',8
# parameters
iters = 3
epochs_I, lr_I, gamma = 150, 0.5, 0.9
epochs_II, lr_II, T_max = 801, 3e-2 ,100

inputpath, savepath= '.\data', '.\\results'
filepath = os.path.join(inputpath, filename)
zerpath = os.path.join(inputpath, 'Lens_Zernike_Lib.xlsx')

x = np.linspace(-1, 1, 256)
Y, X = np.meshgrid(x, x)
rou = np.abs(np.sqrt(X ** 2 + Y ** 2))
input = 0.1 * torch.ones(16)

# load wavefront aberration (GT)
data = pd.read_excel(filepath, sheet_name='Sheet' + str(Sheet), header=None, index_col=None)
wf_gt = data.iloc[16:, :].values
wf_gt = np.array(wf_gt, dtype=np.float32)
a = "".join(list(filter(str.isdigit, data.iloc[8, 0][-15:-5])))
fov = float(a) / 100
lensname = data.iloc[2,0].split('\\')[-1].split('.')[0]
max_zer = up_limit_of_zer(zerpath, fov)
# generate PSF (GT)
GT = FourierTransform(wf_gt)

for iter in range(iters):
    psf_arr,wf_arr = [],[]
    psf_arr.append(GT)
    wf_arr.append(wf_gt)
    net = StageII()
    loss_MSE, loss_L1 = nn.MSELoss(), nn.L1Loss()
    zer_init,direct_out = StageI(GT)
    StageI_error = np.mean(abs(wf_gt - direct_out))
    print('Lens:{} @ fov = {:d}\u00b0, run count:{}'.format(lensname,int(fov),iter))

    'optimize the net to output predicted zernike coefficients'
    optimizer = torch.optim.Adam(net.parameters(), lr = lr_I)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    for epoch in range(epochs_I):
        optimizer.zero_grad()
        zer,wf,psf = net(input,max_zer)
        loss = loss_MSE(zer.reshape(21),zer_init.reshape(21))
        loss.backward()
        optimizer.step()

    'optimize the net by PSF identical loss'
    optimizer = torch.optim.AdamW(net.parameters(), lr = lr_II)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max)
    min_loss = float('inf')
    for epoch in tqdm(range(epochs_II), leave=False):
        optimizer.zero_grad()
        zer,wf,psf = net(input,max_zer)
        loss = loss_L1(psf.repeat(1, 1, 1, 1), torch.from_numpy(GT).repeat(1, 1, 1, 1))
        loss.backward()
        optimizer.step()

        if epoch % 10 ==0  and loss < min_loss:
            wfdata = wf.detach().numpy()
            wfdata = wfdata - wfdata[128,128]
            wfdata = np.where(rou>=1,0,wfdata)
            min_loss = loss.detach()
            psf_predict = psf.detach().numpy().reshape(256, 256)
            diff_wf = np.min([np.mean(abs(wfdata-wf_gt)), np.mean(abs(-wfdata-wf_gt))])

    psf_arr.append(psf_predict)
    wf_arr.append(wfdata)
    filename = 'lens_{}_fov_{:d}\u00b0_count_{}'.format(lensname,int(fov),iter)
    multi_plot(psf_arr, wf_arr, filename, savepath)
    print('L1 Loss of LWNet:{:.8f},  L1 loss of Stage I:{:.8f}\n'.format(diff_wf, StageI_error))





