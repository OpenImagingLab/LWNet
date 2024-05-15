# --coding:utf-8--
from model.stage_II.StageII_Net import *
from tqdm import tqdm
from utils.utils_plot import *
from utils.pretreatment import *
import yaml

def config(source):
    # Config file
    current_path = os.getcwd()
    with open(source) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args['in_path']= os.path.join(current_path,args['inputpath'], args['filename'] + '.xlsx')
    args['zerpath'] = os.path.join(current_path,args['inputpath'], 'Lens_Zernike_Lib.xlsx')
    args['savepath']= os.path.join(current_path,args['savepath'])
    args['modelpath'] = os.path.abspath(args['modelpath'])

    list= ['epochs_I','epochs_II','T_max','seed']
    for key in list:
        args[key] = int(args[key])

    list= ['lr_I','lr_II','gamma']
    for key in list:
        args[key] = float(args[key])

    return args

if __name__ == "__main__":

    source = 'configs/lwnet.yaml'
    args = config(source)
    torch.manual_seed(args['seed'])

    x = np.linspace(-1, 1, 256)
    Y, X = np.meshgrid(x, x)
    rou = np.abs(np.sqrt(X ** 2 + Y ** 2))

    # load wavefront aberration (GT)
    data = pd.read_excel(args['in_path'], sheet_name = args['Sheet'], header=None, index_col=None)
    wf_gt = data.iloc[16:, :].values
    wf_gt = np.array(wf_gt, dtype=np.float32)
    a = "".join(list(filter(str.isdigit, data.iloc[8, 0][-15:-5])))
    fov = float(a) / 100
    lensname = data.iloc[2,0].split('\\')[-1].split('.')[0]
    max_zer = up_limit_of_zer(args['zerpath'], fov)
    # generate PSF (GT)
    GT = FourierTransform(wf_gt)

    # for iter in range(iters):
    psf_arr,wf_arr = [],[]
    psf_arr.append(GT)
    wf_arr.append(wf_gt)
    net = StageII()
    loss_MSE, loss_L1 = nn.MSELoss(), nn.L1Loss()
    zer_init, direct_out = precondition(GT,args['modelpath'])
    zer_init = sample(GT, zer_init, num=50)
    # zer_init,direct_out = StageI(GT)
    StageI_error = np.mean(abs(wf_gt - direct_out))
    print('Lens:{} @ fov = {:d}\u00b0'.format(lensname,int(fov),iter))

    'optimize the net to output predicted zernike coefficients'
    optimizer = torch.optim.Adam(net.parameters(), lr = args['lr_I'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma= args['gamma'])
    for epoch in range(args['epochs_I']):
        optimizer.zero_grad()
        zer,wf,psf = net(max_zer)
        loss = loss_MSE(zer.reshape(21), zer_init.reshape(21))
        loss.backward()
        optimizer.step()
    print('L1 Loss of zer:{:.8f}'.format(loss))

    'optimize the net by PSF identical loss'
    optimizer = torch.optim.AdamW(
        [{'params': net.fc0.parameters(), 'lr': args['lr_II']}, {'params': net.fc1.parameters(), 'lr': args['lr_II']},
         {'params': net.linears.parameters(), 'lr': args['lr_II']},{'params': net.res.parameters(), 'lr': 0.01*args['lr_II']}])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args['T_max'])
    min_loss = float('inf')
    for epoch in tqdm(range(args['epochs_II']), leave=False):
        optimizer.zero_grad()
        zer,wf,psf = net(max_zer)
        loss = loss_L1(psf.repeat(1, 1, 1, 1), torch.from_numpy(GT).repeat(1, 1, 1, 1))
        loss.backward()
        optimizer.step()

        if epoch > 200 and epoch % 10 ==0  and loss < min_loss:
            wfdata = wf.detach().numpy()
            wfdata = wfdata - wfdata[128,128]
            wfdata = np.where(rou>=1,0,wfdata)
            min_loss = loss.detach()
            psf_predict = psf.detach().numpy().reshape(256, 256)
            diff_wf = np.min([np.mean(abs(wfdata-wf_gt)), np.mean(abs(-wfdata-wf_gt))])
            print('L1 Loss of wf:{:.8f}, l1 loss of PSF:{:.8f}\n'.format(diff_wf, min_loss))

    psf_arr.append(psf_predict)
    wf_arr.append(wfdata)
    print('L1 loss of Stage I:{:.8f},  L1 Loss of LWNet:{:.8f}\n'.format(StageI_error, diff_wf))
    filename = 'lens_{}_fov_{:d}\u00b0'.format(lensname,int(fov))
    multi_plot(psf_arr, wf_arr, filename, args['savepath'])





