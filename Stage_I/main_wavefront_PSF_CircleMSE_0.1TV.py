# --coding:utf-8--
import torch
import os
from torch.utils.data import DataLoader
from UNetbig2 import *
from torch.autograd import Variable
from Dataset_from_h5 import *
import pandas as pd


if __name__ == '__main__':
    epochnum = 601
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    BS =32

    model = UNet()
    addition = TVLoss()

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except Exception as e:
            model = model().cuda()

    loss_func = nn.MSELoss()
    loss_func2 = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    #prepare data
    savepath = './Model/loss_circlemse_0.1TV.csv'
    ckpt_dir = './Model'
    train_path = './h5py/train_psf_wavefront.h5'
    valid_path = './h5py/valid_psf_wavefront.h5'

    train_dataset = Dataset_from_h5(train_path)
    valid_dataset = Dataset_from_h5(valid_path)
    dataloader = DataLoader(dataset=train_dataset, batch_size=BS, shuffle=True, num_workers=8, drop_last=True)
    dataloader_val = DataLoader(dataset=valid_dataset, batch_size=BS, shuffle=True, num_workers=8, drop_last=True)
    loss_ar = np.zeros([1300,4])
    index = 0

    x = torch.linspace(-1, 1, 256)
    X, Y = torch.meshgrid(x, x)
    rou2 = X ** 2 + Y ** 2

    zero = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        rou2, zero = rou2.cuda(), zero.cuda()
        BS = 32

    for epoch in range (epochnum):
        loss_sum = 0
        loss_sum_l1 = 0
        for i, data in enumerate(dataloader):
            # print(i)
            input, label = data

            if torch.cuda.is_available():
                input, label = input.cuda(), label.cuda()

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            output = model(input)

            circleout = torch.where(rou2 >= 1, zero, output)
            loss1 = loss_func(circleout, label)
            loss2 = addition(circleout)
            loss = loss1 +0.1*loss2
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss3 = loss_func2(circleout, label)
            loss_sum_l1 += loss3.item()

        scheduler.step()
        loss_train = loss_sum / len(dataloader)
        loss_train_l1 = loss_sum_l1 / len(dataloader)
        print('epoch:{:0>3}, Training Loss:{:.8f}, L1 Loss:{:.8f}'.format(epoch, loss_train,loss_train_l1))

        if epoch% 10==0:
            loss_val = 0
            loss_val_l1 = 0
            model.eval()
            for i,data in enumerate(dataloader_val):
                input,label = data
                if torch.cuda.is_available():
                    input, label = input.cuda(), label.cuda()
                input, label = Variable(input), Variable(label)
                test_out = model(input)
                test_out.detach()
                circletestout = torch.where(rou2 >= 1, zero, test_out)

                loss_val += loss_func(circletestout,label).item()
                loss_val_l1 += loss_func2(circletestout, label).item()

            loss_val = loss_val/len(dataloader_val)
            loss_val_l1 = loss_val_l1 / len(dataloader_val)

            loss_ar[index,0] = loss_train
            loss_ar[index,1] = loss_train_l1
            loss_ar[index,2] = loss_val
            loss_ar[index,3] = loss_val_l1
            index  +=1
            print('epoch:{:0>3}, Training Loss:{:.8f}, Validation Loss:{:.8f}'.format(epoch,loss_train,loss_val))

        if epoch % 100 == 0 and epoch>0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
            pd.DataFrame(loss_ar).to_csv(savepath, header=False, index=False)

