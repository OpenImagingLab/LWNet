
import os
from utils_plot import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def image_pre(ax_list):
    for i in range(len(ax_list)):
        temp = ax_list[i]
        temp.set_xticks([])
        temp.set_yticks([])
        temp.spines['top'].set_color('none')
        temp.spines['bottom'].set_color('none')
        temp.spines['left'].set_color('none')
        temp.spines['right'].set_color('none')

def multi_plot(psf,wf,filename,savepath):
    max, min = np.max(np.array(wf)), np.min(np.array(wf))
    psf_gt, psf_predict= psf[0], psf[1]
    wf_gt, wf_predict = wf[0], wf[1]
    fig = plt.figure(figsize=(21, 5), dpi=100)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.05)
    ax1, ax2, ax3, ax4= fig.add_subplot(1, 4, 1), fig.add_subplot(1, 4, 2), fig.add_subplot(1, 4, 3), fig.add_subplot(1, 4, 4)
    ax1.imshow(psf_gt,cmap='bone')
    ax1.set_title("PSF (GT)")
    ax2.imshow(wf_gt,cmap='jet',vmin=min,vmax=max)
    ax2.set_title("Wavefront aberration (GT)")
    ax3.imshow(psf_predict,cmap='bone')
    ax3.set_title("PSF (Reconstructed by LWNet)")
    ax4.imshow(wf_predict,cmap='jet',vmin=min,vmax=max)
    ax4.set_title("Wavefront aberration (Recovered by LWNet)")
    ax_list = [ax1,ax2,ax3,ax4]
    image_pre(ax_list)
    tick_number = np.trunc(np.linspace(min,max,4)*10)/10
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # print(min,max)
    clbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=[ax1,ax2,ax3,ax4], shrink=0.75,ticks=tick_number)
    clbar.ax.tick_params(labelsize=20)
    savefilepath = os.path.join(savepath,filename+'.png')
    plt.savefig(savefilepath,bbox_inches='tight')
    print('image has been saved in {}'.format(savefilepath))
