import pandas as pd
import numpy as np


zerpath = '.\data\input\Lens_Zernike_Lib.xlsx'
zer = pd.read_excel(zerpath, sheet_name='Sheet1', header=None, index_col=None)
zer = zer.drop(columns=[1, 4, 7, 9, 12, 14, 15, 17, 19, 22, 24, 26, 29, 31, 33, 35]).values
info = pd.read_excel(zerpath, sheet_name='Sheet2', header=None, index_col=None)
matrix = np.array([[1,4],[4,9],[1,5],[3,7],[6,12],[1,9]])

'statistical probability'
p = np.empty([14],float)
length = matrix.shape[0]
for i in range(length):
    [first,second] = matrix[i]
    diff1 =  abs(zer[:,first]) - abs(zer[:,second])
    p[i] = len(list(filter(lambda x: x >= 0, diff1))) / len(diff1)
print('---------probability---------')
for i in range(length):
    print('{}>={}------------------{:.2%}'.format(matrix[i][0],matrix[i][1],p[i]))



