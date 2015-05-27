import numpy as np
from PIL import Image

def lowpass(imarray, filtersize=3):
    halff = filtersize / 2

    newarray = imarray.copy()
    nx, ny = np.shape(newarray)

    # extend by replication
    for i in range(halff):
        newarray = np.concatenate((newarray[:, 0:1], newarray, newarray[:, -1:]), 1)
        newarray = np.concatenate((newarray[0:1, :], newarray, newarray[-1:, :]), 0)

    filterarr = np.ones((filtersize, filtersize)) / float(filtersize ** 2)
    
    nl = [(newarray[i-halff:i+halff+1, j-halff:j+halff+1] * filterarr).sum() \
            for i in range(halff, halff + nx)
            for j in range(halff, halff + ny)]    

    nl = np.reshape(nl, (nx, ny))

    return nl

def calcmse(data1, data2):
    nx, ny = np.shape(data1)
    mse = np.sum([(data1[i, j] - data2[i, j]) ** 2 \
                for i in range(nx)
                for j in range(ny)]) / float(nx * ny)
    return mse

def calcpsnr(data1, data2):
#    maxpixel = max(data1.max(), data2.max())
    maxpixel = 1.0
    mse = calcmse(data1, data2)
    psnr = 10 * np.log10(maxpixel ** 2 / mse)
#    print "maxI", maxpixel, "psnr", psnr
    return psnr
    

def hw2q7():
    
    imagefn = 'C:\\shanying\\mooc\\digital-002\\digital_images_week2_quizzes_lena.gif'
    im = Image.open(imagefn)
    nx, ny = im.size
    idata = im.getdata()
    newdata = np.reshape(np.array(idata), (nx, ny)) / 255.0
    
    newdata3 = lowpass(newdata, 3)    
    psnr = calcpsnr(newdata, newdata3)
    
    newdata5 = lowpass(newdata, 5)    
    psnr = calcpsnr(newdata, newdata5)
    
    #return newdata, newdata3, newdata5
    return

    