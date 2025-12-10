import numpy as np
import matplotlib.pyplot as plt

"""FUNZIONI"""

def Half_mask(T, sigma):
    var = -np.log(T) * 2 * (sigma ** 2)
    return np.round(np.sqrt(var))

def Filter_size(T, sigma):
    return 2*Half_mask(T, sigma) + 1


def MaskGen(T, sigma):
    N = Filter_size(T, sigma)
    half = Half_mask(T, sigma)
    y, x = np.meshgrid(range(-int(half), int(half) + 1), range(-int(half), int(half) + 1))
    return x, y

def fGaussian(x,y, sigma):
    var = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return (np.exp(-var))

def gradientX(x,y, sigma):
    var = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((x * np.exp(-var)) / sigma ** 2)


def gradientY(x,y, sigma):
    var = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((y * np.exp(-var)) / sigma ** 2)

def Gx(fx, fy,sigma):
    gx = gradientX(fx, fy, sigma)
    gx = (gx * 255)
    return np.around(gx)

def Gy(fx, fy,sigma):    
    gy = gradientY(fx, fy, sigma)
    gy = (gy * 255)
    return np.around(gy)

def pad(Img, kernel):
    righe, colonne = Img.shape
    krighe, kcolonne = kernel.shape
    padd = np.zeros((righe + krighe,colonne + kcolonne), dtype=Img.dtype)
    insert =int((krighe)/2)
    padd[insert: insert + righe, insert: insert + colonne] = Img
    return padd

def smooth(Img, kernel=None):
    if kernel is None:
        maschera = np.array([[1,1,1],[1,1,1],[1,1,1]])
    else:
        maschera = kernel
    i, j = maschera.shape
    var = np.zeros((Img.shape[0], Img.shape[1]))           
    image_padded = pad(Img, maschera)
    for x in range(Img.shape[0]):    
        for y in range(Img.shape[1]):
            var[x, y] = (maschera * image_padded[x:x+i, y:y+j]).sum() / maschera.sum()  
    return var

def ApplyGradientMask(Img, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))    
    var = np.zeros_like(Img)           
    image_padded = pad(Img, kernel)
    for x in range(Img.shape[0]):    
        for y in range(Img.shape[1]):
            var[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()        
    return var

def Gradient_Magnitude(fx, fy):
    mg = np.zeros((fx.shape[0], fx.shape[1]))
    mg = np.sqrt((fx ** 2) + (fy ** 2))
    mg = mg * 100 / mg.max()
    return np.around(mg)

def DirezioneGradiente(fx, fy):
    Gdir = np.zeros((fx.shape[0], fx.shape[1]))
    Gdir = np.rad2deg(np.arctan2(fy, fx)) + 180
    return Gdir

def Quantizza_angoli(Angolo):
    quantized = np.zeros((Angolo.shape[0], Angolo.shape[1]))
    for i in range(Angolo.shape[0]):
        for j in range(Angolo.shape[1]):
            if 0 <= Angolo[i, j] <= 22.5 or 157.5 <= Angolo[i, j] <= 202.5 or 337.5 < Angolo[i, j] < 360:
                quantized[i, j] = 0
            elif 22.5 <= Angolo[i, j] <= 67.5 or 202.5 <= Angolo[i, j] <= 247.5:
                quantized[i, j] = 1
            elif 67.5 <= Angolo[i, j] <= 122.5 or 247.5 <= Angolo[i, j] <= 292.5:
                quantized[i, j] = 2
            elif 112.5 <= Angolo[i, j] <= 157.5 or 292.5 <= Angolo[i, j] <= 337.5:
                quantized[i, j] = 3
    return quantized

def NonMaxSupp(quantZ, Direction, magnitude):
    M = np.zeros(quantZ.shape)
    a, b = np.shape(quantZ)
    for i in range(a-1):
        for j in range(b-1):
            if quantZ[i,j] == 0:
                if  Direction[i,j-1]< Direction[i,j] or Direction[i,j] > Direction[i,j+1]:
                    M[i,j] = magnitude[i,j]
                else:
                    M[i,j] = 0
            if quantZ[i,j]==1:
                if  Direction[i-1,j+1]<= Direction[i,j] or Direction[i,j] >= Direction[i+1,j-1]:
                    M[i,j] = magnitude[i,j]
                else:
                    M[i,j] = 0       
            if quantZ[i,j] == 2:
                if  Direction[i-1,j]<= Direction[i,j] or Direction[i,j] >= Direction[i+1,j]:
                    M[i,j] = magnitude[i,j]
                else:
                    M[i,j] = 0
            if quantZ[i,j] == 3:
                if  Direction[i-1,j-1]<= Direction[i,j] or Direction[i,j] >= Direction[i+1,j+1]:
                    M[i,j] = magnitude[i,j]
                else:
                    M[i,j] = 0
    return M

def DoubleThresholding(nms, low_threshold, high_threshold):
    Gthresholded = np.zeros(nms.shape)
    for i in range(0, nms.shape[0]):		
        for j in range(0, nms.shape[1]):
            if nms[i,j] < low_threshold:	
                Gthresholded[i,j] = 0
            elif nms[i,j] >= low_threshold and nms[i,j] < high_threshold:
                Gthresholded[i,j] = 128
            else:					        
                Gthresholded[i,j] = 255
    return Gthresholded

"""CANNY IMPLEMENTAZIONE"""
def canny_pip(img, low=10, high=100, sigma=0.5,T=0.3):
    x, y = MaskGen(T, sigma)

    gauss = fGaussian(x, y, sigma)
    smooth_img = smooth(img, gauss)

    gx = -Gx(x, y,sigma)
    gy = -Gy(x, y,sigma)

    fx = ApplyGradientMask(smooth_img, gx)
    fy = ApplyGradientMask(smooth_img, gy)

    mg = Gradient_Magnitude(fx, fy)
    mg = mg.astype(int)

    Angolo = DirezioneGradiente(fx, fy)

    quantZ = Quantizza_angoli(Angolo)
    nms = NonMaxSupp(quantZ, Angolo, mg)
    threshold = DoubleThresholding(nms, low, high)
    print(threshold.min(), threshold.max(), threshold.dtype)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(threshold, cmap='gray')
    plt.axis("off")
    plt.show()


