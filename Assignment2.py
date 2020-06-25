'''Normando de Campos Amazonas Filho, 11561949
Image Processing, SCC0251_Turma01_1Sem_2020_ET
Assignment 1: intensity transformations
https://github.com/normandoamazonas/Assignment2'''



import numpy as np
import matplotlib.pyplot as plt
import imageio as imageio
import math

'''Functions definitions'''

kernel1= [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
kernel2= [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

kernels = [kernel1,kernel2]
def crop(im):
   w,h=im.shape
   return im[1:w-1, 1:h-1]
def padding(im):
   w,h=im.shape
   im2= np.zeros ((w+2,h+2),np.float32)
   im2[1:w+1, 1:h+1] = im  
   return im2

def conv2d(image,kernel):
    temp=padding(image)
    w,h=temp.shape
    wk,hk=kernel.shape    
    result= np.zeros ((w,h),np.float32)
    mk=int((wk-1)/2)
    for i in range(mk,h-mk):
      for j in range(mk,h-mk):
           reg=temp[i-mk:i+mk+1,j-mk:j+mk+1]
           result[i,j]=np.sum(reg*kernel)

    return crop(result).astype(np.uint8)

'''def padding (input_image):
    saida= np.pad (input_image, ((1,1),(1,1)), 'constant') #function to fill the borders of matrix with zeros
    return saida'''

def G (x, sigma):
     
    p= 1/np.sqrt(2*np.pi *sigma**2) * np.exp (-x**2/(2*sigma**2))
    #p= 1/(2*np.pi *sigma**2) * np.exp (-x**2/(2*sigma**2))
          # print("p\n %.7f"%p)
    return p

def E (x, y):
    return np.sqrt (x**2 + y**2)

'''def kernel(kernlen=3, nsig=1): #numpy
    Returns a 2D Gaussian kernel array.

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel'''

def kernelgs (n, sigma): 
    k= np.zeros ((n,n),np.float64)
    #k= [[0]*n]*n
    cx =int (n/2)
    cy= int (n/2)
    
    for i in range (n):  #o..n-1
        for j in range (n):
           #print (i,j,E (cx-j, cy-i))
           k [i,j]= G (E (cx-j, cy-i), sigma)
           #print (k[i,j])

    return k

def kernelgr (I, n, sigma): 
    k= np.zeros ((n,n),np.float64)
    #k= [[0]*n]*n
    cx =int (n/2)
    cy= int (n/2)
   
    for i in range (n):
        for j in range (n):
           k [i,j]= G ( I[cx, cy]- I[j, i], sigma)

    return k


def conv2d_alternative (image, kernel):
    from scipy import signal
    return signal.convolve2d (image, kernel, 'valid')
    
'''def conv2d_point (f, w, x, y):

    w_flip = np.flip (np.flip (w,0), 1)

    n,m = w.shape #dimensions of w
    a= int ((n-1)/2)
    b= int ((m-1)/2)

    #region centred at x, y
    region_f = f [x-a : x+ (a+1), y-b: y+(b+1)]
    value = np.sum (np.multiply (region_f, w_flip))
    return value


def con2d_image (f, w):

    N,M = f.shape #dimensions of f
    n,m = w.shape #dimensions of w

    w_flip = np.flip (np.flip (w, 0), 1)

    a= int ((n-1)/2)
    b= int ((m-1)/2)

    #create a new image to store the filtered pixels
    g= np.zeros (f.shape, dtype= np.uint8)

    #for every pixel that is valid for convolution
    for x in range (a, N-a):
        for y in range (b, M-b):
            #region centred at x, y
            region_f = f [x-a : x+ (a+1), y-b: y+(b+1)]
            g [x, y] = np.sum (np.multiply (region_f, w_flip)).astype (np.uint8)

    return g'''

def Scaling(I):
    Min = np.min(I)
    Max = np.max(I)
    #n,m =I.shape
    im= (I-Min)*(255.0/float(Max-Min))
    return im#.astype(int)

def Adding(I,c,r):
    return c*I + r


def crop (image):
    w,h=image.shape
    return image [1:w-1,1:h-1]




#Error from image reference to processed image
def RSE (img_final, input_image):  
    erro = np.sqrt(np.sum(np.square(img_final.astype(float) - input_image.astype(float))))
    return erro


       


def process(reg,nk,sigmae,sigmai):
  gs=kernelgs(nk,sigmae) #calcula o kernel gaussiano espacial
  #gs=gs/np.sum(gs)
  gr= kernelgr (reg, nk, sigmai) #calcula o kernel gaussiano de intensidade
  #gr=gr/np.sum(gr)
  wi= gr*gs #combinando ambos os filtros
  Wp= np.sum (wi) #calculando o fator de normalização
  If= np.sum (reg*wi)  #operando na região de imagem

  #print (reg)
  #print (gs)
  #print (gr)

  return If/Wp


def method1 (image,nk,sigmae,sigmai):
    temp=padding(image)
    w,h=temp.shape
    

    result= np.zeros ((w,h),np.float32)
    mk=int((nk-1)/2)

    for i in range(mk,h-mk):
      for j in range(mk,w-mk):


           reg=temp[i-mk:i+mk+1,j-mk:j+mk+1] #região da imagem a multiplicar
           result[i,j]=process(reg,nk,sigmae,sigmai).astype(np.uint8)
           #exit (0)

    return crop(result)

def method2 (image, c, nkernel):

    #im = con2d_image(r,kernels[nkernel-1])
    im2= padding (image)
    im3= conv2d_alternative (im2, np.array(kernels[nkernel-1]))
    #im3= conv2d (im2, np.array(kernels[nkernel-1]))
    im4 = Scaling(im3)
    print (im4.shape, im2.shape)
    im5 = Adding(im4,c,image)
    im6 = Scaling(im5)
    return im6




def method3 (input_image, sigma_row, sigma_col):
    pass

filename = str (input ()).rstrip()
input_image = imageio.imread(filename)
method = int (input ()) # 1, 2 or 3
save = int(input ()) # 0, 1

if method == 1:
    n = int (input()) #size of the filter
    sigma_s = float (input())
    sigma_r = float (input ())
    img_final = method1 (input_image, n, sigma_s, sigma_r)

if method == 2:
    c = float (input()) # <= 1
    kernel = int (input()) # 1 or 2
    img_final = method2 (input_image, c, kernel)
    

if method == 3:
    sigma_row = float (input())
    sigma_col = float (input())
    img_final = method3 (input_image, sigma_row, sigma_col)

erro=RSE(img_final,input_image)
print(round(erro,4))


