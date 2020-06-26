'''Normando de Campos Amazonas Filho, 11561949
Image Enhancement, SCC0251_Turma01_1Sem_2020_ET
Assignment 2: Image Enhancement and Filtering
https://github.com/normandoamazonas/Assignment2'''



import numpy as np
import imageio as imageio
import math

'''Functions definitions'''

kernel1= [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
kernel2= [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

kernels = [kernel1,kernel2]

def padding (image,nk):
    bs=int( (nk-1)/2 ) #(3-1)/2=1
    saida= np.pad (image, ((bs,bs),(bs,bs)), 'constant') #function to fill the borders of matrix with zeros
    return saida

def crop (image,nk):
    bs=int( (nk-1)/2 ) #(3-1)/2=1
    w,h=image.shape 
    return image[bs:w-bs,bs:h-bs]


def conv2d(image,kernel):
    nk=kernel.shape[0]
    h,w=image.shape
    result= np.zeros ((h,w),np.float32)

    mk=int((nk-1)/2)

    for i in range(mk,h-mk):
      for j in range(mk,w-mk):
           reg=image[i-mk:i+mk+1,j-mk:j+mk+1]
           result[i,j]=np.sum(reg*kernel)

    return result



def G (x, sigma):
     
    p= (1/(2.0*np.pi *sigma**2)) * np.exp (-x**2/(2*sigma**2))
#    print("p\n %.7f"%p)
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
    k= np.zeros ((n,n),np.float32)
    #k= [[0]*n]*n
    cx =int ((n-1)/2)
    cy= int ((n-1)/2)
    
    for i in range (n):  #o..n-1
        for j in range (n):
           #print (i,j,E (cx-j, cy-i))
           k [i,j]= G (E (cx-j, cy-i), sigma)
           #print (k[i,j])

    return k

def kernelgr (I, n, sigma): 
    k= np.zeros ((n,n),np.float32)
    #k= [[0]*n]*n
    cx =int ((n-1)/2)
    cy= int ((n-1)/2)

    for i in range (n):
        for j in range (n):
           diff=int(I[cy,cx])-int(I[i,j])
           k [i,j]= G ( diff, sigma)
    return k



def Scaling(I):
    Min = np.min(I)
    Max = np.max(I)
    #n,m =I.shape
    im= (I-Min)*(255.0/float(Max-Min))
    return im

def Adding(I,c,r):
    return c*I + r

#Error from image reference to processed image
def RSE (img_final, input_image):  
    erro = np.sqrt(np.sum(np.square(img_final.astype(float) - input_image.astype(float))))
    return erro

def process(reg,nk,sigmas,sigmar):
  gs=kernelgs(nk,sigmas) # calculates the spatial Gaussian kernel
  gr= kernelgr (reg, nk, sigmar) #calculates the intensity Gaussian kernel
  wi= gr*gs #combining both filters
  Wp= np.sum (wi) #calculating the normalization factor
  If= np.sum (reg*wi) #operating the imaging region


  return If/Wp


def method1 (image,nk,sigmas,sigmar):
    temp=padding(image,nk)
    w,h=temp.shape
    

    result= np.zeros ((w,h),np.float32)
    mk=int((nk-1)/2)

    for i in range(mk,h-mk):
      for j in range(mk,w-mk):
           reg=temp[i-mk:i+mk+1,j-mk:j+mk+1] #region of the image to be multiplied
           result[i,j]=process(reg,nk,sigmas,sigmar).astype(np.uint8)
           #exit (0)

    return crop(result,nk)

def method2 (image, c, nkernel):

    imp=padding(image,3)
    im3= conv2d (imp, np.array(kernels[nkernel-1]))
    im3=crop(im3,3)
    im4 = Scaling(im3)
    im5 = Adding(im4,c,image)
    im6 = Scaling(im5)
    return im6


def slicekernel(n,sigma):  #odd n
  m=int((n-1)/2) # 3-1/2 = 1, -1,0,1
                 # 5-1/2 = 2, -2,-1,0,1,2 
  v=n*[0]
  p=0
  for i in range(-m,m+1):
    v[p]=G(i,sigma)
    p=p+1
  return v
def method3 (input_image, sigma_row, sigma_col):
    h,w=input_image.shape

    hs=slicekernel(h,sigma_row)
    ws=slicekernel(w,sigma_col)

    hs=np.array(hs).reshape(h,1)
    ws=np.array(ws).reshape(1,w)

    ww=hs*ws
    R=  input_image*ww
    return Scaling(R)



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


