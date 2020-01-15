import cv2
import scipy.io.wavfile
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.fftpack import fft

rate,data=scipy.io.wavfile.read('project.wav')

#resize
scaledwav=cv2.resize(data,None,fx=1,fy=0.5)
scaledwav = np.asarray(scaledwav, dtype=np.int16)
scipy.io.wavfile.write('./music/resize05.wav',rate,scaledwav)
scaledwav=cv2.resize(data,None,fx=1,fy=1.5)
scaledwav = np.asarray(scaledwav, dtype=np.int16)
scipy.io.wavfile.write('./music/resize15.wav',rate,scaledwav)
#resize

#gaussian noise
std=np.std(data)
mean=np.mean(data)
noise = np.random.normal(mean*0.001,std,data.size)
noise=noise.reshape(int(data.size/2),2)
noised=np.add(data,noise)
noised = np.asarray(noised, dtype=np.int16)
scipy.io.wavfile.write('./music/noise.wav',rate,noised)
#gaussian noise

#gaussian blur
blurO = cv2.GaussianBlur(data, (55,55), 0)
scipy.io.wavfile.write('./music/blurO.wav',rate,blurO)
blur2 = cv2.GaussianBlur(noised, (105,105), 0)
scipy.io.wavfile.write('./music/blur2.wav',rate,blur2)
blur1 = cv2.GaussianBlur(noised, (55,55), 0)
scipy.io.wavfile.write('./music/blur1.wav',rate,blur1)
blur = cv2.GaussianBlur(noised, (25,25), 0)
scipy.io.wavfile.write('./music/blur.wav',rate,blur)
dur=data[:,0].size/44100
x=np.linspace(0,6,dur*44100) 
plt.subplot(621)
plt.plot(x,data[:,0])
plt.title('Original wave')
plt.subplot(622)
plt.plot(x,blurO[:,0])
plt.title('Blur Original wave')
plt.subplot(623)
plt.plot(x,noised[:,0])
plt.title('Noise wave')
plt.subplot(624)
plt.plot(x,blur[:,0])
plt.title('Blur Noise size small wave')
plt.subplot(625)
plt.plot(x,blur1[:,0])
plt.title('Blur Noise size mid wave')
plt.subplot(626)
plt.plot(x,blur2[:,0])
plt.title('Blur Noise size large wave')
#plt.show()
img = cv2.imread('openingtest.jpg')
testblur = cv2.GaussianBlur(img, (15,15), 0)
cv2.imwrite('gblur.jpg', testblur)
#gaussian blur

#opening,dilation,erosion
kernel = np.ones((3,3),np.int16)
erosion = cv2.erode(noised,kernel,iterations = 1)
scipy.io.wavfile.write('./music/erosion.wav',rate,erosion)
dilation = cv2.dilate(noised,kernel,iterations = 1)
scipy.io.wavfile.write('./music/dilation.wav',rate,dilation)
opening = cv2.morphologyEx(noised, cv2.MORPH_OPEN, kernel)
scipy.io.wavfile.write('./music/opening.wav',rate,opening)
kernel = np.ones((5,5),np.uint8)
img = cv2.imread('openingtest.jpg')
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('opening.jpg', opening)
#opening,dilation,erosion

#sharpen
rate,data=scipy.io.wavfile.read('project.wav')
kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
output = cv2.filter2D(data, -1, kernel_sharpen)
scipy.io.wavfile.write('./music/sharpen.wav',rate,output)
plt.subplot(211)
plt.plot(x,data[:,0])
plt.title('Original')
plt.subplot(212)
plt.plot(x,output[:,0])
plt.title('Sharpen')
#plt.show()
#sharpen

#contrast dark bright
contrast25 = cv2.addWeighted(data, 2.5, np.zeros(data.shape, data.dtype), 0, 0)
scipy.io.wavfile.write('./music/contrast25.wav',rate,contrast25)
contrast01 = cv2.addWeighted(data, 0.1, np.zeros(data.shape, data.dtype), 0, 0)
scipy.io.wavfile.write('./music/contrast01.wav',rate,contrast01)
bright=np.add(data,np.ones(data.shape, data.dtype)*5000)
scipy.io.wavfile.write('./music/bright.wav',rate,bright)
dark=np.add(data,np.ones(data.shape, data.dtype)*-5000)
scipy.io.wavfile.write('./music/dark.wav',rate,dark)
#contrast dark bright

#mix channel
easonrate,eason=scipy.io.wavfile.read('1channelmusic.wav')
mix=[]
a=eason.shape
for i in range(a[0]):
    mix.append(int((eason[i][0]+eason[i][1])/2))
mix=np.array(mix)
mix = np.asarray(mix, dtype=np.int16)
scipy.io.wavfile.write('./music/mix.wav',easonrate,mix)
dureason=eason[:,0].size/44100
x=np.linspace(0,12,dureason*44100) 
plt.subplot(311)
plt.plot(x,eason[:,0])
plt.title('Left')
plt.subplot(312)
plt.plot(x,eason[:,1])
plt.title('Right')
plt.subplot(313)
plt.plot(x,mix[:])
plt.title('Mix')
#plt.show()
#mix channel

# warping
rate,data=scipy.io.wavfile.read('project.wav')
output = np.zeros(data.shape, dtype=data.dtype)
rows, cols = data.shape
for i in range(rows):
    for j in range(cols):
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
        if i+offset_y < rows and j+offset_x < cols:
            output[i,j] = data[(i+offset_y)%rows,(j+offset_x)%cols]
        else:
            output[i,j] = 0
scipy.io.wavfile.write('./music/warping.wav',rate,output)
# warping