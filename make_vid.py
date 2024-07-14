import cv2
import numpy as np

img=[]

for i in range(0,50):
    img.append(cv2.imread(f'renderings/humans2/run12/1stable{i}.png'))


height,width,layers=img[1].shape

video=cv2.VideoWriter('videos/video1.avi',-1,1,(width,height))

for j in range(0,5):
    video.write(img[j])

cv2.destroyAllWindows()
video.release()