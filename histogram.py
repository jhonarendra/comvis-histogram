import cv2
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

img = cv2.imread('gr1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = cv2.normalize(hist, hist).flatten()

img2 = cv2.imread('gr2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist2 = cv2.normalize(hist2, hist2).flatten()

img3 = cv2.imread('gr3.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
hist3 = cv2.normalize(hist3, hist3).flatten()

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
fig.add_subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
fig.add_subplot(1, 3, 3)
plt.imshow(img3, cmap='gray')
plt.show()

fig2 = plt.figure(figsize=(20, 5))
fig2.add_subplot(1, 3, 1)
plt.hist(img.ravel(), 256, [0,256])
fig2.add_subplot(1, 3, 2)
plt.hist(img2.ravel(), 256, [0,256])
fig2.add_subplot(1, 3, 3)
plt.hist(img3.ravel(), 256, [0,256])
plt.show()

print('Citra 1')
print('Correlation\t: '+str(cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)))
print('Chi-Square\t: '+str(cv2.compareHist(hist, hist2, cv2.HISTCMP_CHISQR)))
print('Intersection\t: '+str(cv2.compareHist(hist, hist2, cv2.HISTCMP_INTERSECT)))
print('Bhattacharyya\t: '+str(cv2.compareHist(hist, hist2, cv2.HISTCMP_BHATTACHARYYA)))
print('')
print('Citra 2')
print('Correlation\t: '+str(cv2.compareHist(hist, hist3, cv2.HISTCMP_CORREL)))
print('Chi-Square\t: '+str(cv2.compareHist(hist, hist3, cv2.HISTCMP_CHISQR)))
print('Intersection\t: '+str(cv2.compareHist(hist, hist3, cv2.HISTCMP_INTERSECT)))
print('Bhattacharyya\t: '+str(cv2.compareHist(hist, hist3, cv2.HISTCMP_BHATTACHARYYA)))

# Putu Jhonarendra
# 1605551049