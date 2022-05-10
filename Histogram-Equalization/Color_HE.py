import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plot_grid_size = (4, 2)
plt.figure(figsize=(15,15))

def plot_hist_and_cdf(index, img, hist, cdf, title):
    plt.subplot(plot_grid_size[0],plot_grid_size[1], index)
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256]), plt.title(title) 
    plt.legend(('cdf','histogram'), loc = 'upper left')

def plot_img(index, img, title):
    plt.subplot(plot_grid_size[0],plot_grid_size[1], index)
    plt.imshow(img)
    plt.axis('off'), plt.title(title) 
    
img = cv.imread('sun1.jpeg',cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_HSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)

h, s, v = cv.split(img_HSV)
hist,bins = np.histogram(v.flatten(), 256, [0,256])
cdf = hist.cumsum()

plot_img(1, img, "original")
plot_hist_and_cdf(2, v, hist, cdf, "original")

#HE
v_equalize = cv.equalizeHist(v)
equalize_hsv = cv.merge([h,s,v_equalize])
hsv2rgb = cv.cvtColor(equalize_hsv, cv.COLOR_HSV2RGB)

hist2,bins2 = np.histogram(v_equalize.flatten(), 256, [0,256])
cdf2 = hist2.cumsum()

plot_img(3, hsv2rgb, "cv.HE")
plot_hist_and_cdf(4, v_equalize, hist2, cdf2, "cv.HE")


# AHE
ahe = cv.createCLAHE(clipLimit=2555, tileGridSize=(8,8))
v_ahe = ahe.apply(v)
ahe_hsv = cv.merge([h,s,v_ahe])
ahe_hsv2rgb = cv.cvtColor(ahe_hsv, cv.COLOR_HSV2RGB)

hist3,bins3 = np.histogram(v_ahe.flatten(), 256, [0,256])
cdf3 = hist3.cumsum()

plot_img(5, ahe_hsv2rgb, "AHE")
plot_hist_and_cdf(6, v_ahe, hist3, cdf3, "AHE")

# CLAHE
clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
v_clahe = clahe.apply(v)
clahe_hsv = cv.merge([h,s,v_clahe])
clahe_hsv2rgb = cv.cvtColor(clahe_hsv, cv.COLOR_HSV2RGB)

hist4,bins4 = np.histogram(v_clahe.flatten(), 256, [0,256])
cdf4 = hist4.cumsum()

plot_img(7, clahe_hsv2rgb, "CLAHE")
plot_hist_and_cdf(8, v_clahe, hist4, cdf4, "CLAHE")

plt.show()
