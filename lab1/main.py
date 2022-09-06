import cv2 
import matplotlib.pyplot as plt
import numpy as np

def get_channel(image, channel):
    return image[:,:, channel]

def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compute_hists(image):
    hist_b = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    hist_g = cv2.calcHist([image], channels=[1], mask=None, histSize=[256], ranges=[0, 256])
    hist_r = cv2.calcHist([image], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
    return (hist_b, hist_g, hist_r)

def show_hists(rgb_image, name="hist.png"):
    hb, hg, hr = compute_hists(rgb_image)
    fig = plt.figure()
    plt.plot(hb, color="b")
    plt.plot(hg, color="g")
    plt.plot(hr, color="r")
    plt.savefig(name)
    plot = cv2.imread(name)
    cv2.imshow(name, plot)
    cv2.waitKey(0)
    
image = cv2.imread("1.jpg")

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image[:, :, 0] = 0
# image[:, :, 2] = 0

# hb, hg, hr = compute_hists(image)

        
# fig = plt.figure()
# plt.plot(hb, color="b")
# plt.plot(hg, color="g")
# plt.plot(hr, color="r")
# plt.savefig("plot.png")

# plot = cv2.imread("plot.png")

# cv2.imshow("Plot", plot)
# cv2.waitKey(0)

# cv2.imshow("image", image)
# cv2.waitKey(0)


# R, G, B = cv2.split(image)
# output1_R = cv2.equalizeHist(R)
# output1_G = cv2.equalizeHist(G)
# output1_B = cv2.equalizeHist(B)
# image = cv2.merge((output1_R, output1_G, output1_B))


# hb, hg, hr = compute_hists(image)

# fig = plt.figure()
# plt.plot(hb, color="b")
# plt.plot(hg, color="g")
# plt.plot(hr, color="r")
# plt.savefig("plot_equalized.png")


# hist_equalized = cv2.imread("plot_equalized.png")

# cv2.imshow("hist_equalized", hist_equalized)
# cv2.waitKey(0)

# cv2.imshow("equalize", image)
# cv2.waitKey(0)
