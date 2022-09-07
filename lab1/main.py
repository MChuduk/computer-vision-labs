import cv2 
import matplotlib.pyplot as plt
import numpy as np

def task1():
    image = cv2.imread('1.jpg');
    cv2.imshow('image', image);
    cv2.waitKey(0);
    return image;

def task2():
    image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE);
    cv2.imshow('image', image);
    cv2.waitKey(0);
    return image;

def task3():
    image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE);
    thresh = 128;
    image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1];
    cv2.imshow('image', image);
    cv2.waitKey(0);
    return image;

def task4(filename, image):
    cv2.imwrite(filename, image);

def task5():
    image = cv2.imread('2.jpg');
    show_hists(image, "hist_before.png");
    R, G, B = cv2.split(image)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    image = cv2.merge((output1_R, output1_G, output1_B))
    cv2.imshow("image", image);
    cv2.waitKey(0);
    show_hists(image, "hist_after.png");

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

task5();