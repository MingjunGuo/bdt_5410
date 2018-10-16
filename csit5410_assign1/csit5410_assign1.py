# coding: utf-8

import numpy as np
import cv2
from PIL import Image, ImageDraw
import skimage
from skimage import io, draw
from skimage import transform
import sys


def threshold(image, T):
    """
    generating the binary output image according to the threshold T
    :param image: image with different values in the response map(obtained by applying prewitt edge detection)
    :param T: the threshold
    :return:
    """
    final_image = np.zeros(image.shape)

    if T is None:
        T = (np.max(image) + np.min(image)) * 0.5
        for i in range(40):
            G1 = []
            G2 = []
            for m in range(image.shape[0]):
                for n in range(image.shape[1]):
                    if image[m, n] >= T:
                        G1.append(image[m, n])
                    else:
                        G2.append(image[m, n])
            m1 = np.average(G1)
            m2 = np.average(G2)
            T = (m1 + m2) * 0.5
    for m in range(image.shape[0]):
        for n in range(image.shape[1]):
            if image[m, n] > T:
                final_image[m, n] = 1
            else:
                final_image[m, n] = 0
    return final_image


def myprewittoperator(roi, direction):
    """
    computes the binary edge image from the 3*3 roi
    :param roi: 3*3 Neighborhood of one pixel
    :param T: the threshold
    :param direction: 'horizontal'|'vertical'|'pos45'|'neg45'|'all'
    :return: the value of the pixel
    """
    prewitt_operator =[]
    prewitt_operator.append(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_operator.append(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_operator.append(np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]))
    prewitt_operator.append(np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]))
    if direction == 'horizontal':
        result = np.abs(np.sum(roi * prewitt_operator[0]))
    elif direction == 'vertical':
        result = np.abs(np.sum(roi * prewitt_operator[1]))
    elif direction == 'pos45':
        result = np.abs(np.sum(roi * prewitt_operator[2]))
    elif direction == 'neg45':
        result = np.abs(np.sum(roi * prewitt_operator[3]))
    elif direction == 'all':
        result = np.abs(np.sum(roi * prewitt_operator[0]))
        for i in range(1, 4):
            temp = np.abs(np.sum(roi * prewitt_operator[i]))
            if temp > result:
                result = temp
    else:
        print('type error')
    return result


def myprewittedge(Im, T, direction):
    """
    computes the binary edge image from the input image Im
    :param Im: an Intensity gray scale image, Im.shape:array[m,n]|Im.type:double-precision floating point
    :param T: the threshold for generating the binary output image
    :param direction: A string for specifying whether to look for 'horizontal'|'vertical',etc.
    :return: Image contains edges at those points where the absolute filter response is above or equal to the threshold
    T, shape:array[m,n]
    """
    new_image = np.zeros(Im.shape)
    Im_expan = cv2.copyMakeBorder(Im, 1, 1, 1, 1, cv2.BORDER_DEFAULT)  # boundary expansion
    for i in range(1, Im_expan.shape[0]-1):
        for j in range(1, Im_expan.shape[1]-1):
            new_image[i-1, j-1] = myprewittoperator(Im_expan[i-1:i+2, j-1:j+2], direction)
    final_image = threshold(new_image, T)
    final_image = skimage.img_as_ubyte(final_image, force_copy=False)
    return final_image


def mydistance(bp, ep):
    """
    compute the distance between two point
    :param bp: the first point
    :param ep: the second point
    :return: the distance
    """
    distance = np.sqrt(np.square(bp[0]-ep[0]) + np.square(bp[1]-ep[1]))
    return distance


def mylineextraction(image):
    """
    extract the logest line segment from the binary image
    :param image: binary image
    :return: bp(beginning point)and ep(ending point),with format tuple(x,y) for each point of the longest line segment found
    in image based on Hough transform
    """
    lines = transform.probabilistic_hough_line(image, threshold=30, line_length=350, line_gap=12)
    bp = lines[0][0]
    ep = lines[0][1]
    distance = mydistance(bp, ep)
    for i in range(1, len(lines)):
        temp_bp = lines[i][0]
        temp_ep = lines[i][1]
        temp_distance = mydistance(temp_bp, temp_ep)
        if temp_distance > distance:
            distance = temp_distance
            bp = temp_bp
            ep = temp_ep
    return bp, ep


def mysiftalignment(img1, img2):
    """
    using SIFT to do Image alignment
    :param I1: the first image
    :param I2: the second image
    :return: the image after matching
    """
    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.62 * n.distance]

    # cv2.drawMatchedKnn expects lists as matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return img3


def csit5410_assign1(FILENAME):
    """
    Main routine function for image edge detection and for image alignment
    :param FILENAME:
    :return:
    """
    # Task1: read and save the image.
    Im = io.imread(FILENAME, as_grey=True)
    Im = skimage.img_as_float(Im, force_copy=False)
    if Im is None:
        print('fail to load image')
    else:
        # io.imshow(Im)
        # io.show()
        io.imsave('01original.jpg', Im)
        print('Original image is read and displayed successfully.')

    # Task2:Generate the corresponding binary edge image of the given image Im with the threshold T.
        T = np.max(Im) * 0.2
        direction = 'all'
        g = myprewittedge(Im, T, direction)
        # io.imshow(g)
        # io.show()
        io.imsave('02binary1.png', g)
        print('The corresponding binary edge image is computed and displayed successfully.')

    # Task3:Generate the corresponding binary edge image of the given image Im without the specified threshold.
        direction = 'all'
        f = myprewittedge(Im, None, direction)
        # io.imshow(f)
        # io.show()
        io.imsave('03binary2.png', f)
        print('The corresponding binary edge image is computed and displayed successfully.')

    # Task4:find the longest line segment based on Hough transform
        image = np.array(Image.open('03binary2.png'))
        bp, ep = mylineextraction(image)
        Im_2 = Image.open('fig.tif')
        Im_2 = Im_2.convert('RGB')
        draw = ImageDraw.Draw(Im_2)
        draw.line([bp, ep], fill='blue', width=2)
        draw.point([bp, ep], fill='red')
        # Im_2.show()
        Im_2.save('04longestline.jpg')

    # Task5：Image alignment using sift
    image1 = io.imread('image1.png')
    image2 = io.imread('image2.png')
    image3 = io.imread('image3.png')
    code = io.imread('QR-Code.png')
    code_image1 = mysiftalignment(code, image1)
    code_image2 = mysiftalignment(code, image2)
    code_image3 = mysiftalignment(code, image3)
    io.imsave('05QR_img1.png', code_image1)
    io.imsave('06QR_img2.png', code_image2)
    io.imsave('07QR_img3.png', code_image3)


if __name__ == '__main__':
    csit5410_assign1('fig.tif')