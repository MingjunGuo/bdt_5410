import cv2
import numpy as np
from copy import deepcopy


def myhoughcircle(input, radius, threshold):
    circles = []
    rows = input.shape[0]
    cols = input.shape[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi/180)
        cosang[angle] = np.cos(angle * np.pi/180)

    # initializing an empty 2D array with zeros
    acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

    # iterating through the original image
    for x in range(rows):
        for y in range(cols):
            if input[x][y] == 255: # edge
                # increment in the accumulator cells
                for angle in range(0, 360):
                    b = y - int(round(radius * sinang[angle], ndigits=0))
                    a = x - int(round(radius * cosang[angle], ndigits=0))
                    if a >= 0 and a < rows and b >= 0 and b < cols:
                        acc_cells[a][b] += 1

    print('for radius:', radius)
    acc_cells_max = np.amax(acc_cells)
    print('max_accumulator_value:', acc_cells_max)

    if acc_cells_max >= threshold:
        # print('Detecting the circles for radius:', radius)

        # initial threshold
        acc_cells[acc_cells < threshold] = 0

        # find the circles for this radius
        for i in range(rows):
            for j in range(cols):
                if i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >=threshold:
                    avg_sum = np.float32((acc_cells[i][j]
                                          +acc_cells[i-1][j]+acc_cells[i+1][j]
                                          +acc_cells[i][j-1]+acc_cells[i][j+1]
                                          +acc_cells[i-1][j-1]+acc_cells[i-1][j+1]
                                          +acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9)
                    if avg_sum > 8.6:
                        print('for radius:', radius, 'coordinate:', [j, i], 'average:', avg_sum)
                        circles.append((i, j, radius))
                        acc_cells[i:i+7, j:j+7] = 0
    return circles


def task1_main():
    img_path = 'qiqiu.png'
    # reading the input image and converting to gray scale
    input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('input', input)

    # create copy of the original image
    input = deepcopy(input_image)

    # steps
    # 1. Denoise using Gaussian filter
    smoothed_img = cv2.GaussianBlur(input, (5, 5), 0)
    #cv2.imshow('smoothed_img', smoothed_img)

    # 2. Detect Edges using Canny Edge Detector
    edged_img = cv2.Canny(smoothed_img, 0, 75)
   # cv2.imshow('edged_img', edged_img)

    # 3. Detect circles
    radius = 114
    threshold = 76
    circles = myhoughcircle(edged_img,radius,threshold)
    print(circles)

    # 4. Print the output
    for vertex in circles:
        cv2.circle(input_image, (vertex[1], vertex[0]), vertex[2], (0, 255, 0), 1)

    # cv2.imshow('Circle Detected Image:', input_image)

    # 5. Print the output in the empty array
    output_image = np.zeros((input.shape[0], input.shape[1]), np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    for vertex in circles:
        cv2.circle(output_image, (vertex[1], vertex[0]), vertex[2], (255, 255, 255), 1)
    # cv2.imshow('output_image', output_image)
    cv2.imwrite('output_image.jpg', output_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def myfld(input_sample, class1_sample, class2_sample):
    # compute the mean_value for each class sample
    m1 = len(class1_sample)
    m2 = len(class2_sample)

    mean_c1 = np.sum(class1_sample, axis=0)/m1
    mean_c2 = np.sum(class2_sample, axis=0)/m2

    # compute the in-class variance
    s1 = 0
    s2 = 0
    for i in range(m1):
        s1 = s1 + np.transpose([class1_sample[i]-mean_c1]) * (class1_sample[i]-mean_c1)
    for j in range(m2):
        s2 = s2 + np.transpose([class2_sample[j]-mean_c2]) * (class2_sample[j]-mean_c2)
    sw = s1 + s2

    # compute the weight and normalize
    sw_I = np.linalg.inv(sw)
    w = np.dot(sw_I, np.transpose([mean_c1-mean_c2]))
    w1 = w[0] / np.sqrt(w[0] * w[0] + w[1] * w[1])
    w2 = w[1] / np.sqrt(w[0] * w[0] + w[1] * w[1])
    w = [float(w1), float(w2)]

    # compute the separation point
    sep_point = np.sum(w * (mean_c1+mean_c2) * 0.5)

    # compute the class
    output_class = (np.sum(np.array(input_sample) * np.array(w)) < sep_point) + 1
    return mean_c1, mean_c2, sw, w, output_class


def task2_main():
    input_sample = [2, 5]
    class1_sample = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
    class2_sample = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])
    mean_c1, mean_c2, sw, w, output_class = myfld(input_sample, class1_sample, class2_sample)
    print('a):the predicted class number:', output_class)
    print('b):the within-class scatter matrix:', sw)
    print('c):the weight vector:', w)


if __name__ == '__main__':
    task1_main()
    task2_main()
