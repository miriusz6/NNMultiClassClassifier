from os import listdir
from os.path import isfile, join
import cv2

import numpy as np

def scatter_point(img, x, y, size):
    y_low_lim = max(0, y-size)
    y_high_lim = min(img.shape[0], y+size)
    x_low_lim = max(0, x-size)
    x_high_lim = min(img.shape[1], x+size)
    img[y_low_lim:y_high_lim, x_low_lim:x_high_lim] = 255
    return img


def get_files_in_dir(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def get_rand_rgb(min,max):
    return (np.random.randint(min, max), np.random.randint(min, max), np.random.randint(min, max))

def draw_random_circle(img_size, radius_range, thickness_range):    # Create a black image
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # Draw a circle
    center = (np.random.randint(0, img_size), np.random.randint(0, img_size))
    radius = np.random.randint(*radius_range)
    # random color
    color = get_rand_rgb(15, 250)
    thickness = np.random.randint(*thickness_range)
    img = cv2.circle(img, center, radius, color, thickness)
    return img

def draw_random_rectangle(img_size, thickness_range):
    # Create a black image
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # Draw a rectangle
    pt1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
    pt2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
    color = get_rand_rgb(15, 250)
    thickness = np.random.randint(*thickness_range)
    img = cv2.rectangle(img, pt1, pt2, color, thickness)
    return img

def draw_random_elipse(img_size, thickness_range):
    # Create a black image
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # Draw a rectangle
    pt1 = (np.random.randint(img_size//10, img_size//1.2), np.random.randint(img_size//10, img_size//1.2))
    pt2 = (np.random.randint(img_size//8,img_size//2), np.random.randint(img_size//8,img_size//2))

    rand_angle = np.random.randint(0, 360)
    color = get_rand_rgb(15, 250)
    thickness = np.random.randint(*thickness_range)
    img = cv2.ellipse(img, pt1, pt2, rand_angle, 0, 360, color, thickness)
    return img

def draw_random_line(img_size, thickness_range):
    # Create a black image
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # Draw a rectangle
    pt1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
    pt2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
    color = get_rand_rgb(15, 250)
    thickness = np.random.randint(*thickness_range)
    img = cv2.line(img, pt1, pt2, color, thickness)
    return img

def draw_random_triangle(img_size, thickness_range):
    # Create a black image
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # Draw a rectangle
    #50,70
    x1,y1 = np.random.randint(2, img_size), np.random.randint(2, img_size)

    x2_lim = min(x1+1, img_size)    
    
    x2,y2 = np.random.randint(0,x2_lim), np.random.randint(y1, img_size)

    x3,y3 = np.random.randint(x2, img_size), np.random.randint(y1, img_size)

    #img = scatter_point(img, x1, y1, 5)

    pts = np.array([[x1,y1], [x2,y2], [x3,y3]], np.int32)

    # calc area of the triangle
    area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2)
    if area < 600:
        return draw_random_triangle(img_size, thickness_range)
    #print(area)

    color = get_rand_rgb(15, 250)
    thickness = np.random.randint(*thickness_range)
    img = cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img



def draw_random_figure(img_size, thickness_range):
    fig = np.random.randint(0, 4)
    if fig == 0:
        return fig,draw_random_circle(img_size, (10, img_size//5), thickness_range)
    elif fig == 1:
        return fig,draw_random_rectangle(img_size, thickness_range)
    elif fig == 2:
        return fig,draw_random_elipse(img_size, thickness_range)
    elif fig == 3:
        return fig,draw_random_line(img_size, thickness_range)
    elif fig == 4:
        return fig,draw_random_triangle(img_size, thickness_range)


import matplotlib.pyplot as plt
# circle_img = draw_random_circle(100, (5, 30), (1, 10))
# plt.imshow(circle_img)
# plt.show()

# rectangle_img = draw_random_rectangle(100, (1, 5))
# plt.imshow(rectangle_img)
# plt.show()

# elipse_img = draw_random_elipse(100, (1, 5))
# plt.imshow(elipse_img)
# plt.show()

# line_img = draw_random_line(224, (1, 10))
# plt.imshow(line_img)
# plt.show()

# polygon_img = draw_random_triangle(224, (1, 8))
# plt.imshow(polygon_img)
# plt.show()