import os, sys, linecache
import cv2
import numpy as np
from PIL import Image, ImageDraw
from math import sqrt, atan2, pi, cos, sin
from IPython.display import Image as pic
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from IPython.display import display
from skimage.measure import compare_ssim
from skimage import io
import argparse
import imutils
import imageio
import time
import pandas as pd
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import os.path as osp
import matplotlib.image as mpimg
import pytesseract
#
from django.core.files import File as FileWrapper
from django.core.files.storage import FileSystemStorage
import simplejson as json

class Tracker :
    def __init__(self, given_path, grit = 10) :
        self.given_path = 'F:\django-eye-tracker\mysite\media\%s' % (given_path)
        self.grit = grit

    #finds 'gradient' of change between pictures. This is represented as an array where areas without changes are black
    #and areas with change are colored
    def get_page_diffs(self, video_name, pages_array):
        start = time.time()
        cam = cv2.VideoCapture(video_name)
        fr = cam.get(cv2.CAP_PROP_FPS)
        anchors = []
        #counter and exf are used to make sure every single frame isn't polled. One second is about 10 frames and
        #polling every individual frame runs the risk of including the white space from when the image and text
        #are still loading.
        counter = 0
        exf = self.grit
        #page index keeps track of the position in the anchors array.
        page_index = 1
        old_index = 100000000
        current_frame = 0
        np_prev = []
        #search_for_last indicates whether or not the program is looking for the final page.
        search_for_last = False
        page_turned = False
        #Coordinates of the picture in each image.
        x = 248
        y = 112
        w = 351
        h = 264
        time_scaled = []
        #A buffer is the number of frames before and after a switch is detected in order to avoid polling
        #the white space from the delay of the image loading into the gradient
        buffer = 10
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == False:
                break
            #Condition for the first page 
            elif current_frame == pages_array[0] + buffer:
                diffs = np.zeros_like(np.array(frame))
                np_prev = np.array(frame).astype(float)
                old_index = pages_array[0] + buffer
                counter = 0
            #resets the gradient in response to the next page
            elif page_turned and old_index < current_frame and counter == exf:
                diffs = np.zeros_like(np.array(frame))
                np_prev = np.array(frame).astype(float)
                counter = 0
                page_turned = False
            #Aggregating the change in gradient
            elif old_index < current_frame and current_frame < pages_array[page_index] and counter == exf and not search_for_last:
                np_tar = np.array(frame).astype(float)
                diffs = np.absolute(np_tar - np_prev) + diffs
                diffs = diffs.astype(float)
                np_prev = np_tar
                counter = 0
            #Adds the gradient array gathered above to the time_scaled array
            elif counter == exf and old_index < current_frame and not search_for_last:
                counter = 0
                oi = old_index
                old_index = pages_array[page_index] + buffer + 10
                if page_index + 1 == len(pages_array):
                    print(current_frame)
                    search_for_last = True
                    page_turned = True
                else:
                    page_index += 1
                    crop = self.convert_cv2_to_PIL(frame).crop((x,y , x+w, y+h))
                    crop = self.convert_PIL_to_cv2(crop)
                img_cv = cv2.resize(diffs, (854, 504))
                #img_cv = img_cv * 255
                (thresh, img_cv) = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
                time_scaled.append([img_cv, current_frame])
                cv2.imwrite(os.path.join(self.path2, str(oi) + 'diffframe' + str(current_frame) + '.jpg'), img_cv)
                page_turned = True
            #condition to search for last page.
            elif search_for_last and counter == exf and page_turned:
                diffs = np.zeros_like(np.array(frame))
                np_prev = np.array(frame).astype(float)
                counter = 0
                ag = np.sum(self.compute_avg_shade(x + 50, y + 50, 20, self.convert_cv2_to_PIL(frame)))
                print(ag)
                if ag < 700:
                    page_turned = False
                    crop = self.convert_cv2_to_PIL(frame).crop((x,y , x+w, y+h))
                    crop = self.convert_PIL_to_cv2(crop)
                print(current_frame)
            #base case: going through each frame and aggregating the differences between them.
            elif search_for_last and counter == exf and page_turned == False:
                gray1 = cv2.cvtColor(np.float32(crop), cv2.COLOR_BGR2GRAY)
                crop2 = self.convert_cv2_to_PIL(frame).crop((x,y , x+w, y+h))
                crop_cv = self.convert_PIL_to_cv2(crop2)
                gray2 = cv2.cvtColor(np.float32(crop_cv), cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(gray1, gray2, full = True)
                print(current_frame, "aks at ", score)
                if score < .5:
                    print(current_frame, " breaks at ", score)
                    img_cv = cv2.resize(diffs, (854, 504))
                    (thresh, img_cv) = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
                    time_scaled.append([img_cv, current_frame])
                    cv2.imwrite(os.path.join(self.path2, 'diffframe' + str(current_frame) + '.jpg'), img_cv)
                    pages_array.append(current_frame)
                    break
                np_tar = np.array(frame).astype(float)
                diffs = np.absolute(np_tar - np_prev) + diffs
                diffs = diffs.astype(float)
                np_prev = np_tar
                counter = 0
                crop = crop2
            elif counter == exf:
                counter = 0
                
            current_frame += 1
            counter += 1
        
        cam.release()
        cv2.destroyAllWindows
        
        end = time.time()
        
        print(end - start)
        return time_scaled

    # cv2 uses bgr format vs PIL which uses rbg. These are functions meant to handle the conversation between formats used throughout the code
    def convert_PIL_to_cv2(self, PIL_image):
        np_array = np.array(PIL_image)
        return np_array[:, :, ::-1].copy()

    def convert_cv2_to_PIL(self, cv2_image):
        cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image)

    # converts the image to grayscale
    def compute_grayscale(self, input_pixels, width, height):
        grayscale = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                pixel = input_pixels[x, y]
                grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
        return grayscale

    #applies a gaussian filter to the image to make the edges stand out more
    def compute_blur(self, input_pixels, width, height):
        clip = lambda x, l, u: l if x < l else u if x > u else x

        kernel = np.array([
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
        ])

        offset = len(kernel) // 2

        blurred = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                acc = 0
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = clip(x + a - offset, 0, width - 1)
                        yn = clip(y + b - offset, 0, height - 1)
                        acc += input_pixels[xn, yn] * kernel[a, b]
                blurred[x, y] = int(acc)
        return blurred

    # Searches for edges based on differences between intensities
    def compute_gradient(self, input_pixels, width, height):
        gradient = np.zeros((width, height))
        direction = np.zeros((width, height))
        for x in range(width):
            for y in range(height):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                    magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                    gradient[x, y] = sqrt(magx**2 + magy**2)
                    direction[x, y] = atan2(magy, magx)
        return gradient, direction

    #Detects edges based on curvature
    def filter_out_non_maximum(self, gradient, direction, width, height):
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
                rangle = round(angle / (pi / 4))
                mag = gradient[x, y]
                #print(x, y)
                if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                        or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                        or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                        or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                    gradient[x, y] = 0
                    
    #filters out edges that don't meet a defined parameter              
    def filter_strong_edges(self, gradient, width, height, low, high):
        keep = set()
        for x in range(width):
            for y in range(height):
                if gradient[x, y] > high:
                # print("high passed ", gradient[x, y])
                    keep.add((x, y))
                    
        lastiter = keep
        while lastiter:
            newkeep = set()
            for x, y in lastiter:
                for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                        #print("low passed " , gradient[x, y])
                        newkeep.add((x+a, y+b))
            keep.update(newkeep)
            lastiter = newkeep

        return list(keep)

    #Averages and returns the shade of a x + d by y + d image    
    def compute_avg_shade(self, x, y, d, input_image):
        crop = input_image.crop((x - d, y - d, x + d, y + d))
        np_crop = np.array(crop)
        average = np_crop.mean(axis=0).mean(axis=0)
        avg_patch = np.ones(shape=np_crop.shape, dtype=np.uint8)*np.uint8(average)
        
        return avg_patch[0][0]

    # Loads the image and runs it through the previously defined functions
    def canny_edge_detector(self, input_image, low_edge, high_edge):
        input_pixels = input_image.load()
        width = input_image.width
        height = input_image.height

        blurred = self.compute_blur(input_pixels, width, height)

        gradient, direction = self.compute_gradient(blurred, width, height)

        self.filter_out_non_maximum(gradient, direction, width, height)
        
        keep = self.filter_strong_edges(gradient, width, height, low_edge, high_edge)
        
        return keep

    # Parses the circles found by the canny edge detector and filters them out by comparing them to a weighted average of the circle shade
    def identify_circles(self, rmin, rmax, steps, threshold, image_name, mask, tolerance, low_edge, high_edge):
        points = []
        input_image = image_name
        keep = self.canny_edge_detector(mask, low_edge, high_edge)
        
        for r in range(rmin, rmax+1):
            for t in range(steps):
                points.append((r, int(r * cos(2*pi * t/ steps)), int(r*sin(2*pi*t/steps))))
                
        acc = defaultdict(int)
        
        for x, y in keep:
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1
        
        circles = []
        other = []
        not_circles = []
        not_others = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                circle_average = self.compute_avg_shade(x, y, r * 3 / 4, image_name)
                patch = self.compute_avg_shade(x, y, r * 3 /4, mask)
                diff = circle_average - self.average_color2
                diff_abs = np.absolute(diff)
                diff_sum = sum(diff_abs)
                prob = np.dot([circle_average[0], circle_average[1], circle_average[2], r], self.l_weights)
                if prob > 12 and diff_sum < tolerance and patch < 20:
                    circles.append((x, y, r))
                    other.append((prob, diff_sum, patch))
                else:
                    not_circles.append((x, y, r))
                    not_others.append((prob, diff_sum, patch))

        return circles, other, not_circles, not_others

    #Crops an area of the image where there is change to speed up the process. This way, instead of looking at the whole frame, we only have to look at a small part/parts of it.
    def crop_circles2(self, crop_edges, n, rmin, rmax, steps, threshold, image_name, mask, tolerance, low_edge, high_edge):
        crop = image_name.crop((crop_edges[n][0], crop_edges[n][1], crop_edges[n][2] + crop_edges[n][0], crop_edges[n][3] + crop_edges[n][1]))
        crop2 = mask.crop((crop_edges[n][0], crop_edges[n][1], crop_edges[n][2] + crop_edges[n][0], crop_edges[n][3] + crop_edges[n][1]))
        circles, other_info, not_circles, not_info = self.identify_circles(rmin, rmax, steps, threshold, crop, crop2, tolerance, low_edge, high_edge)
        circles = np.asarray(circles)
        not_circles = np.asarray(not_circles)
        other_info = np.asarray(other_info)
        not_info = np.asarray(not_info)
        adjusted_circles = []
        others = []
        adjusted_not_circles = []
        not_others = []
        for k in range(len(circles)):
            adjusted_circles.append((crop_edges[n][0] + circles[k][0], crop_edges[n][1] + circles[k][1], circles[k][2]))
            
        for k in range(len(not_circles)):
            adjusted_not_circles.append((crop_edges[n][0] + not_circles[k][0], crop_edges[n][1] + not_circles[k][1], not_circles[k][2]))
            
        #print("crop circles:" , np.array(adjusted_circles).shape, np.array(other_info).shape, np.array(adjusted_not_circles).shape, np.array(not_info).shape)
        return adjusted_circles, other_info, adjusted_not_circles, not_info

    #Helper function to round to array of longs to ints
    def round_array(self, array, num):
        for k in range(len(array)):
            array[k] = [math.ceil(array[k][0] / num) * num, math.ceil(array[k][1] / num) * num, array[k][2]]
            
        return array

    #Prevents a circle from being picked up multiple times
    def spread_clusters(self, circles, info):
        no_copies = np.empty((0, 3))
        no_copies = np.vstack([no_copies, circles[0]])
        no_copies_info = np.empty((0, 3))
        no_copies_info = np.vstack([no_copies_info, info[0]])
        for k in range(len(circles)):
            unique = True
            for q in range(len(no_copies)):
                if circles[k][0] == no_copies[q][0] and circles[k][1] == no_copies[q][1]:
                    unique = False
            
            if unique:
                no_copies = np.vstack([no_copies, circles[k]])
                no_copies_info = np.vstack([no_copies_info, info[k]])
                    
        return no_copies, no_copies_info

    #Puts together arrays of circles/not circles
    def aggregate_arrays(self, not_circles, adjusted_not_circles, not_info, crop_edges, n):
        adjusted_not_circles = np.empty((0,3))
        not_others = np.empty((0, 3))
        for k in range(len(not_circles)):
            not_circles[k] = [crop_edges[n][0] + not_circles[k][0], crop_edges[n][1] + not_circles[k][1], not_circles[k][2]]
            if len(adjusted_not_circles) > 0:
                overlap = False
                for q in range(len(adjusted_not_circles)):
                    center_dist = np.sqrt(np.square(adjusted_not_circles[q][0] - not_circles[k][0]) + np.square(adjusted_not_circles[q][1]-not_circles[k][1]))
                    #minimum distance the center of the circles must be in order to be added
                    spacing = 7
                    if center_dist < spacing:
                        overlap = True
                        break
                if not overlap:
                    adjusted_not_circles = np.vstack([adjusted_not_circles, not_circles[k]])
                    not_others = np.vstack([not_others, not_info[k]])
                    
            else:
                adjusted_not_circles = np.vstack([adjusted_not_circles, not_circles[k]])
                not_others = np.vstack([not_others, not_info[k]])
                
        return adjusted_not_circles, not_others
        
    #crops the image to a smaller area around the detected circle so that empty space is not processed
    def crop_circles(self, crop_edges, n, rmin, rmax, steps, threshold, image_name, mask, tolerance, low_edge, high_edge):
        crop = image_name.crop((crop_edges[n][0], crop_edges[n][1], crop_edges[n][2] + crop_edges[n][0], crop_edges[n][3] + crop_edges[n][1]))
        crop2 = mask.crop((crop_edges[n][0], crop_edges[n][1], crop_edges[n][2] + crop_edges[n][0], crop_edges[n][3] + crop_edges[n][1]))

        # print('\n\n\n rmin : %s \n\n rmax : %s \n\n steps : %s \n\n threshold : %s \n\n crop : %s \n\n crop2 : %s \n\n tolerance: %s \n\n low_edge : %s \n\n high_edge : %s \n\n\n' % ((rmin, rmax, steps, threshold, crop, crop2, tolerance, low_edge, high_edge)))
        # print(crop)
        # print('\n\n\n')
        # print(crop2)
        # print('\n\n\n')

        # preent = self.identify_circles(rmin, rmax, steps, threshold, crop, crop2, tolerance, low_edge, high_edge)

        # print('\n\n\n')
        # print(preent)
        # print('\n\n\n')

        circles, other_info, not_circles, not_info = self.identify_circles(rmin, rmax, steps, threshold, crop, crop2, tolerance, low_edge, high_edge)
        circles = np.asarray(circles)
        not_circles = np.asarray(not_circles)
        other_info = np.asarray(other_info)
        not_info = np.asarray(not_info)
        circles = self.round_array(circles, 7)
        not_circles = self.round_array(not_circles, 7)
        adjusted_circles = np.empty((0, 3))
        others_info=np.empty((0, 3))
        adjusted_not_circles = np.empty((0,3))
        not_others = np.empty((0, 3))
        if not_circles.shape[0] > 0:
            not_circles, not_info = self.spread_clusters(not_circles, not_info)
            
        if circles.shape[0] > 0:
            circles, other_info = self.spread_clusters(circles, other_info)
        
        adjusted_circles, other_info = self.aggregate_arrays(circles, adjusted_circles, other_info, crop_edges, n)
            
        adjusted_not_circles, not_others = self.aggregate_arrays(not_circles, adjusted_not_circles, not_info, crop_edges, n)
            
        #print("crop circles:" , np.array(adjusted_circles).shape, np.array(other_info).shape, np.array(adjusted_not_circles).shape, np.array(not_info).shape)
        return adjusted_circles, other_info, adjusted_not_circles, not_others

    # Method for testing. Captures a frame and the frame before and after it with a set buffer
    def get_frame(self, target_frame, video_name, buffer):
        cam = cv2.VideoCapture(video_name)
        current_frame = 0
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == False:
                cam.release()
                cv2.destroyAllWindows
                break
            elif current_frame == target_frame + buffer + 1:
                cam.release()
                cv2.destroyAllWindows
                break
            elif current_frame == target_frame - buffer:
                name = 'previous_frame.jpg'
                cv2.imwrite(name, frame)
                image1_name = name
            elif current_frame == target_frame:
                name = 'target_frame.jpg'
                cv2.imwrite(name, frame)
                image2_name = name
            elif current_frame == target_frame + buffer:
                name = 'next_frame.jpg'
                cv2.imwrite(name, frame)
                image3_name = name
            
            current_frame += 1

    #Formats the array returned by find_circle
    def flatten_array(self, circle_array):
        original_array = circle_array
        if (circle_array.shape[0] == 1 and len(circle_array.shape) >= 2):
            return original_array[0]
        elif len(circle_array.shape) == 2:
            return original_array
        elif circle_array.shape[0] >= 2:
            np_shape = np.array(circle_array)
            np_shape = np_shape.reshape(1, -1)
            np_shape = np_shape.flatten('K')
            np_shape2 = np.empty((0, 3))
            try:
                for k in range(len(np_shape)):
                    np_shape2 = np.vstack([np_shape2, np_shape[k]])
            except:
                return original_array  
            return np_shape2
        
        return original_array

    #Checks if a circle has been recently added to the array of circles. This is to avoid a circle being picked up multiple times when it appears across multiple frames.
    def check_recent(self, circle_array, circles, d):
        stamps = []
        inner_stamps = []
        for k in range(len(circles)):
            for j in range(len(circle_array)):
                diffs = np.subtract(circles[k], circle_array[j])
                diff_abs = np.abs(diffs)
                diff_sum = np.sum(diff_abs)
                if diff_sum < d:
                    inner_stamps.append(j)
            try:
                stamps.append(np.amax(inner_stamps))
                inner_stamps = []
            except: 
                stamps.append(-1)
        return stamps

    def find_circles(self, rmin, rmax, steps, threshold, image_name, mask, tolerance, low_edge, high_edge):
        points = []
        input_image = image_name
        output_image = Image.new("RGB", input_image.size)
        draw = ImageDraw.Draw(output_image)
        
        anchor_image = cv2.imread('previous_frame.jpg')
        cv2_image = self.convert_PIL_to_cv2(input_image)

        gray1 = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

        crop_edges = []
        adjusted_circles = np.empty((0, 3))
        others_array = np.empty((0, 3))
        not_circles = np.empty((0, 3))
        not_others_array = np.empty((0, 3))

        (score, diff) = compare_ssim(gray1, gray2, full = True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w * h) > 500:
                crop_edges.append((x - 10, y - 10, w + 10, h + 10))
                    
        for n in range(len(crop_edges)):
            circle_array, others, not_circle_array, not_others = self.crop_circles(crop_edges, n, rmin, rmax, steps, threshold, image_name, mask, tolerance, low_edge, high_edge)
            circle_array = np.array(circle_array)
            not_circle_array = np.array(not_circle_array)
            others = np.array(others)
            not_others = np.array(not_others)
            try:
                adjusted_circles = np.vstack((adjusted_circles, circle_array))
                others_array = np.vstack((others_array, others))
            except:
                pass
                
            try:
                not_circles = np.vstack((not_circles, not_circle_array))
                not_others_array = np.vstack((not_others_array, not_others))
            except:
                continue

        return adjusted_circles, others_array, not_circles, not_others_array

    #the path to the videos
    #path2 = r'C:\Users\ccui9\Testfolder'
    fs = FileSystemStorage()
    path2 = '/media/tests'

    #code by Professor Bishop
    def readFrames(self, vid):
        vc = cv2.VideoCapture(vid)
        for i in range(int(vc.get(cv2.CAP_PROP_FRAME_COUNT))):
            rval, im = vc.read()
            if not rval:
                break
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            yield im
            
    def upperRight(self, frames):
        for frame in frames:
            yield frame[60:100, -50:-15]
            #for non-transparent circle videos
            #yield frame[43:60, -45:-25]
            
    def roundTo(self, x, y):
        return x / y * y

    def show(self, f):
        plt.imshow(f, cmap='gray', vmin=0, vmax=255)
        
    def i2s(self, im):
        return pytesseract.image_to_string(im, lang="eng", config='--psm 10 digits --oem 3 -c tessedit_char_whitelist=0123456789')

    def find_uFrames(self, uniqueFrames, pFrames):
        array = []
        x = 0
        for k in range(len(uniqueFrames)):
            try:
                y = int(self.i2s(uniqueFrames[k]))
                print(y, pFrames[k])
                if y > x:
                    x = y
                    array.append(pFrames[k])
            except:
                continue
        return array

    #gets the frames at which the pages change
    def get_pages(self, video_name):
        start = time.time()
        cam = cv2.VideoCapture(video_name)
        fr = cam.get(cv2.CAP_PROP_FPS)
        anchors = []
        counter = 0
        current_frame = 0
        x = 248
        y = 102
        w = 351
        h = 264
        frames = []
        while (cam.isOpened()):
            #print("in while")
            ret, frame = cam.read()
            if ret == False:
                break
            elif current_frame == 0:
                name = 'previous_frame.jpg'
                cv2.imwrite(name, frame)
                PIL_image = Image.open('previous_frame.jpg')
                crop = PIL_image.crop((x,y , x + w, y +h))
                crop = self.convert_PIL_to_cv2(crop)
                cv2.imwrite(name, crop)
                current_anchor = name
                diffs = np.zeros_like(np.array(frame))
                np_prev = np.array(frame).astype(float)
            elif counter == self.grit:
                name = 'current_frame.jpg'
                cv2.imwrite(name, frame)
                im1 = Image.open('current_frame.jpg')
                crop = im1.crop((x,y , x + w, y +h))
                crop = self.convert_PIL_to_cv2(crop)
                anchor = Image.open('previous_frame.jpg')
                anchor = self.convert_PIL_to_cv2(anchor)
                gray1 = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(gray1, gray2, full = True)
                diff = (diff * 255).astype("uint8")
                np_tar = np.array(frame).astype(float)
                diffs = np.absolute(np_prev - np_tar) + diffs
                diffs = diffs.astype(float)
                np_prev = np_tar
                if score < .55:
                    frames.append(current_frame)
                    img_cv = cv2.resize(diffs, (854, 504))
                    img_cv = img_cv * 255
                    cv2.imwrite(os.path.join(self.path2, 'diffframe' + str(current_frame) + '.jpg'), img_cv)
                    name = 'frame' + str(current_frame) + '.jpg'
                    cv2.imwrite(os.path.join(self.path2, 'frame' + str(current_frame) + '.jpg'), frame)
                    cv2.imwrite('previous_frame.jpg', crop)
                    diffs = np.zeros_like(np.array(frame))
                    np_prev = np.array(frame).astype(float)
                counter = 0

            current_frame += 1
            counter += 1
        
        cam.release()
        cv2.destroyAllWindows
        
        end = time.time()
        
        print(end - start)
        return frames

    #trained weights from taking the results of applying a linear regression to circle detection based on color intensity
    l_weights = [-0.01808259,  0.03428339,  0.03944322,  0.22787824]
    average_color2 = [104.91666667, 142.58333333, 148.16666667]

    def PrintException(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print ('EXCEPTION IN ({}, LINE {} "{}"):{}'.format(filename, lineno, line.strip(), exc_obj))

    def get_points(self, video_name, time_scaled, page_frames_adj):
        print("hihi")
        start = time.time()
        cam = cv2.VideoCapture(video_name)
        fr = cam.get(cv2.CAP_PROP_FPS)
        anchors = []
        scores = []
        circles = np.empty((0, 3))
        nCircles = np.empty((0, 3))
        info = np.empty((0, 3))
        info2 = np.empty((0, 3))
        time_stamps = np.empty((0, 1))
        time_stamps2 = np.empty((0, 1))
        counter = 0
        current_anchor = ""
        current_frame = 0
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == False:
                break
            elif current_frame == 0:
                name = 'previous_frame.jpg'
                cv2.imwrite(name, frame)
                current_anchor = name
            elif current_frame != 0:
                c_anchor = cv2.imread('previous_frame.jpg')
                gray1 = cv2.cvtColor(c_anchor, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(gray1, gray2, full = True)
                diff = (diff * 255).astype("uint8")
                if counter == self.grit:
                    print(current_frame)
                    name = 'target_frame.jpg'
                    cv2.imwrite(name, frame)
                    im2 = Image.open('target_frame.jpg')
                    np_im2 = np.array(im2)
                    hsv = cv2.cvtColor(np_im2, cv2.COLOR_BGR2HSV)
                    lower_blue = np.array([50, 50, 50])
                    upper_blue = np.array([255, 255, 255])
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    name = 'mask.jpg'
                    cv2.imwrite(name, mask)
                    mask = Image.open('mask.jpg')
                    circle_array, others_array, not_circles_array, not_others_array = self.find_circles(6, 25, 200, 0.5, im2, mask, 120, 10,30)
                    try:
                        circle_array = self.flatten_array(circle_array)
                        others_array = self.flatten_array(others_array)
                        not_circles_array = self.flatten_array(not_circles_array) 
                        not_others_array = self.flatten_array(not_others_array)
                        #stamps = check_recent(circles, circle_array, 10)
                        #for k in range(len(stamps)):
                        if circle_array.ndim == 1:
                            circles = np.vstack([circles, circle_array])
                            time_stamps = np.append(time_stamps, current_frame)
                            info = np.vstack([info, others_array])
                        else:
                            for k in range(len(circle_array)):
                                circles = np.vstack([circles, circle_array[k]])
                                time_stamps = np.append(time_stamps, current_frame)
                                info = np.vstack([info, others_array[k]])
                        for k in range(len(page_frames_adj)):
                            if current_frame < page_frames_adj[k]:
                                filter_frame = time_scaled[k- 1][0]
                                break
                        if not_circles_array.ndim == 1:
                            not_circles_array = [[not_circles_array[0], not_circles_array[1],not_circles_array[2]]]
                            
                        if not_others_array.ndim == 1:
                            not_others_array = [[not_others_array[0], not_others_array[1], not_others_array[2]]]
                            
                        for x, y, r in not_circles_array:
                            patch = self.compute_avg_shade(x, y, r * 3 /4, self.convert_cv2_to_PIL(filter_frame.astype('uint8')))
                            patch= np.sum(patch)
                            nCircles = np.vstack([nCircles, [x, y, r]])
                            time_stamps2 = np.append(time_stamps2, current_frame)
                            
                        for p in range(len(not_others_array)): 
                            info2 = np.vstack([info2, not_others_array[p]]) 
                                
                    except Exception as e:
                        print("Something failed at frame ", current_frame)
                        self.PrintException()
                        print(not_circles_array)
                        print(not_others_array)
                            
                        pass
                    name = 'previous_frame.jpg'
                    cv2.imwrite(name, frame)
                    current_anchor = name
                    counter = 0
                                
            counter += 1
            current_frame += 1

        cam.release()
        cv2.destroyAllWindows
        
        end = time.time()
        
        print(end - start)
        return circles, time_stamps, info, nCircles, info2, time_stamps2

    #gets page numbers with pytesseract

    def pytessGetPoints(self, video):
        pFrames = self.get_pages(video)
        print('pFrames %s' % (pFrames))
        allFrames = list(self.upperRight(self.readFrames(video)))
        print("...")
        _, ui = np.unique(allFrames, axis=0, return_index=True)
        uniqueFrames = [ allFrames[i] for i in pFrames ]
        np.sort(np.round(ui.astype(int) / 10) * 10)
        im = allFrames[len(pFrames)]

        #Set path to wherever pytessteract is installed
        #pytesseract.pytesseract.tesseract_cmd = r"C:/Users/ccui9/tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = "F:/django-eye-tracker/mysite/mysite/static/tesseract/tesseract.exe"

        print('\n starting page_frames \n')
        page_frames = self.find_uFrames(uniqueFrames, pFrames)
        print('\n %s \n' % page_frames)
        page_frames_adj = []
        for frame in page_frames:
            page_frames_adj.append(frame - 10)
        time_scaled = self.get_page_diffs(video, page_frames_adj)
        return time_scaled, page_frames_adj

    #Circles = x and y coordinate of circle
    #time_stamps = frames of above
    #info = the information of the circles, gradient, color, radius
    #nCircles = circles found from gradient detection but judged from the criteria above to not be a circle
    #info2 = info of nCircles
    #time_stamps2 = frames of nCircles


    def appendCirclesAndTimeStamps(self, circles, info, time_stamps, nCircles, info2, time_stamps2):
        circles_and_time_stamps = np.empty((circles.shape[0], 4))
        not_circles_and_time_stamps = np.empty((nCircles.shape[0], 4))
        info_and_time_stamps = np.empty((info.shape[0], 4))
        info2_and_time_stamps = np.empty((info2.shape[0], 4))

        for k in range(len(circles)):
            circles_and_time_stamps[k] = [circles[k][0], circles[k][1], circles[k][2], time_stamps[k]]

        for k in range(len(nCircles)):
            not_circles_and_time_stamps[k] = [nCircles[k][0], nCircles[k][1], nCircles[k][2], time_stamps2[k]]

        for k in range(len(info)):
            info_and_time_stamps[k] = [info[k][0], info[k][1], info[k][2], time_stamps[k]]

        for k in range(len(info2)):
            info2_and_time_stamps[k] = [info2[k][0], info2[k][1], info2[k][2], time_stamps2[k]]

        return circles_and_time_stamps, info_and_time_stamps


    #Not Done below here. Use appendCirclesAndTimeStamps
    def getGradientFoundCircles(self, circles_and_time_stamps, not_circles_and_time_stamps):
            circles = []
            nCircles = []
            for x, y, r, t in circles_and_time_stamps:
                if t > current_frame - S and t < current_frame + S:
                    patch = compute_avg_shade(x, y, r, convert_cv2_to_PIL(filter_frame.astype('uint8')))
                    patch = np.sum(patch)
                    if patch > 150:
                        im_draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 0,255,0))
                        print(x, y, r)
                        print(patch)
                        print(info[c])
                        circles.append()

                    c+= 1

            print("Printing non-circles...") 

            for x, y, r, t in not_circles_and_time_stamps:
                if t > current_frame - S and t < current_frame + S:
                    patch = compute_avg_shade(x, y, r * 3 /4, convert_cv2_to_PIL(filter_frame.astype('uint8')))
                    patch= np.sum(patch)
                    k+=1
                    if patch >100:
                        im_draw.ellipse((x-r, y-r, x+r, y+r), outline=(0, 0,0,0)) 
                        print(patch)
                        print(info2[k])

    def createJSON(self, circles_and_time_stamps, info_and_time_stamps) :
        if (len(info_and_time_stamps) == len(circles_and_time_stamps)) :
            circles_json = {}

            for i in range(0, len(info_and_time_stamps)) :
                # name = ("circle_%d" % i)

                obj = {
                    "x": circles_and_time_stamps[i][0], 
                    "y": circles_and_time_stamps[i][1],
                    "time": info_and_time_stamps[i][3],
                }

                circles_json[i] = obj
            
            return json.dumps(circles_json)
        else :
            return "error size doesn't match"


    def getFinish(self) :
        print("ihih")
        time_scaled, page_frames_adj = self.pytessGetPoints(self.given_path)
        print('\n done with pytessGet \n')
        circles, time_stamps, info, nCircles, info2, time_stamps2 = self.get_points(self.given_path, time_scaled, page_frames_adj)
        print('\n done with finalGet \n')
        circles_and_time_stamps, info_and_time_stamps = self.appendCirclesAndTimeStamps(circles, info, time_stamps, nCircles, info2, time_stamps2)

        return self.createJSON(circles_and_time_stamps, info_and_time_stamps)


#hihi