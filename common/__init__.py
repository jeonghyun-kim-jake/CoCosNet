
import numpy as np

import pathlib
import cv2
import math

import io

from datetime import datetime
from PIL import Image

import torch

import numpy as np

import logging
import logging.handlers


def create_ade20k_challenge_label_colormap():
    return np.load('common/colormap/ade20k_challenge_colormap.npy')

MATRIX = None
#create_ade20k_challenge_label_colormap()

def convert_image_to_bytes(image, img_format='PNG'):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=img_format)
    return imgByteArr.getvalue()

def convert_ade20k_to_label(image):
    image = image.convert('RGB')
    width , height = image.size
    numpy_image = np.array(image)
    all_pixels = np.zeros((height, width), dtype=np.uint8)
    for label in range(0, len(MATRIX)):
        if label > 0 :
            where = np.where(np.all(MATRIX[label] == numpy_image, axis=-1))
            for y, x in zip(where[0], where[1]):
                all_pixels[y][x] = label
    return Image.fromarray(all_pixels)

def convert_label_to_ade20k(image):
    image = image.convert('L')
    width, height = image.size
    all_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    img_array = np.array(image)
    for x in range(width):
        for y in range(height):
            label = img_array[y,x]
            all_pixels[y][x] = MATRIX[label]
    return Image.fromarray(all_pixels)

def createDir(path):
    pathlib.Path(path).mkdir(exist_ok=True)

def distance(a1, b1):
    a = a1[0] - b1[0]
    b = a1[1] - b1[1]
    c = math.sqrt((a*a)+(b*b))
    return c

def blending_patch(bg_img, fr_img, blending_area, position, patch_size):
    # bg_img = cv2.imread(args.bg_img_path, -1)
    # fr_img = cv2.imread(args., -1)
    h, w, depth = bg_img.shape
    patch_w, patch_h = patch_size[0], patch_size[1]
    fr_img_resize = cv2.cvtColor(cv2.resize(fr_img, dsize=(patch_w, patch_h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)

    result = bg_img

    fr_start_x, fr_start_y= position[0], position[1]
    center_x, center_y= fr_start_x+int(patch_w/2), fr_start_y+int(patch_h/2)

    distance_base = distance((fr_start_x, fr_start_y),(center_x, center_y))

    # area = (args.fr_start_x + args.fr_start_y)/4
    max_blending_area = int((patch_w+patch_h)/4)
    area = min(blending_area, max_blending_area)


    for y in range(fr_start_y, fr_start_y+patch_h):
        if y >= h:
            continue
        p_y = y - fr_start_y
        for x in range(fr_start_x, fr_start_x+patch_w):
            if x >= w:
                continue
            p_x = x - fr_start_x
            color_fr = fr_img_resize[p_y, p_x]
            color_bg = bg_img[y, x]

            if (x>=fr_start_x+area and x<fr_start_x+patch_w-area) and (y>=fr_start_y+area and y<fr_start_y+patch_h-area):
                new_color = color_fr
                # new_color = (100, 100, 100)
            else:
                if x<=fr_start_x+area:
                  x_r = abs(fr_start_x+area-x)
                elif x>=fr_start_x+patch_w-area:
                  x_r = abs(x-(fr_start_x+patch_w-area))
                else:
                  x_r = 0

                if y<=fr_start_y+area :
                  y_r = abs(fr_start_y+area-y)
                elif y>fr_start_y+patch_h-area:
                  y_r = abs(y-(fr_start_y+patch_h-area))
                else:
                  y_r = 0

                if x_r is 0:
                  alpha = y_r/area
                elif y_r is 0:
                  alpha = x_r/area
                else:
                  ret = math.sqrt(pow(y_r/area, 2) + pow(x_r/area, 2))
                  alpha = ret if ret <=1 else 1
                # print("area",area,"x_r",x_r,"y_r",y_r,"alpha",alpha)
                new_color = [ (1 - alpha) * color_fr[0] + alpha * color_bg[0],
                              (1 - alpha) * color_fr[1] + alpha * color_bg[1],
                              (1 - alpha) * color_fr[2] + alpha * color_bg[2]]
                # new_color = [0,0,0]
            result[y, x] = (new_color)

    return result


def blending_patch_pil(bg_img, fr_img, blending_area, position, patch_size):
    # bg_img = cv2.imread(args.bg_img_path, -1)
    # fr_img = cv2.imread(args., -1)
    w, h = bg_img.size
    patch_w, patch_h = patch_size[0], patch_size[1]
    
    fr_img_resize = np.asarray(fr_img.resize((patch_w, patch_h)))
    result = np.copy(np.asarray(bg_img))

    fr_start_x, fr_start_y= position[0], position[1]
    center_x, center_y= fr_start_x+int(patch_w/2), fr_start_y+int(patch_h/2)

    distance_base = distance((fr_start_x, fr_start_y),(center_x, center_y))

    # area = (args.fr_start_x + args.fr_start_y)/4
    max_blending_area = int((patch_w+patch_h)/4)
    area = min(blending_area, max_blending_area)


    for y in range(fr_start_y, fr_start_y+patch_h):
        if y >= h:
            continue
        p_y = y - fr_start_y
        for x in range(fr_start_x, fr_start_x+patch_w):
            if x >= w:
                continue
            p_x = x - fr_start_x
            color_fr = fr_img_resize[p_y, p_x]
            color_bg = result[y, x]

            if (x>=fr_start_x+area and x<fr_start_x+patch_w-area) and (y>=fr_start_y+area and y<fr_start_y+patch_h-area):
                new_color = color_fr
                # new_color = (100, 100, 100)
            else:
                if x<=fr_start_x+area:
                  x_r = abs(fr_start_x+area-x)
                elif x>=fr_start_x+patch_w-area:
                  x_r = abs(x-(fr_start_x+patch_w-area))
                else:
                  x_r = 0

                if y<=fr_start_y+area :
                  y_r = abs(fr_start_y+area-y)
                elif y>fr_start_y+patch_h-area:
                  y_r = abs(y-(fr_start_y+patch_h-area))
                else:
                  y_r = 0

                if x_r is 0:
                  alpha = y_r/area
                elif y_r is 0:
                  alpha = x_r/area
                else:
                  ret = math.sqrt(pow(y_r/area, 2) + pow(x_r/area, 2))
                  alpha = ret if ret <=1 else 1
                # print("area",area,"x_r",x_r,"y_r",y_r,"alpha",alpha)
                new_color = [ (1 - alpha) * color_fr[0] + alpha * color_bg[0],
                              (1 - alpha) * color_fr[1] + alpha * color_bg[1],
                              (1 - alpha) * color_fr[2] + alpha * color_bg[2]]
                # new_color = [0,0,0]
            result[y, x] = (new_color)
                               
    return Image.fromarray(result)

def get_bytes_from_image(image, format="PNG"):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=format)
    return imgByteArr.getvalue()

def get_image_from_bytes(image):
    return Image.open(io.BytesIO(bytes))


global _time_latest
_time_latest = datetime.now()
def printTimeDiff(msg):
    global _time_latest
    now = datetime.now()
    diff = now - _time_latest
    print ("\t[TIME_PROFILE] {}".format(msg), str(diff))
    _time_latest = now



def getGPUAvailable():
    return torch.cuda.is_available()


def getTimeStr():
    now = datetime.now()
    return now.strftime("%m%d_%H%M")

def config_logger(prefix):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] >> %(message)s")
    fileMaxBytes = 1024 * 1024 * 100 #100MB

    streamHandler = logging.StreamHandler()
    logger.addHandler(streamHandler)
    streamHandler.setFormatter(formatter)

    fileHandler = logging.handlers.RotatingFileHandler("./logs/{}_{}.log".format(prefix, getTimeStr()), maxBytes=fileMaxBytes, backupCount=20)
    logger.addHandler(fileHandler)
    fileHandler.setFormatter(formatter)

    logger.setLevel(level=logging.DEBUG)

    return logger


    
