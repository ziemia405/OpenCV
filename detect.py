import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

    [x, y, n] = np.shape(img)

    sizex = 1000
    sizey = 1000

    if x > y:
        raw_scaled = cv2.resize(img, (sizex, sizey), interpolation=cv2.INTER_LANCZOS4)
    elif y > x:
        raw_scaled = cv2.resize(img, (sizey, sizex), interpolation=cv2.INTER_LANCZOS4)

    raw_scaled_gray = cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2GRAY)

    hsv_g = cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([33, 25, 25])
    upper_bound = np.array([115, 255, 255])
    mask = cv2.inRange(hsv_g, lower_bound, upper_bound)
    imask = mask > 0
    green = np.zeros_like(raw_scaled, np.uint8)
    green[imask] = raw_scaled[imask]
    green = cv2.medianBlur(green, 3)

    image2 = green
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4, minDist=46, param1=12, param2=27.5, minRadius=11,
                               maxRadius=19)
    g = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image2, (x, y), r, (36, 255, 12), 1)
            g = g + 1
    # print('Na zdjeciu jest '+str(g)+'. zielonych cukierków.')
    # cv2.imshow('img', image2)
    # cv2.waitKey(0)

    hsv_p = cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([110, 50, 53]) #150 - 3.19 #140 3.043 #135 3.043 140, 53, 53
    upper_bound = np.array([179, 170, 170])
    mask = cv2.inRange(hsv_p, lower_bound, upper_bound)
    imask = mask > 0
    purple = np.zeros_like(raw_scaled, np.uint8)
    purple[imask] = raw_scaled[imask]
    purple = cv2.medianBlur(purple, 3)

    image2 = purple
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4, minDist=42.25, param1=12, param2=30, minRadius=11,
                               maxRadius=20)
    p = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image2, (x, y), r, (36, 255, 12), 1)
            p = p + 1
    # print('Na zdjeciu jest '+str(p)+'. fioletowych cukierków.')
    # cv2.imshow('img1', image2)
    # cv2.waitKey(0)

    hsv_y = cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_y, lower_bound, upper_bound)
    imask = mask > 0
    yellow = np.zeros_like(raw_scaled, np.uint8)
    yellow[imask] = raw_scaled[imask]
    yellow = cv2.medianBlur(yellow, 3)

    image2 = yellow
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4, minDist=33.575, param1=19, param2=29, minRadius=12,
                               maxRadius=16)
    z = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image2, (x, y), r, (36, 255, 12), 1)
            z = z + 1
    # print('Na zdjeciu jest '+str(z)+'. żółtych cukierków.')
    # cv2.imshow('img3', image2)
    # cv2.waitKey(0)

    hsv_r = cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([150, 130, 130])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_r, lower_bound, upper_bound)
    imask = mask > 0
    red = np.zeros_like(raw_scaled, np.uint8)
    red[imask] = raw_scaled[imask]
    red = cv2.medianBlur(red, 3)

    image2 = red
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4, minDist=42, param1=10, param2=24, minRadius=8,
                               maxRadius=22)
    c = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image2, (x, y), r, (36, 255, 12), 1)
            c = c + 1
    red = c
    yellow = z
    green = g
    purple = p

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
