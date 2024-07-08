import os
import glob
import math
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.transform import resize
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PHOTO_PATH = "Photo"
TELEMETRY_FILE = "Photo/telemetry.csv"
SAVE_DIR = "Polygons"

# # https://support.pix4d.com/hc/en-us/articles/202557469-Step-1-Before-Starting-a-Project-1-Designing-the-Images-Acquisition-Plan-b-Computing-the-Flight-Height-for-a-Given-GSD
# FOCAL_LENGTH_MM = (FOCAL_LENGTH_35_MM * CCD_WIDTH_MM) / 34.6
# FOCAL_LENGTH_35_MM = 20.0
FOCAL_LENGTH_MM = 35.0

# # http://forum.dji.com/thread-28597-1-1.html
# CCD_WIDTH_MM = 6.16
# CCD_HEIGHT_MM = 4.62

CCD_WIDTH_MM = 35.8
CCD_HEIGHT_MM = 23.9


def four_point_transform(img, rot_y, rot_x):
    height, width = img.shape[:2]

    # the from pts to be warped
    src_rect = create_src_rect(height, width)
    # the rotated destination pts
    dst_rect = pitch_roll_pts(height, width, rot_y, rot_x)
    (tl, tr, br, bl) = dst_rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # compute the perspective transform matrix and then apply it
    # M, mask = cv2.findHomography(src_rect, dst_rect, cv2.RANSAC, 5.0) return same result for M
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # return the warped image
    return warped


def create_src_rect(height, width):
    coords = np.array([(0, height), (0, 0), (width, 0), (width, height)])
    return order_points(coords)


# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# order the envelope points so that they are clockwise starting at upper left
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def pitch_roll_pts(height, width, rot_y, rot_x):
    # pts represent the corners of the image that will be tilted about the center
    pts = np.matrix([[0, 0, width, width], [height, 0, 0, height], [0, 0, 0, 0]])

    center_x = width / 2.0
    center_y = height / 2.0
    # the offset is needed to tilt the pts about the center instead of about the origin
    center_offset = np.matrix(
        [
            [center_x, center_x, center_x, center_x],
            [center_y, center_y, center_y, center_y],
            [0, 0, 0, 0],
        ]
    )
    # shift coordinates so that origin is the center of the image
    pts = pts - center_offset

    # rotate image about the center and add the offset back into the coordinates
    pts = rot_y * rot_x * pts + center_offset

    coords = np.array(
        [
            (pts[0, 0], pts[1, 0]),
            (pts[0, 1], pts[1, 1]),
            (pts[0, 2], pts[1, 2]),
            (pts[0, 3], pts[1, 3]),
        ]
    )
    return order_points(coords)


def calculate_drone_position_pixel(img, pitch_rad, roll_rad):
    height, width = img.shape[:2]

    shift_x = FOCAL_LENGTH_MM * math.tan(roll_rad) * (width / CCD_WIDTH_MM)
    shift_y = FOCAL_LENGTH_MM * math.tan(pitch_rad) * (height / CCD_HEIGHT_MM)

    # shift_x = math.tan(roll_rad) * (width)
    # shift_y = math.tan(pitch_rad) * (height)

    center_x = width / 2.0
    center_y = height / 2.0

    return int(center_y + shift_y), int(center_x + shift_x)


# http://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
def rotate_image(img, angle, pivot):
    padX = [img.shape[1] - pivot[1], pivot[1]]
    padY = [img.shape[0] - pivot[0], pivot[0]]
    imgP = np.pad(img, [padY, padX, [0, 0]], "constant")
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR


def calculate_earth_radius(latitude):
    # Constants
    a = 6378137  # Equatorial radius of the Earth in meters
    f = 1 / 298.257223563  # Earth flattening
    e2 = f * (2 - f)  # Square of the first eccentricity

    # Calculate the radius of curvature in the prime vertical
    N = a / math.sqrt(1 - e2 * math.sin(math.radians(latitude)) ** 2)

    return N


def correct_lat_lon(lat, lon, alt_gps, roll, pitch):
    # Constants
    earth_radius = calculate_earth_radius(lat)

    # Convert GPS altitude to geoidal height
    geo_height = alt_gps

    # Displacement in X and Y directions
    disp_x = geo_height * math.tan(math.radians(roll))
    disp_y = geo_height * math.tan(math.radians(pitch))

    # Correction in latitude and longitude
    lat_correction = math.degrees(math.asin(disp_y / earth_radius))
    lon_correction = math.degrees(
        math.asin(disp_x / (earth_radius * math.cos(math.radians(lat))))
    )

    new_lat = lat + lat_correction
    new_lon = lon + lon_correction

    return new_lat, new_lon


def haversine(lat1, lon1, lat2, lon2):
    R = calculate_earth_radius(lat1)

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def cm_per_pixel(img, pitch_rad, roll_rad, elevation_m):
    height, width = img.shape[:2]

    pitch_offset = elevation_m * math.tan(pitch_rad)
    roll_offset = elevation_m * math.tan(roll_rad)

    distance_m = math.sqrt(math.sqrt(pitch_offset**2 + roll_offset**2) + elevation_m**2)
    return (CCD_WIDTH_MM * distance_m * 100) / (FOCAL_LENGTH_MM * width)


def envelope_from_image(img, lon1_deg, lat1_deg, x_pixel, y_pixel, GSD):
    height, width = img.shape[:2]

    #               up               right                       down                        left
    azimuths_deg = [0.0, 90.0, 180.0, 270.0]

    distances_cm = [
        y_pixel * GSD,
        (width - x_pixel) * GSD,
        (height - y_pixel) * GSD,
        x_pixel * GSD,
    ]

    lon_null, lat_up = coords_from_azi_distance(
        lat1_deg, lon1_deg, azimuths_deg[0], distances_cm[0]
    )
    lon_right, lat_null = coords_from_azi_distance(
        lat1_deg, lon1_deg, azimuths_deg[1], distances_cm[1]
    )
    lon_null, lat_down = coords_from_azi_distance(
        lat1_deg, lon1_deg, azimuths_deg[2], distances_cm[2]
    )
    lon_left, lat_null = coords_from_azi_distance(
        lat1_deg, lon1_deg, azimuths_deg[3], distances_cm[3]
    )

    # ulx uly lrx lry
    return lon_left, lat_up, lon_right, lat_down


def coords_from_azi_distance(lat1_deg, lon1_deg, azimuth_deg, distance_cm):
    radius_km = (
        calculate_earth_radius(lat1_deg) / 1000.0
    )  # 6378.1  # Radius of the Earth

    azimuth_rad = math.radians(azimuth_deg)
    distance_km = distance_cm / 100000
    lat1_rad = math.radians(lat1_deg)  # Current lat point converted to radians
    lon1_rad = math.radians(lon1_deg)  # Current long point converted to radians

    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(distance_km / radius_km)
        + math.cos(lat1_rad) * math.sin(distance_km / radius_km) * math.cos(azimuth_rad)
    )
    lon2_rad = lon1_rad + math.atan2(
        math.sin(azimuth_rad) * math.sin(distance_km / radius_km) * math.cos(lat1_rad),
        math.cos(distance_km / radius_km) - math.sin(lat1_rad) * math.sin(lat2_rad),
    )

    lat2_deg = math.degrees(lat2_rad)
    lon2_deg = math.degrees(lon2_rad)

    return lon2_deg, lat2_deg


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    telemetry = pd.read_csv("telemetry.csv")

    rotated_images = {}

    if os.path.isfile(os.path.join(SAVE_DIR, "edges.csv")):
        edges = pd.read_csv(os.path.join(SAVE_DIR, "edges.csv")).to_dict("list")
    else:
        edges = {
            "file": [],
            "ulx": [],
            "uly": [],
            "lrx": [],
            "lry": [],
            "lat_img": [],
            "lon_img": [],
        }

    for index, row in telemetry.iterrows():
        pt = os.path.join(SAVE_DIR, row["file"].split(".")[0] + ".webp")
        if pt in glob.glob(SAVE_DIR + "/*"):
            continue

        try:

            img = mpimg.imread(os.path.join(PHOTO_PATH, row["file"]))
            img = resize(img, (img.shape[0] // 2, img.shape[1] // 2))

            # get yaw, pitch, roll and elevation
            yaw_deg = float(row["yaw"])
            pitch_rad = -math.radians(float(row["pitch"]))
            roll_rad = -math.radians(float(row["roll"]))
            elevation_meters = float(row["alt"])

            # pitch rotation (axis through wing perpendicular to flight path)
            RotY = np.matrix(
                [
                    [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
                    [0, 1, 1],
                    [-math.sin(pitch_rad), 0, math.cos(pitch_rad)],
                ]
            )

            # roll rotation (axis through direction of flight path)
            RotX = np.matrix(
                [
                    [1, 0, 0],
                    [0, math.cos(roll_rad), -math.sin(roll_rad)],
                    [0, math.sin(roll_rad), math.cos(roll_rad)],
                ]
            )

            # use rotation matrices to adjust image for pitch and roll errors. Rotate about the center of image.
            warped_image = four_point_transform(img, RotY, RotX)

            # get the pixel directly below the drone (not the center of the image)
            drone_pixel_x, drone_pixel_y = calculate_drone_position_pixel(
                warped_image, pitch_rad, roll_rad
            )

            rotated_image = rotate_image(
                warped_image, -yaw_deg, (drone_pixel_x, drone_pixel_y)
            )
            # rotated_image= ndimage.rotate(img, -yaw_deg, (1, 0))
            rotated_images[row["file"]] = rotated_image

            GSD = cm_per_pixel(
                warped_image, pitch_rad, roll_rad, elevation_meters
            )  # Ground sampling distance
            GSD *= 0.525

            lon_deg = float(row["lon"])
            lat_deg = float(row["lat"])

            height, width = rotated_image.shape[:2]

            ulx, uly, lrx, lry = envelope_from_image(
                rotated_image, lon_deg, lat_deg, width / 2, height / 2, GSD
            )

            edges["file"].append(pt)
            edges["ulx"].append(ulx)
            edges["uly"].append(uly)
            edges["lrx"].append(lrx)
            edges["lry"].append(lry)

            new_lat, new_lon = correct_lat_lon(
                row["lat"], row["lon"], row["alt"], row["roll"], row["pitch"]
            )

            edges["lat_img"].append(new_lat)
            edges["lon_img"].append(new_lon)

            # Convert to RGBA
            arr_rgba = np.empty(
                (rotated_image.shape[0], rotated_image.shape[1], 4),
                dtype=rotated_image.dtype,
            )

            arr_rgba[:, :, :3] = rotated_image

            rgba_uint8_array = (arr_rgba * 255).astype(np.uint8)

            rgba_uint8_array[:, :, 3] = (
                255 - (abs(rgba_uint8_array.sum(axis=2)) == 0) * 255
            )  # set alpha channel to 0 for black pixels

            Image.fromarray(rgba_uint8_array, mode="RGBA").save(
                pt, "WEBP", exact=True, quality=0, method=0
            )
            print("ready", row["file"])

            # fig = plt.figure(figsize=(10, 7))
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(img)
            # fig.add_subplot(1, 2, 2)
            # plt.imshow(rotated_image)

        except:
            print("failed ", row["file"])

        pd.DataFrame(edges).to_csv(os.path.join(SAVE_DIR, "edges.csv"), index=False)
