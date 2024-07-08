import os
import numpy as np
import pandas as pd
import cv2

PHOTO_PATH = "Polygons"
SAVE_DIR = "Canvas_cv"


def stitch_images(img1, img2, H):
    """
    function for warping/stitching two images using the homography matrix H

    Args:
        img1  : first image, or source image
        img2  : second image, that needs to be mapped to the frame of img1
        H     : homography matrix that maps img2 to img1

    Returns:
        output_img: img2 warped to img1 using H
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(
        -1, 1, 2
    )
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )

    points_2 = cv2.perspectiveTransform(temp_points, H)
    points_concat = np.concatenate((points_1, points_2), axis=0)

    [x_min, y_min] = np.int32(points_concat.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points_concat.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )

    output_img = cv2.warpPerspective(
        img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
    )

    frame_size = output_img.shape
    new_image = img2.shape

    output_img[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = img1

    origin_r = int(points_2[0][0][1])
    origin_c = int(points_2[0][0][0])

    # if the origin of projected image is out of bounds, then mapping to ()
    if origin_r < 0:
        origin_r = 0
    if origin_c < 0:
        origin_c = 0

    # Clipping the new image, if it's size is more than the frame
    if new_image[0] > frame_size[0] - origin_r:
        img2 = img2[0 : frame_size[0] - origin_r, :]

    if new_image[1] > frame_size[1] - origin_c:
        img2 = img2[:, 0 : frame_size[1] - origin_c]

    output_img[origin_r : new_image[0] + origin_r, origin_c : new_image[1] + origin_c][
        img2[:, :, 3] != 0.0
    ] = img2[img2[:, :, 3] != 0.0]

    return output_img


def ratio_test(good, matches, ratio):
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def build_mosaic(img_names, save_dir, inlier_thresh=0.2, scaling_f=0.15):

    nfeatures = 200000
    # sift = cv2.ORB_create(nfeatures)
    sift = cv2.SIFT_create(nfeatures)

    mosaic_path = img_names.pop(0)
    total_mosaic = cv2.imread(mosaic_path, cv2.IMREAD_UNCHANGED)

    height, width = total_mosaic.shape[:2]

    total_mosaic = cv2.resize(
        total_mosaic, (int(width * scaling_f), int(height * scaling_f))
    )

    for i in range(0, len(img_names)):

        img_path = img_names.pop(0)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width * scaling_f), int(height * scaling_f)))

        # Now get the kepoints from mosaic and image

        # convert images to grayscale
        base_gray = cv2.cvtColor(total_mosaic, cv2.COLOR_BGRA2GRAY)
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # create a mask using the alpha channel of the original image--don't
        # use transparent or partially transparent parts
        base_cond = total_mosaic[:, :, 3] == 255
        base_mask = np.array(np.where(base_cond, 255, 0), dtype=np.uint8)

        curr_cond = image[:, :, 3] == 255
        curr_mask = np.array(np.where(curr_cond, 255, 0), dtype=np.uint8)

        keypoints_mosaic, descriptors_mosaic = sift.detectAndCompute(
            base_gray, mask=base_mask
        )

        keypoints_image, descriptors_image = sift.detectAndCompute(
            curr_gray, mask=curr_mask
        )

        # Now match the keypoints
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_mosaic, descriptors_image, k=2)

        # Apply ratio test
        good = []
        matches_needed = 20
        # the lower the ratio the more stringent the requirement
        ratio_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        # ratio values will keep appending values as matches are met
        for ratio in ratio_values:
            good = ratio_test(good, matches, ratio)
            if len(good) > matches_needed:
                break

        if len(good) > matches_needed - 1:
            # Extract matched keypoints
            dst_pts = np.float32(
                [keypoints_mosaic[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            src_pts = np.float32(
                [keypoints_image[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            # Calculate the homography using RANSAC algorithm
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

            # Check the reliability of the individual homography using the inlier count
            inlier_ratio = np.sum(mask) / mask.size
            if inlier_ratio < inlier_thresh:
                print("Inlier ratio was lower than the threshold for ", img_path)
            else:
                result = stitch_images(total_mosaic, image, H)

                total_mosaic = result
                # Save the mosaic
                cv2.imwrite(os.path.join(save_dir, f"{i+2}.png"), total_mosaic)
                print("Ready", img_path)
        else:
            print("Not enough matches found for ", img_path)

    return


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    edges = pd.read_csv(os.path.join(PHOTO_PATH, "edges.csv"))

    img_names = edges["file"].values.tolist()

    build_mosaic(img_names, SAVE_DIR)
