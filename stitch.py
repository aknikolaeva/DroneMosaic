import os
import numpy as np
import pandas as pd
import cv2

PHOTO_PATH = "Polygons"
SAVE_DIR = "Canvas"


def stitch(images_metadata, save_dir, scaling_f=0.15):

    meta = images_metadata.pop(0)
    canvas = cv2.imread(meta["file"], cv2.IMREAD_UNCHANGED)

    canvas = cv2.resize(
        canvas, (0, 0), fx=scaling_f, fy=scaling_f, interpolation=cv2.INTER_AREA
    )

    ULX = meta["ulx"]
    ULY = meta["uly"]

    for i in range(0, len(images_metadata)):

        meta = images_metadata.pop(0)
        image = cv2.imread(meta["file"], cv2.IMREAD_UNCHANGED)

        image = cv2.resize(
            image, (0, 0), fx=scaling_f, fy=scaling_f, interpolation=cv2.INTER_AREA
        )

        X_scale = abs(meta["lrx"] - meta["ulx"]) / image.shape[1]
        Y_scale = abs(meta["lry"] - meta["uly"]) / image.shape[0]

        ulx_pix = int((meta["ulx"] - ULX) / X_scale)
        uly_pix = int((meta["uly"] - ULY) / Y_scale)

        ulx_pix_abs = int(abs(ulx_pix))
        uly_pix_abs = int(abs(uly_pix))

        if uly_pix >= 0:
            pad_widths = ((uly_pix_abs, 0), (0, 0), (0, 0))
            canvas = np.pad(canvas, pad_widths, mode="constant", constant_values=0)

        if ulx_pix < 0:
            pad_widths = ((0, 0), (ulx_pix_abs, 0), (0, 0))
            canvas = np.pad(canvas, pad_widths, mode="constant", constant_values=0)

        ULX = min(ULX, meta["ulx"])
        ULY = max(ULY, meta["uly"])

        ulx_pix = int((meta["ulx"] - ULX) / X_scale)
        uly_pix = int((meta["uly"] - ULY) / Y_scale)

        ulx_pix_abs = int(abs(ulx_pix))
        uly_pix_abs = int(abs(uly_pix))

        canvas_y = image.shape[0]
        canvas_x = image.shape[1]

        if uly_pix <= 0:
            dy = image.shape[0] - canvas[uly_pix_abs:, :, :].shape[0]
            if dy > 0:
                canvas = np.pad(
                    canvas,
                    ((0, dy), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            dy = image.shape[0] - canvas[uly_pix_abs:, :, :].shape[0]
            if dy > 0:
                canvas = np.pad(
                    canvas,
                    ((dy, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        if ulx_pix <= 0:
            dx = image.shape[1] - canvas[:, ulx_pix_abs:, :].shape[1]
            if dx > 0:
                canvas = np.pad(
                    canvas,
                    ((0, 0), (dx, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            dx = image.shape[1] - canvas[:, ulx_pix_abs:, :].shape[1]
            if dx > 0:
                canvas = np.pad(
                    canvas,
                    ((0, 0), (0, dx), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        canvas[
            uly_pix_abs : uly_pix_abs + canvas_y,
            ulx_pix_abs : ulx_pix_abs + canvas_x,
            :,
        ][image[:, :, 3] != 0] = image[image[:, :, 3] != 0]

        cv2.imwrite(os.path.join(save_dir, f"{i+2}.png"), canvas)
        print("Ready", meta["file"])
    return


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    edges = pd.read_csv(os.path.join(PHOTO_PATH, "edges.csv"))

    images_metadata = edges.to_dict("records")

    stitch(images_metadata, SAVE_DIR)
