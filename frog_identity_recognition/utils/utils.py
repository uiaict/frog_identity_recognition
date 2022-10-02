import numpy as np
import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def contour_resize(mask, orginal_image, image):
    dw = image.shape[1] / orginal_image.shape[1]
    dh = image.shape[0] / orginal_image.shape[0]
    scaler = np.array([[dw, dh]])

    mask = (mask * scaler).astype("int64")
    return mask


def bbox_resize(box, orginal_image, image):
    x, y, w, h = box
    dw = image.shape[1] / orginal_image.shape[1]
    dh = image.shape[0] / orginal_image.shape[0]
    return [int(x * dw), int(y * dh), int(w * dw), int(h * dh)]