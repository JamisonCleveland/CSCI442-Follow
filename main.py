import cv2 as cv
import numpy as np
import math

# Steps:
# 1. preprocess image to remove noise
# 2. segment image from background to directional components
# 3. find center of gravity from directional components  

def main(path):
    cap = cv.VideoCapture(path)
    while True:
        _ret, img = cap.read()
        img = cv.resize(img, (200, 200))

        prepro_img = preprocess(img, debug=True)
        seg_img = segment(prepro_img, debug=True)
        seg_img[img.shape[0] // 2:, :] = 0 # blackout everything below midpoint
        _direction = get_direction(seg_img, debug=True)

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def get_direction(segment_img, debug=False):
    max_y, max_x = segment_img.shape[:2]
    mid_x = max_x // 2
    mid_y = max_y // 2

    # find segment of greatest area in forground
    contours, _thing = cv.findContours(segment_img, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
    #a = []
    #for c in contours:
    #    perimeter = cv.arcLength(c, True)
    #    approx = cv.approxPolyDP(c, 0.4 * perimeter, True)
    #    if len(approx) == 4:
    #        a.append(c)
    #contours = a
    largest_contour = max((c for c in contours), key=lambda c: cv.contourArea(c), default=None) 
    if largest_contour is None:
        return 0, 0

    masked_img = np.zeros(segment_img.shape[:2], dtype=np.uint8)
    cv.drawContours(masked_img, [largest_contour], 0, 1, -1)

    # calculate center of gravity and direction
    moments = cv.moments(masked_img, True)
    cog_x = moments['m10'] / moments['m00']
    cog_y = moments['m01'] / moments['m00']

    cog_diff_x = cog_x - mid_x
    cog_diff_y = cog_y - mid_y
    dist = math.sqrt(cog_diff_x ** 2 + cog_diff_y ** 2)
    dir_x = cog_diff_x / dist
    dir_y = cog_diff_y / dist

    # draw result
    if debug:
        dir_len = 30
        p_mid = mid_x, mid_y
        p_cog = int(cog_x), int(cog_y)
        p_dir = (int(mid_x + dir_len * dir_x), int(mid_y + dir_len * dir_y))

        debug_img = np.float32(segment_img) * 255.0
        debug_img = cv.cvtColor(debug_img, cv.COLOR_GRAY2BGR)
        cv.drawContours(debug_img, [largest_contour], -1, (0, 0, 255))
        cv.circle(debug_img, p_cog, 5, (255, 255, 0), -1)
        cv.line(debug_img, p_mid, p_cog, (255, 255, 0))
        cv.circle(debug_img, p_dir, 5, (0, 255, 0), -1)
        cv.line(debug_img, p_mid, p_dir, (0, 255, 0))
        cv.imshow('result', debug_img)

    return dir_x, dir_y

def preprocess(img, debug=False):
    blurred_img = cv.blur(img, (3, 3))

    # blur and subtract to remove lighting
    blur_gray_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)
    blur_gray_img = cv.blur(blur_gray_img, (300, 300))
    blur_gray_img = cv.cvtColor(blur_gray_img, cv.COLOR_GRAY2BGR)

    diff_img = cv.absdiff(blurred_img, blur_gray_img)

    if debug:
        cv.imshow('preprocessed', diff_img)

    return diff_img


def segment(img, debug=False):
    samples = np.float32(img.reshape((-1, img.shape[2])))

    _compactness, labels, centers = \
        cv.kmeans(samples,
                  2,
                  None,
                  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                  1,
                  cv.KMEANS_RANDOM_CENTERS)

    meaned_img = np.uint8(centers[labels.flatten()].reshape(img.shape))
    
    ## find background label as kmeans can randomly initialize centers
    centers_hsv = cv.cvtColor(np.expand_dims(centers, axis=0), cv.COLOR_BGR2HSV).squeeze(axis=0)
    bg_label = centers_hsv[:, 2].argmin() # background should be have the lowest intensity of 'value'

    bin_img = np.where(labels.flatten() == bg_label, 0, 1).reshape(img.shape[:2])

    if debug:
        print('compactness:', _compactness)
        cv.imshow('segmentation', meaned_img)

    return bin_img

if __name__ == '__main__':
    main('tracking_test.mp4')