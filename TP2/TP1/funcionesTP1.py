import cv2

def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel) # erosion dilatacion
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # dilatacion y erosion
    return closing


def draw_contours(frame, contours, color, thickness):
    # -1 for all contours
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def threshold(frame, slider_max, binary, trackbar_value):
    ret, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th


def filter_contours_byArea(contours, min_area, max_area):
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                filtered_contours.append(contour)
        return filtered_contours

def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name) / 2) * 2 + 3

def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours

def get_saved_contours(path):
    imagen = cv2.imread(path)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ret, imagen = cv2.threshold(imagen, 120, 255, cv2.THRESH_BINARY)
    return get_contours(imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
