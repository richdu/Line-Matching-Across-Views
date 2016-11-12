import numpy as np
import cv2

LINE_FINITE = 0
LINE_INFINITE = 1


def draw_line(img, line, color, thickness, line_type):
    if line_type == LINE_FINITE:
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color, thickness, cv2.LINE_AA)
    # Infinite lines ax + by + c = 0 are represented by a homogeneous 3-vector (a, b, c)
    elif line_type == LINE_INFINITE:
        _, c, _ = img.shape
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def __segment2line(segment):
    x0, y0 = [segment[0], segment[1]]
    x1, y1 = [segment[2], segment[3]]
    return np.asarray([y0-y1, x1-x0, y1*x0-x1*y0]).astype(float)


def intersect_lines(line1, line2):
    homogeneous = np.cross(line1, line2)
    homogeneous = homogeneous.astype(float)
    homogeneous /= homogeneous[2]
    return homogeneous[0:2]


def intersect_segment_line(segment, line):
    x0, y0 = [segment[0], segment[1]]
    x1, y1 = [segment[2], segment[3]]
    segment = __segment2line(segment)
    x, y = intersect_lines(segment, line)
    if (x0 <= x <= x1 or x0 >= x >= x1) and (y0 <= y <= y1 or y0 >= y >= y1):
        return True, x, y
    else:
        return False


def in_beam(segment, line1, line2):
    a, b = [segment[0], segment[1]]
    c, d = [segment[2], segment[3]]
    segment = __segment2line(segment)
    x0, y0 = intersect_lines(segment, line1)
    print x0, y0
    x1, y1 = intersect_lines(segment, line2)
    print x1, y1
    min_x = min(x0, x1)
    max_x = max(x0, x1)
    min_y = min(y0, y1)
    max_y = max(y0, y1)

    if (a <= min_x and c <= min_x) or (a >= max_x and c >= max_x):
        return False
    elif (b < min_y and d < min_y) or (b > max_y and d > max_y):
        print b, d, min_y, max_y
        return False
    else:
        return True

# print intersect_segment_line([0, 0, 0, 10], np.asarray([1, -2, 10]))
print in_beam([-10, 1, 10, 1], np.asarray([1, -2, 0]), np.asarray([-2, 1, 0]))
# print intersect_lines(np.asarray([1, -2, 0]), np.asarray([1, 1, -4]))
# print __segment2line([0, 0, 0, 1])
