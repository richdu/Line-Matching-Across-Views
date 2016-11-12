import numpy as np
import cv2

LINE_FINITE = 0
LINE_INFINITE = 1


def length(segment):
    x1, y1 = [segment[0], segment[1]]
    x2, y2 = [segment[2], segment[3]]
    dx = x2 - x1
    dy = y2 - y1
    return (dx*dx+dy*dy)**0.5


def points_on_segment(segment):
    """
    Implements Bresenham's algorithm to get all points on a line segment
    :type segment: numpy array of length 4
    """

    # Segment endpoints
    x1, y1 = [segment[0], segment[1]]
    x2, y2 = [segment[2], segment[3]]
    dx = x2 - x1
    dy = y2 - y1

    # Check segment steepness
    steep = abs(dy) > abs(dx)

    # Rotate if necessary
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap if necessary
    swap = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swap = True

    # New dx, dy
    dx = x2 - x1
    dy = y2 - y1

    # Error needed to determine threshold
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Bresenham's algorithm
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Swaps if necessary
    if swap:
        points.reverse()
    return points


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
        return True, (x, y)
    else:
        return False, None


def intersects_beam(segment, line1, line2):
    intersects1, _ = intersect_segment_line(segment, line1)
    intersects2, _ = intersect_segment_line(segment, line2)
    if intersects1 or intersects2:
        return True
    else:
        return False
    # a, b = [segment[0], segment[1]]
    # c, d = [segment[2], segment[3]]
    # segment = __segment2line(segment)
    # x0, y0 = intersect_lines(segment, line1)
    # print x0, y0
    # x1, y1 = intersect_lines(segment, line2)
    # print x1, y1
    # min_x = min(x0, x1)
    # max_x = max(x0, x1)
    # min_y = min(y0, y1)
    # max_y = max(y0, y1)
    #
    # if (a <= min_x and c <= min_x) or (a >= max_x and c >= max_x):
    #     return False
    # elif (b < min_y and d < min_y) or (b > max_y and d > max_y):
    #     print b, d, min_y, max_y
    #     return False
    # else:
    #     return True

# img = cv2.imread('images/aerial1/1726_p1_s.pgm')
# segment = np.asarray([0, 400, 100, 105])
# pts =  points_on_segment(segment)
# line1 = np.asarray([5.21, -4.321, -50])
# line2 = np.asarray([0.1, 10, -50])
#
# for p in pts:
#     cv2.circle(img, p, 0, (0,255,0), 2)
#
# draw_line(img, segment, (255, 0, 0), 1, LINE_FINITE)
# draw_line(img, line1, (255,0,0), 2, LINE_INFINITE)
# draw_line(img, line2, (255,0,0), 2, LINE_INFINITE)
# if intersects_beam(segment, line1, line2):
#     draw_line(img, segment, (255, 0, 0), 2, LINE_FINITE)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
