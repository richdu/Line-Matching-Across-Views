import numpy as np
import cv2
import fundamental_matrix_helpers as fmat
import line_helpers as lineh
import correlation_helpers as corrh
from matplotlib import pyplot as plt


# def drawlines(img, lines, pts):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     for r, pt in zip(lines,pts):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         cv2.circle(img1, (pt[0], pt[1]), 2, color, 2)
#         lineh.draw_line(img, r, color, 1, lineh.LINE_INFINITE)
#     return img

LINE_LENGTH_THRESHOLD = 15

# Read data
p1 = np.matrix(np.genfromtxt("data/house/house.000.P", delimiter=" "))
p2 = np.matrix(np.genfromtxt("data/house/house.001.P", delimiter=" "))
img1 = cv2.imread("images/house/house.000.pgm")
img2 = cv2.imread("images/house/house.001.pgm")
img1_single_channel = cv2.convertScaleAbs(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
img2_single_channel = cv2.convertScaleAbs(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

# Compute Fundamental Matrix
F = fmat.compute_fundamental(p2, p1)

# Detect and prune lines
LineSegmentDetector = cv2.createLineSegmentDetector()
lines1, _, _, _ = LineSegmentDetector.detect(img1_single_channel)
lines2, _, _, _ = LineSegmentDetector.detect(img2_single_channel)

lines1_pruned = [l[0] for l in lines1 if lineh.length(l[0]) > LINE_LENGTH_THRESHOLD]
lines2_pruned = [l[0] for l in lines2 if lineh.length(l[0]) > LINE_LENGTH_THRESHOLD]

# Selects line
current = lines1_pruned[100]
color = (255, 0, 0)
lineh.draw_line(img1, current, color, 2, lineh.LINE_FINITE)

# Retrieves endpoints
endpts1 = np.int32([(current[0], current[1]), (current[2], current[3])])

# Computes corresponding endpoints
epilines1 = cv2.computeCorrespondEpilines(endpts1, 2, F).reshape(-1, 3)
epiline11 = epilines1[0]
epiline12 = epilines1[1]

segment_pts = lineh.points_on_segment(map(int, current))

possible = []
for segment in lines2_pruned:
    endpts2 = np.int32([(segment[0], segment[1]), (segment[2], segment[3])])
    epilines2 = cv2.computeCorrespondEpilines(endpts2, 1, F).reshape(-1, 3)
    epiline21 = epilines2[0]
    epiline22 = epilines2[1]
    if lineh.intersects_beam(segment, epiline11, epiline12) or lineh.intersects_beam(current, epiline21, epiline22):
        possible.append(segment)

correlations = []
for segment in possible:
    num_success = 0
    sum = 0
    for p in segment_pts:
        epiline = cv2.computeCorrespondEpilines(np.int32([p]), 2, F).reshape(-1, 3)[0]
        is_intersection, p_prime = lineh.intersect_segment_line(segment, epiline)
        if is_intersection:
            p_prime = tuple(map(int, p_prime))
            num_success += 1
            sum += corrh.standard_correlation(img1_single_channel, img2_single_channel, p, p_prime, 7, 7)
    if num_success < 5:
        correlations.append(-1)
    else:
        correlations.append(sum/float(num_success))
correct_segment = possible[np.argmax(correlations)]
print np.max(correlations)
lineh.draw_line(img2, correct_segment, (255, 0, 0), 2, lineh.LINE_FINITE)


# pts = []
# for i in xrange(10):
#     color = tuple(np.random.randint(0, 255, 3).tolist())
#     r = np.random.randint(0, height)
#     c = np.random.randint(0, width)
#     pts.append((c, r))
#
# pts = np.int32(pts)
# # np.reshape(pts, (-1, 1, 2))
#
# # Lines on image 2
# lines = cv2.computeCorrespondEpilines(pts.reshape(-1,1,2), 2, F)
# lines = lines.reshape(-1,3)
# newimg = drawlines(img2, lines, pts)
#
plt.subplot(121),plt.imshow(img1)
plt.subplot(122),plt.imshow(img2)
plt.show()
