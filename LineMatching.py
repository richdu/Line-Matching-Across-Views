import numpy as np
import cv2
import fundamental_matrix_helpers as fundamentals
import line_helpers as lines
import correlation_helpers as correlations
from matplotlib import pyplot as plt

# Constants
LINE_LENGTH_THRESHOLD = 15
LINE_MATCH_CORRELATION_THRESHOLD = 0.85

# Read data
p1 = np.matrix(np.genfromtxt("data/house/house.002.P", delimiter=" "))
p2 = np.matrix(np.genfromtxt("data/house/house.003.P", delimiter=" "))
img1 = cv2.imread("images/house/house.002.pgm")
img2 = cv2.imread("images/house/house.003.pgm")
img1_single_channel = cv2.convertScaleAbs(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
img2_single_channel = cv2.convertScaleAbs(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

# Compute Fundamental Matrix
F = fundamentals.compute_fundamental(p2, p1)

# Detect lines
LineSegmentDetector = cv2.createLineSegmentDetector()
lines1, _, _, _ = LineSegmentDetector.detect(img1_single_channel)
lines2, _, _, _ = LineSegmentDetector.detect(img2_single_channel)

lines1_pruned = [l[0] for l in lines1 if lines.length(l[0]) > LINE_LENGTH_THRESHOLD]
lines2_pruned = [l[0] for l in lines2 if lines.length(l[0]) > LINE_LENGTH_THRESHOLD]

matches = []
# Loops through line segments in first image
for current in lines1_pruned:
    # Retrieves endpoints
    endpoints1 = np.int32([(current[0], current[1]), (current[2], current[3])])

    # Computes corresponding endpoints
    epilines1 = cv2.computeCorrespondEpilines(endpoints1, 2, F).reshape(-1, 3)
    epiline11 = epilines1[0]
    epiline12 = epilines1[1]

    # Compute points on
    segment_pts = lines.points_on_segment(map(int, current))

    # Looks for possible points using the concept of the epipolar beam
    possible = []
    for segment in lines2_pruned:
        endpoints2 = np.int32([(segment[0], segment[1]), (segment[2], segment[3])])
        epilines2 = cv2.computeCorrespondEpilines(endpoints2, 1, F).reshape(-1, 3)
        epiline21 = epilines2[0]
        epiline22 = epilines2[1]
        if lines.intersects_beam(segment, epiline11, epiline12) or lines.intersects_beam(current, epiline21, epiline22):
            possible.append(segment)

    # Computes correlations between current segment and all segments in the other image
    segment_correlations = []
    for segment in possible:
        num_success = 0
        current_sum = 0
        for p in segment_pts:
            epiline = cv2.computeCorrespondEpilines(np.int32([p]), 2, F).reshape(-1, 3)[0]
            is_intersection, p_prime = lines.intersect_segment_line(segment, epiline)
            if is_intersection:
                p_prime = tuple(map(int, p_prime))
                num_success += 1
                current_sum += correlations.standard_correlation(img1_single_channel, img2_single_channel, p, p_prime, 7, 7)
        # Ignores segment if too few points in common
        if num_success < 5:
            segment_correlations.append(-1)
        else:
            segment_correlations.append(current_sum / float(num_success))

    if segment_correlations:
        correct_segment = possible[np.argmax(segment_correlations)]
        # Only matches above a certain threshold are included
        if np.max(segment_correlations) > LINE_MATCH_CORRELATION_THRESHOLD:
            matches.append([current, correct_segment])
            color = tuple(np.random.randint(128, 255, 3).tolist())
            lines.draw_line(img1, current, color, 2, lines.LINE_FINITE)
            lines.draw_line(img2, correct_segment, color, 2, lines.LINE_FINITE)

# Plots images
plt.subplot(121), plt.imshow(img1)
plt.subplot(122), plt.imshow(img2)
plt.show()
