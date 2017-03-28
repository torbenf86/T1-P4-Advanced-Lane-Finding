from functions import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from moviepy.editor import VideoFileClip


# Step 1 Camera Calibration


try:
    mtx, dist = pickle.load(open("dist.p", "rb"))
    print("Undistortion matrix loaded...")
except (OSError, IOError) as e:
    mtx, dist = calibration(nx=9, ny=6)
    pickle.dump([mtx, dist], open("dist.p", "wb"))
    print("Undistortion matrix saved...")


def pipeline(img):

# Step 2 Distortion correction
    undistimg = undistort(img,mtx, dist)
    #visualize_distortion(img, undistimg)

# Step 3 Color/Gradient Threshold
    binary = thresh(undistimg, s_thresh=(150,255), sx_thresh=(40,100))  #190, 255, 70,100  bzw. 20 100
    #binary = thresh2(undistimg, thresh=(20, 40), mag_thresh=(30, 90), thresh_dir=(0.7, 1.3))
    #visualize_thresholding(undistimg, binary)

# Step 4 Perspective Transform

    # define 4 source points
    top_left = [585, 460]  # [600, 440]
    bottom_left = [203, 720]  # [290,650]
    bottom_right = [1127, 720]  # [1000, 650]
    top_right = [695, 460]  # [670,440]
    src_points = np.float32([top_left,bottom_left, bottom_right,top_right])
    # define 4 destination points
    dst_top_left = [320, 0]  # [290, 100]
    dst_bottom_left = [320, 720]  # [290,650]
    dst_bottom_right = [960, 720]  # [1000, 650]
    dst_top_right = [960, 0]  # [1000,100]
    dst_points = np.float32([dst_top_left, dst_bottom_left, dst_bottom_right, dst_top_right])

    warped, M, M_inv = perspective_transform(undistimg, src_points, dst_points)
    #visualize_transform(undistimg, warped, src_points, dst_points)

# Step 5 Detect lane lines
    binary_warped = cv2.warpPerspective(binary, M, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_LINEAR)
    #visualize_thresholding(warped, binary_warped)


    left_fit,right_fit = detect_lane_lines(binary_warped, line_right, line_left)
    #visualize_lanes(binary_warped, left_fit, right_fit)

# Step 6 Determine the lane curvature
    left_curverad, right_curverad = curvature(img, left_fit, right_fit)

# Step 7 Visualization
    result = visualize_final(undistimg, binary_warped, left_fit, right_fit, M_inv, left_curverad, right_curverad)


    return result

# Visualization chessboard

#img = mpimg.imread("camera_cal/calibration1.jpg")
#visualize_chessboard_distortion(img, mtx, dist)

# Creating Lines Object
line_right = Line()
line_left = Line()

# Standalone

#img = mpimg.imread("test_images/straight_lines1.jpg")
#result = pipeline(img)
#plt.imshow(result)
#plt.show()


# Video
output = 'result.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(pipeline)
clip.write_videofile(output, audio=False)
