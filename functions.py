import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.image as mpimg


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]


def calibration(nx=9, ny=6):
    imgpoints = []
    objpoints = []
    images = glob.glob("camera_cal/calibration*.jpg")
    #Prepare object points
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        print( fname + " done ...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def visualize_distortion(img_original, img_undistort):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_original)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(img_undistort)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def thresh(img, s_thresh, sx_thresh):
    img = np.copy(img)
    # Convert to LAB and LUV color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV).astype(np.float)
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = luv[:, :, 0]
    b_channel = lab[:, :, 2]
    # Sobel x l channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(b_channel)
    s_binary[(b_channel >= s_thresh[0]) & (b_channel <= s_thresh[1])] = 1
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    binary_combined = np.zeros_like(s_binary)
    binary_combined[((sxbinary == 1) | (s_binary == 1))] = 1

    return binary_combined

def thresh2(img, thresh, mag_thresh, thresh_dir):

    sobel_kernel = 3
        # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x and y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
    gradx = np.zeros_like(scaled_sobelx)
    grady = np.zeros_like(scaled_sobely)
    gradx[(scaled_sobelx >= thresh[0]) & (scaled_sobelx <= thresh[1])] = 1
    grady[(scaled_sobely >= thresh[0]) & (scaled_sobely <= thresh[1])] = 1

    #plt.imshow(gradx)
    #plt.show()
    #plt.imshow(grady)
    #plt.show()



        # 1) Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 2) Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
        # 3) Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    #plt.imshow(mag_binary)
    #plt.show()
        # 4) Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh_dir[0]) & (absgraddir <= thresh_dir[1])] = 1

    binary_combined = np.zeros_like(dir_binary)
    binary_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return binary_combined


def visualize_thresholding(img_original, result):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img_original)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result, cmap="gray")
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def perspective_transform(img_undist, src, dst):
    src_size = (img_undist.shape[1],img_undist.shape[0])
    #  use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    #  use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img_undist, M, src_size, flags=cv2.INTER_LINEAR)

    return warped, M, M_inv

def visualize_transform(img_undist, img_warped, src, dst):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.plot(src[0:2, 0], src[0:2, 1], color = 'red', marker='o')
    ax1.plot(src[2:4, 0], src[2:4, 1], color='red', marker='o')
    ax1.imshow(img_undist)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(img_warped, cmap="gray")
    ax2.set_title('Warped Result', fontsize=40)
    ax2.plot(dst[0:2, 0], dst[0:2, 1], color = 'red', marker='o')
    ax2.plot(dst[2:4, 0], dst[2:4, 1], color='red', marker='o')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def validate_lines(leftx, left_fit, rightx, right_fit):
    # Calculate slope in mean x
    slope_diff = np.abs((2*left_fit[0]*np.mean(leftx) + left_fit[1]) - (2*right_fit[0]*np.mean(rightx) + right_fit[1]))

    # Calculate mean distance
    distance = np.mean(leftx) - np.mean(rightx)
    is_distance_ok = 500 < distance < 800
    is_parallel = 0 < slope_diff < 0.2
    if (is_distance_ok & is_parallel):
        return True
    else:
        return False


def detect_lane_lines(binary_warped, line_right, line_left):
    if (line_right.detected & line_left.detected):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 80
        left_lane_inds = (
        (nonzerox > (line_left.best_fit[0] * (nonzeroy ** 2) + line_left.best_fit[1] * nonzeroy + line_left.best_fit[2] - margin)) & (
        nonzerox < (line_left.best_fit[0] * (nonzeroy ** 2) + line_left.best_fit[1] * nonzeroy + line_left.best_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (line_right.best_fit[0] * (nonzeroy ** 2) + line_right.best_fit[1] * nonzeroy + line_right.best_fit[2] - margin)) & (
        nonzerox < (line_right.best_fit[0] * (nonzeroy ** 2) + line_right.best_fit[1] * nonzeroy + line_right.best_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        line_left.best_fit = left_fit
        line_right.best_fit = right_fit
        if validate_lines(leftx, lefty, rightx, righty):
            line_left.detected = True
            line_right.detected = True
        else:
            line_left.detected = False
            line_right.detected = False


    else:

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, 50:1150], axis=0)
        #plt.plot(histogram)
        #plt.show()

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
             # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Update objects
        line_left.best_fit = left_fit
        line_right.best_fit = right_fit

        if validate_lines(leftx, lefty, rightx, righty):
            line_left.detected = True
            line_right.detected = True
        else:
            line_left.detected = False
            line_right.detected = False


    return left_fit, right_fit


def visualize_lanes(img, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    plt.imshow(img, cmap="gray")
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def curvature(img, left_fit, right_fit):

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])


    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 650  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate center of lane
    left_bottom = left_fit[0] * img.shape[0] ** 2 + left_fit[1] * img.shape[0] + left_fit[2]
    right_bottom = right_fit[0] * img.shape[0] ** 2 + right_fit[1] * img.shape[0] + right_fit[2]
    lane_size_px = right_bottom - left_bottom
    lane_center_px = lane_size_px / 2 + left_bottom
    distance_to_center_m = np.absolute(lane_center_px-img.shape[1]/2)*(3.7 / lane_size_px)


    return left_curverad, right_curverad, distance_to_center_m


def visualize_final(undist, warped, left_fit, right_fit, M_inv, left_curverad, right_curverad, distance_to_center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(result, 'Left curvature radius: %.2f m' % left_curverad , (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, 255 )
    cv2.putText(result, 'Right curvature radius: %.2f m' % right_curverad, (200, 100),cv2.FONT_HERSHEY_SIMPLEX,1, 255)
    cv2.putText(result, 'Distance to lane center: %.2f m' % distance_to_center, (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    #plt.imshow(result)
    #plt.show()
    return result

def  visualize_chessboard_distortion(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

