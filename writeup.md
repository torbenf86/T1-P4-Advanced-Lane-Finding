
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortion_chessboard.jpeg "Undistorted"
[image2]: ./output_images/undistortion.jpeg "Road Transformed"
[image3]: ./output_images/pipeline2.jpeg "Binary Example"
[image4]: ./output_images/warped.jpeg "Warp Example"
[image5]: ./output_images/lane_detection2.jpeg "Fit Visual"
[image6]: ./output_images/output2.jpeg "Output"
[image7]: ./output_images/warped_binary2.jpeg "Binary Warp Example"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 16 through 34 of the file called `functions.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The obtained distortion matrix is now applied to an image of the road.
The result shows a comparison between the orginal and the undistorted image: 
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
First, I converted the image to LAB - color space and LUV color space. Then, I used a combination of color (threshold: (150,255)) and gradient thresholds (threshold: (40,100)) to generate a binary image (thresholding steps at lines 50 through 74 in `functions.py`). 
I used the B channel in LAB for the color thresholding and the L channel in LUV for the gradient thresholding. 
Here's an example of my output for this step. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is inside the function called `perspective_transform()`, which appears in lines 143 through 151 in the file `functions.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
Below you can see the parallel lines in the warped image, and the warped binary image.

![alt text][image4]
![alt text][image7]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After the perspective transformation of the thresholded binary image, I identified the lane-line pixels using the sliding windows approach.
In a first step a histogram of the lower half of the image is taken. This is used to identify the starting points of the right and left lane. This is only done once, if the lanes have not been detected in a former step (for e.g. the found lines are not validated or it is the first frame).
To get the lane line pixels, the sliding window approach is used, which consists in this case of a total number of nine windows. By using the starting point, the lane line pixels are located in the next window within a margin of 100px. After detecting all lane line pixels a quadratic function is fitted through the data points by using numpy.polyfit().
The fitted function is plotted in the image below:

![alt text][image5]

The code can be found in  lines 184-311 in `functions.py`.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 327 through 361 in my code in the function curvature() in `functions.py`.
The formula for calculating the curvature is as follows : R = ​∣2A∣​​(1+(2Ay+B)​^2​​)​^(3/2​​​​)
Finally the radius has to be transformed from pixel space to real world space by using the given constants.

To determine the position of the vehicle with respect to the lane center, I used the identified lane lines and calculated their value at y=y_max yielding the position of the lane center
It can be assumed, that the position of the vehicle is in the middle of the frames, which corresponds to a value of x=640.
The position is then the absolute value of the difference between the lane center and the middle of the frame, corrected by the conversion factor between pixel and real world space.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 364 through 390 in the function visualize_final() in `functions.py`.  
The detected lane lines are used to draw a lane in the warped image by using cv2.fillPoly(). The function cv2.putText() is used to add the calculated curvature of the right and left lane line and the distance of the vehicle to the lane center.
Finally, the warped image is unwarped back to the original image by using the inverted perspective matrix Minv.
Here is an example of my result on a test image:

![alt text][image6]

---
###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4)
 
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The algorithm is pretty reliable, but found its limits, if the captured images are too much shaded when using only th HLS color space. As a consequence I switched to the LAB and LUV color spaces, which treat light changes significantly better. Also, the starting points are identified by using a histogram. 
This could lead to a false starting point, if other parts than the lane lines are dominant in the image. 
A plausibility check is in this case necessary (for example if the distance of the lanes is plausible). This check is implemented in line 170 through 181 in `functions.py`, but it does not yet decide, which peak of the histogram is the right one.


