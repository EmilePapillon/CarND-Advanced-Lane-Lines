
---

**Advanced Lane Finding Project**

Finding lane markings is essential for self-driving cars to operate properly on a road. The following presents an algorithm to achieve such tracking using color thresholding, sliding windows and polynomial fitting. The result is a description of the lane marking that we can use to do path planning or control the vehicle. The results here is used to plot the lane marking over the image. 

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_image.png "Original"
[image2]: ./output_images/undistorted_image.png "Undistorted"
[image3]: ./output_images/thresholded_binary_image.png "Binary Example"
[image4]: ./output_images/source_and_destination.png "Warp Example"
[image5]: ./output_images/warped_result.png "Fit Visual"
[image6]: ./output_images/unwarped_result.png "Output"
[image7]: ./output_images/curvature.png "Curvature"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Lane marking detection using sliding windows.

### Camera Calibration

#### 1. First, compute the camera matrix and distortion coefficients. Use the result to generate a distortion corrected calibration image.

The camera image comes distorted by the lens, and this will be a problem for accurate lane marking detection. We first need to correct the image for distorsion using openCV. Refer to the Lane_Marking_Tracker.ipynb to follow along with the code. 

The code for this step is contained in the second code cell of the Jupyter notebook located in "./Lane_Marking_Tracker.ipynb". The function calibrate(calibration_images_path='camera_cal/cal*.jpg') takes in as argument a string containing the glob wildcard to the calubration data. It returns the matrix mtx and the object dist to use with openCV's cv2.undistort() function to get the undistorted image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]


#### 2. Use color transforms, gradients or other methods to create a thresholded binary image. Below is an example of a binary image result.

The first step is to get rid of some noise by applying color space thresholding to the original frames.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the third code cell of the Jupyter notebook located in "./Lane_Marking_Tracker.ipynb".  Here's an example of my output for this step. 

![alt text][image3]

I used sobel directional threshold, and HLS color space thresholding using the L and S layers because I found them to be the most effectives ones at selecting lane markings. S (saturation) layer information is particularly useful for the yellow lines. 

#### 3. Then, perform a perspective transform and (example of a transformed image below).

We are trying to get the curvature information from the lane marking. To do this, we first need to transform the image to abstract the perspective inherent to camera images.

The code for my perspective transform includes a function called `warper()`, which appears in cell 4 in the Jupyter Notebook `Lane_Marking_Tracker.ipynb` (./Lane_Marking_Tracker.ipynb).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    img_size = binary_image.shape[::-1]
    src= np.float32([[600,450],
        [700,450],
        [200,img_size[1]],
        [1200,img_size[1]]])
    
    dx = img_size[0] / 5
    dy=5
    dst = np.float32([[0+dx,0+dy],
        [img_size[0]-dx,0+dy],
        [0+dx,img_size[1]-dy],
        [img_size[0]-dx,img_size[1]-dy]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 256, 5        | 
| 700, 450      | 1024,5        |
| 200, 720      | 256, 715      |
| 1200, 720     | 1024, 715     |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

In order to fit the lane lines with a 2nd order polynomial, I first used a sliding windows approach that used a histogram technique to locate the initial windows. When I had a good idea of the location of the lane lines by applying a first iteration of the sliding windows, I limited my search to a region around the known lane lines. To reduce the jitter of the lane lines, I used a moving average of correctly fitted lines over 15 frames of the video. This was all implemented in cells 5 and 6 of the Jupyter Notebook at ./Lane_Marking_Tracker.ipynb

The result of the fitting was the following : 

![alt text][image5]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the first cell in my code in `./Lane_Marking_Tracker.ipynb`
The Line class implements private methods for calculating curvature and distance to lane marking. Those methods are `__calculate_curvature(self):` and `__calculate_distance(self)`, respectively. Those methods are called every time a new line marking is written into the line object using the set_line(self, pointsx, pointsy) setter method and passing the x and y points corresponding to the lane lines found. The Line object fits the line points and calculate the curvature using the curvature formula :
![alt text][image7]

The distance to the lane lines was found by taking the x value corresponding to the minimum y in the x-y data of the current line marking for each frame.

To make things look better, curvature and distance were smoothened using a mooving average on 15 frames before being displayed.

#### 6. Example of the result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the ninth line in my code in `./Lane_Marking_Tracker.ipynb` in the function `image_pipeline(frame,left_marker,right_marker, debug=False)` at the end.  In fact, this is where the whole pipeline is put togheter and the code to unwarp was not encapsulated although it would've made sense to do so. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Final video output.  

Here's a [link to my video result](./output_images/video_output.mp4)

---

### Discussion

I used basic thresholding techniques for identifying the lane markings. My pipeline did not do well for the more challenging videos, more especifically where there is a strong contrast between 2 tones of asfalt. The pipeline will identify the interface between the 2 types of asfalt as a lane marking. If I had more time I would implement a refined version of the thresholding to filter out these kind of issues. 
