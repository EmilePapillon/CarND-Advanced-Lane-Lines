
---

**Advanced Lane Finding Project**

Finding lane markings is essential for self-driving cars to operate properly on a road. The following presents an algorithm to achieve such tracking using color thresholding, sliding windows and polynomial fitting. The result is a description of the lane markings that we can use to do path planning or control the vehicle. The results here is used to plot the lane marking over the image. 

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

[image1]: ./output_images/calibration1.jpg "Original"
[image2]: ./output_images/calibration2.jpg "Undistorted"
[image3]: ./output_images/thresholded_binary_image.png "Binary Example"
[image4]: ./output_images/source_and_destination.png "Warp Example"
[image5]: ./output_images/warped_result.png "Fit Visual"
[image6]: ./output_images/unwarped_result.png "Output"
[image7]: ./output_images/curvature.png "Curvature"

---

### Lane marking detection using sliding windows.

### Camera Calibration

#### 1. First, compute the camera matrix and distortion coefficients. Use the result to generate a distortion corrected calibration image.

The camera image comes distorted by the lens, and this will be a problem for accurate lane marking detection. There are two types of distorsion, **angular** distorsion and **tangential** distorsion. 

![alt text][image1]

Radial distorsion make image appear more curved thant they actually are. This is due to lens curvature.. 

![alt text][image2]

Tangential distorsion is when lines that are really parallel seems to be converging on an image. This happens because the camera lens is not aligned perfectly parallel to the imaging plane, where the camera sensor or film  is. 

Distorsion can be corrected using a transformation matrix. OpenCV can be used in conjunction with calibration images (usually of a chessboard) to obtain this calibration matrix. 

We first need to correct the image for distorsion and will do so using openCV. This will be achieved using the `calibrate()` function.

In the first part of this function, we create a Numpy array that will represent what the undistorted grid should look like : the corners of every square in the grid can be described by an array of equally spaced points accross the x and y axis. This grid is of dimension 9 x 6 as the chessboard that we will use for the camera calibration. 

Finally, a container array is created for object points - the points in a theoretical undistorted image points will be stored under the variable name `objpoints` and for image points, the actual coordinates of the sqares corner in images, under the variable name `imgpoints`. We will later append points to those arrays. The `images` is a glob object with all the files containing our calibration grids : 

``` python
def calibrate(calibration_images_path='camera_cal/cal*.jpg'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibration_images_path)
```

In the next section of code, we will run through all our calibration files and call the `ct2.findChessboardCorners()` function to return the coordinate of the distorted chessboard square corners. If those points are found by the function, we will use that file for the calibration, so we append it to the container arrays. 

``` python
    for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

```

In the next code block, we finally calculate a calibration matrix using `cv2.calibrateCamera()` and passing our ground truth and real images. 

```python
img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return ret, mtx, dist
```
The following function simply uses the matrix we computed previously to undistort any image taken using the same camera as the one used to calculate the matrix. 

```python
def undistort(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst
```

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
