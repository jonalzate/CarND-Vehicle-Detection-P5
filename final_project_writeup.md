# Vehicle Detection Project


**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_features.png
[image3]: ./output_images/sliding_fullwindow_search.png
[image4]: ./output_images/sliding_halfwindow_search.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_proc.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function called get_hog_features. To calculate these values I used the Hog function from the skimage.feature framework, that takes the following arguments, the image, orientations, pixels per cell, cells per block, feature vector and visualization flag in case we want to obtain the visualization image.  



I made a function (get_car_notcar_imgs()) for reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and different color spaces to train the classifier but in the end I noticed a better performance using the LUV color space. It got less false positives than others.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM by first extracting the features from the car and notcar images and stacking them in a array to feed them to the classifier.I then splitted the data into train and test data and proceeded to fit the classifier and achieved a 98.83% in accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Basic sliding window algoritm was implemented in the same way as presented in course lectures (See the code under slide_window function). It allows to search a car in a desired region of the frame with a desired window size 

The window size and overlap should be carefully selected. Size of the window should be compared to the size of an expected car. These parameters were set to mimic perspective.

Here are some sample results for a fixed window size of 128x128 pixels and overlap for the provided test images:


![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

*As we can see on examples above, the classifier successfully finds cars on the test images. However, there is a false positives, so we will need to apply some kind of filter (such as heat map). The classifier also failed to find a car on th 3rd image may be due to the size of the car in the image. That is why, we will probably need to use multi scale windows.

* To increase the classifier's accuracy, several feature extraction parameters were tuned, like the color space.

* To reduce number of false positives a heatmap with a threshold approach was implemented in the same to the suggested in the lectures. For video the heatmap is accumulated by two frames which reduces number of outliers false positives.

* To increase performance we need to analize the smallest possible number of windows. That is why, one can scan with a search window not across the whole image, but only areas where a new car can appear and also we are going to scan areas where a car was detected. Giving different x and y starting positions at slide_window also helped to avoid searching for cars in the top (sky, tress) portion of the image

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

*The pipeline may fail in case of difficult light conditions, which could be partly resolved by optimizing the classifier.

*It is possible to improve the classifier by additional data augmentation, hard negative mining, classifier parameters tuning etc.

*The pipeline has some potential problems in case of car overlaps another. To deal this problem one may introduce long term memory of car position and a kind of predictive algorithm which can predict where occluded car can be/appear and where it is worth to look for it.

*To eliminate false positives on areas out of the road, one can deeply combine results from the Advanced Lane Line finding project to correctly determine the wide ROI on the whole frame by the road boundaries. Unfortunately, it was not correctly implemented (just hard coded, which is enought for the project but not a good implementation for a real-world application) due to time limitation.

*The pipeline is not a real-time and this approach seems to take a lot of work. I was thinkin of maybe using a neural network instead. I tried to played around with YOLO network from darknet and it is way faster and it process realtime. So, I  personally spend some work training a DNN than using this approach. But it was definitely a nice experience and I learned a lot about openCV and computer vision approaches for feature extraction that are also helpful when preparing data for DNN training.
