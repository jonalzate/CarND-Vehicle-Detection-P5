import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
import random 
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from helperFunctions import draw_boxes, find_matches, color_hist, bin_spatial, data_look, get_hog_features, extract_features, slide_window, single_img_features, search_windows, add_heat, apply_threshold, draw_labeled_bboxes, get_hog_features2, extract_features2

# vehicle and Non-Vehicle template images 
vehicle_dir = './vehicles'
#print(vehicle_dir)
non_car_dir = './non-vehicles'
#print(non_car_dir)
car_img = []
noncar_img = []

for subdir, dirs, files in os.walk(vehicle_dir):
    for file in files:
        if file.endswith(".png"):
            full_path = os.path.join(subdir, file) 
            car_img.append(full_path)
            #print(os.path.join(subdir, file))
        
for subdir, dirs, files in os.walk(non_car_dir):
    for file in files:
        if file.endswith(".png"):
            full_path = os.path.join(subdir, file)
            noncar_img.append(full_path)
            #print(os.path.join(subdir, file))
            
            
print('car image templates = ', len(car_img))
print('not car image temlates = ',  len(noncar_img))
#test_car_img = mpimg.imread(car_img[random.randint(0, len(car_img))])
#test_noncar_img = mpimg.imread(noncar_img[random.randint(0, len(noncar_img))])
test_car_img = cv2.imread(car_img[random.randint(0, len(car_img))])
test_car_img = cv2.cvtColor(test_car_img, cv2.COLOR_BGR2RGB)
test_noncar_img = cv2.imread(noncar_img[random.randint(0, len(noncar_img))])
test_noncar_img = cv2.cvtColor(test_noncar_img, cv2.COLOR_BGR2RGB)


#cv2.imshow('test', cv2.cvtColor(test_car_img, cv2.COLOR_BGR2RGB))
#cv2.waitKey(0)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f.tight_layout()
#ax1.imshow(test_car_img)
#ax1.set_title('Car Image', fontsize=20)
#ax2.imshow(test_noncar_img)
#ax2.set_title('Non-Car Image', fontsize=20)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()
#plt.close(f)

# Test image for debugging functions
test_image = './test_images/test1.jpg'
#test_image = './vehicles/KITTI_extracted/1.png'

test_img = cv2.imread(test_image)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#test_img = mpimg.imread(test_image)
#plt.imshow(test_img)
#plt.show()

#============================================
# Test DRAW BOXES
bboxes = [((810, 400 ), (950, 500)), ((1050, 400), (1270, 510))]
color = (0, 255, 00)
thick = 3
bbox_test = draw_boxes(test_img, bboxes, color=color, thick=thick)

#cv2.imshow('tst', cv2.cvtColor(bbox_test, cv2.COLOR_BGR2RGB))
#cv2.waitKey(0)

# ============================================
# Test COLOR_HIST
rh, gh, bh, bincen, feature_vec = color_hist(test_car_img, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
#if rh is not None:
#    fig = plt.figure(figsize=(12,3))
#    plt.subplot(131)
#    plt.bar(bincen, rh[0])
#    plt.xlim(0, 256)
#    plt.title('R Histogram')
#    plt.subplot(132)
#    plt.bar(bincen, gh[0])
#    plt.xlim(0, 256)
#    plt.title('G Histogram')
#    plt.subplot(133)
#    plt.bar(bincen, bh[0])
#    plt.xlim(0, 256)
#    plt.title('B Histogram')
#    fig.tight_layout()
#    plt.show()
#else:
#    print('Your function is returning None for at least one variable...')

# ==============================================
# test SPATIAL_BIN
color_space = 'HSV'
size = (32, 32)

feature_vec = bin_spatial(test_car_img, color_space=color_space, size=size)

# Plot features
#plt.plot(feature_vec)
#plt.title('Spatially Binned Features')
#plt.show()

# ===============================================
# test HOG Features
gray = cv2.cvtColor(test_car_img, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)


# Plot the examples
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(test_car_img, cmap='gray')
#plt.title('Example Car Image')
#plt.subplot(122)
#plt.imshow(hog_image)
#plt.show()


#======================================================================
# test EXTRAC_FEATURES HOG
# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = car_img[0:sample_size]
notcars = noncar_img[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"

t=time.time()
car_features = extract_features2(cars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features2(notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)


#print(type(car_features))
#print(type(notcar_features))
#print(len(car_features))
#print(len(notcar_features))

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
#print('stack has type = ', type(X))
#print('stack has shape = ', X.shape)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
#n_predict = 10
#print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
#t2 = time.time()
#print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#================================================================================
#
#
# test slide window
windows = slide_window(test_img, x_start_stop=[None, None], y_start_stop=[400, 720], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(test_img, windows, color=(0, 0, 255), thick=6)                    
#plt.imshow(window_img)
#plt.show()


# ================================================================================
#
color_space = 'HSV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # can be 0, 1, 2 or ALL
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
y_start_stop = [400, 720]

draw_img = np.copy(test_img)
test_img = test_img.astype(np.float32)/255

windows = slide_window(test_img, x_start_stop=[None, None],
                       y_start_stop=y_start_stop, xy_window=(128, 128),
                       xy_overlap=(0.5, 0.5))

hot_windows = search_windows(test_img, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat,  hist_feat=hist_feat,
                            hog_feat=hog_feat)
window_img = draw_boxes(test_img, hot_windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()

