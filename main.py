import pickle
import os.path
import cv2
import tensorflow as tf     # only used for arg parsing
import numpy as np
from moviepy.editor import VideoFileClip

def calibrate_camera():
    import glob

    # number of points in calibration images
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return mtx, dist

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0,255)):

    # Apply the following steps to img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3)
    abs_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir > thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def gradient_threshold(image):

    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('output_images/gray.jpg',gray)
    #red_float = image[:,:,0].astype('float')
    #green_float = image[:,:,1].astype('float')
    #yellow_float = (red_float+green_float)/2
    #yellow_uint8 = yellow_float.astype('uint8')
    #gray = yellow_uint8
    #cv2.imwrite('output_images/yellow.jpg',yellow_uint8)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = hls[:,:,2]
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.blur(gray, (15, 15))
    out = cv2.Canny(gray, 0, 255)
    cv2.imwrite('output_images/lines.jpg',out)
    quit()
    #gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=15, thresh=(10, 255))
    #grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 255))
    #mag_binary = mag_thresh(gray, sobel_kernel=25, mag_thresh=(10, 255))
    #dir_binary = dir_threshold(gray, sobel_kernel=11, thresh=(0.5, 2.1))
    #combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #cv2.imwrite('output_images/'+file_name+'_gradx.jpg', 255*gradx)
    #cv2.imwrite('output_images/'+file_name+'_grady.jpg', 255*grady)
    #cv2.imwrite('output_images/'+file_name+'_mag_binary.jpg', 255*mag_binary)
    #cv2.imwrite('output_images/'+file_name+'_dir_binary.jpg', 255*dir_binary)
    #cv2.imwrite('output_images/'+file_name+'_combined.jpg', 255*combined)

    return out


def hls_threshold(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    bin_image = np.zeros_like(hls[:,:,2])
    bin_image[hls[:,:,2]>10] = 1

    return bin_image


def yellow_threshold(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    bin_image = np.zeros_like(image[:,:,0])
    #bin_image[(image[:,:,0]>140) & (image[:,:,2]<160) & \
    #          (hls[:,:,0]>19) & (hls[:,:,0]<50) & \
    #          (hls[:,:,1]>100) & (hls[:,:,1]<180) & \
    #          (hls[:,:,2]>40) & (hls[:,:,2]>40)] = 1
    bin_image[(image[:,:,0]>140) & (image[:,:,2]<170) & \
              (hls[:,:,0]>19) & (hls[:,:,0]<50) & \
              (hls[:,:,1]>100) & (hls[:,:,1]<180) & \
              (hls[:,:,2]>40) & (hls[:,:,2]>40)] = 1

    return bin_image


def white_threshold(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    bin_image = np.zeros_like(image[:,:,0])
    bin_image[(hls[:,:,1]>200)] = 1

    return bin_image


def color_threshold(image):

    #bin_hls = hls_threshold(image)
    bin_yellow = yellow_threshold(image)
    bin_white = white_threshold(image)
    combined_bin = np.zeros_like(bin_yellow)
    combined_bin[(bin_yellow==1) | (bin_white==1)] = 1

    return combined_bin

def combined_threshold(image):

    bin_grad = gradient_threshold(image)
    bin_color = color_threshold(image)
    combined_bin = np.zeros_like(bin_grad)
    combined_bin[(bin_grad==1) & (bin_color==1)] = 1

    return combined_bin

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.recent_fit = []
        self.n_smooth = 10

def sliding_windows(binary_warped, left=None, right=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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

    if left:
        if len(left.recent_fit)>=left.n_smooth:
            left.recent_fit.pop(0)
        left.recent_fit.append(left_fit)
        left.fit = left_fit = np.mean(left.recent_fit,0)

    if right:
        if len(right.recent_fit)>=right.n_smooth:
            right.recent_fit.pop(0)
        right.recent_fit.append(right_fit)
        right.fit = right_fit = np.mean(right.recent_fit,0)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    # Draw the lane onto the warped blank image
    # Create an output image to draw on and  visualize the result
    lane_area = np.dstack((binary_warped, binary_warped, binary_warped))*255
    pts = np.hstack((pts_left, pts_right))
    cv2.fillConvexPoly(lane_area, np.int_(pts), (0,255, 0))

    lane_lines = np.dstack((binary_warped, binary_warped, binary_warped))*255
    lane_lines[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    lane_lines[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    radius = (left_curverad+right_curverad)/2

    center = (left_fit_cr[0]*(y_eval*ym_per_pix)**2+left_fit_cr[1]*y_eval*ym_per_pix+left_fit_cr[2] + \
             right_fit_cr[0]*(y_eval*ym_per_pix)**2+right_fit_cr[1]*y_eval*ym_per_pix+right_fit_cr[2])/2

    position = 640*xm_per_pix-center

    return lane_area, lane_lines, radius, position


def draw_lane_area(lane_area, lane_lines, image, radius, position):

    unwarp_area = cv2.warpPerspective(lane_area, Minv, img_size)
    result = cv2.addWeighted(image, 1, unwarp_area, 0.3, 0)

    unwarp_lines = cv2.warpPerspective(lane_lines, Minv, img_size)
    result[(unwarp_lines[:,:,0]>0) & (unwarp_lines[:,:,1]==0)] = [255, 0, 0]
    result[(unwarp_lines[:,:,2]>0) & (unwarp_lines[:,:,1]==0)] = [0, 0, 255]

    #result[(unwarp_lines[:,:,1]>0)] = [255, 255, 255]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Radius of Curvature = {:.1f}(km)'.format(radius/1000),
                 (10,60), font, 2, (255,255,255), 2, cv2.LINE_AA)
    m = 'left of center' if position<0 else 'right of center'
    cv2.putText(result,'Position = {:.2f}(m) '.format(np.abs(position))+m,
                 (10,120), font, 2, (255,255,255), 2, cv2.LINE_AA)

    return result


def video_pipeline(image, left=Line(), right=Line()):

    undist = cv2.undistort(image, mtx, dist, None, mtx)
    warped_color = cv2.warpPerspective(undist, M, img_size)
    warped_bin = color_threshold(warped_color)
    lane_area, lane_lines, radius, position = sliding_windows(warped_bin, left, right)
    final = draw_lane_area(lane_area, lane_lines, undist, radius, position)

    return final

def test_pipeline(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    undist = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imwrite('output_images/'+file_name+'_1_undistort.jpg',cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))

    warped_color = cv2.warpPerspective(undist, M, img_size)
    cv2.imwrite('output_images/'+file_name+'_2_warped_color.jpg',cv2.cvtColor(warped_color, cv2.COLOR_BGR2RGB))

    warped_bin = color_threshold(warped_color)
    cv2.imwrite('output_images/'+file_name+'_3_warped_bin.jpg',np.stack((255*warped_bin,),axis=-1))

    lane_area, lane_lines, radius, position = sliding_windows(warped_bin)
    cv2.imwrite('output_images/'+file_name+'_4a_lane_area.jpg',lane_area)
    cv2.imwrite('output_images/'+file_name+'_4b_lane_lines.jpg',cv2.cvtColor(lane_lines, cv2.COLOR_BGR2RGB))

    final = draw_lane_area(lane_area, lane_lines, undist, radius, position)
    cv2.imwrite('output_images/'+file_name+'_5_final.jpg',cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

    #filled = np.zeros_like(undist)
    #cv2.fillConvexPoly(filled, src_pts.astype(int), (0,255,0))
    #filled = cv2.addWeighted(undist, 1., filled, 0.4, 0.)
    #cv2.imwrite('output_images/filled.jpg',filled)

    #rect = 200*warped.copy()
    #cv2.rectangle(rect, tuple(dst_up_left), tuple(dst_low_right), 255, thickness=3)
    #cv2.imwrite('output_images/rect.jpg',rect)


if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_bool('calibrate_camera', 'False', 't/f (re)calculate camera matrix and distortion coefficients')
    flags.DEFINE_string('test_pipeline', '', 'Filename of image to output pipeline stages')
    flags.DEFINE_string('video_file', 'project_video.mp4', 'Filename of video ')

    img_size = (1280, 720)

    src_low_left  = [231,  700]
    src_up_left   = [596,  451]
    src_up_right  = [684,  451]
    src_low_right = [1073, 700]
    src_pts = np.float32([src_low_left, src_up_left, src_up_right, src_low_right])
    dst_low_left  = [390, 720]
    dst_up_left   = [390,   0]
    dst_up_right  = [890,   0]
    dst_low_right = [890, 720]
    dst_pts = np.float32([dst_low_left, dst_up_left, dst_up_right, dst_low_right])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    if FLAGS.calibrate_camera or not os.path.isfile('camera_cal/dist.pkl'):
        print('Calibrating camera...')
        mtx, dist = calibrate_camera()
        dist_pickle = {'mtx':mtx, 'dist':dist}
        pickle.dump(dist_pickle, open('camera_cal/dist.pkl', 'wb'))
        del dist_pickle
        print('Finished')
    else:
        dist_pickle = pickle.load( open( "camera_cal/dist.pkl", "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        del dist_pickle

    if FLAGS.test_pipeline:
        img = cv2.imread(FLAGS.test_pipeline)
        file_name = FLAGS.test_pipeline.split('/')[1].split('.')[0]
        test_pipeline(img)
    else:
        video_output = FLAGS.video_file[:-4] + '_output' + FLAGS.video_file[-4:]
        clip = VideoFileClip(FLAGS.video_file)
        #clip = VideoFileClip(FLAGS.video_file).subclip(30,35)
        video_clip = clip.fl_image(video_pipeline)
        video_clip.write_videofile(video_output, audio=False)
