import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

def dist_vector(v1,v2):
    x1,y1=v1
    x2,y2=v2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def dist_vector_pair(l1,l2):
    v11,v12=l1
    v21,v22=l2
    d1=dist_vector(v11,v21)
    d2=dist_vector(v12,v22)
    return (d1+d2)/2

def dilate_resize(im1, im2, kernel1, kernel2, resize):

    im1_process = dilate(im1, kernel1)
    im2_process = dilate(im2, kernel2)

    width = int(im1_process.shape[1]//resize)
    height = int(im1_process.shape[0]//resize)
    im1_process = cv2.resize(im1_process, (width, height))

    width = int(im2_process.shape[1]//resize)
    height = int(im2_process.shape[0]//resize)
    im2_process = cv2.resize(im2_process, (width, height))
    
    return im1_process, im2_process

def dilate_resize_im(im, kernel, resize):

    im_process = dilate(im, kernel)

    width = int(im_process.shape[1]//resize)
    height = int(im_process.shape[0]//resize)
    im_process = cv2.resize(im_process, (width, height))
    
    return im_process

def erode_resize(im1, im2, kernel1, kernel2, resize):

    im1_process = erode(im1, kernel1)
    im2_process = erode(im2, kernel2)

    width = int(im1_process.shape[1]//resize)
    height = int(im1_process.shape[0]//resize)
    im1_process = cv2.resize(im1_process, (width, height))

    width = int(im2_process.shape[1]//resize)
    height = int(im2_process.shape[0]//resize)
    im2_process = cv2.resize(im2_process, (width, height))

    return im1_process, im2_process

def get_color_list(index):
    
    color_list = [
        'blue',
        'green',
        'red',
        'cyan',
        'magenta',
        'yellow',
        'teal',
        'orange',
        'plum',
        'olive',
        'brown',
        'indigo',
        'hotpink',
        'lime',
        'grey',
        'turquoise',
        'tomato',
        'deepskyblue',
        'pink',
        'greenyellow',
    ]
    
    c = color_list[index%len(color_list)]
    
    return colors.to_rgb(c)

def pltcolor_to_cv2(color):
    return (color[0]*255, color[1]*255, color[2]*255)

def convert_to_rgb(image, r_channel, g_channel, b_channel):
    """
    Convert a binary image of shape (h, w) into a RGB image of shape (h, w, 3).
    """
    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Make True pixels red
    mask = image > 0
    image_rgb[mask]  = [r_channel, g_channel, b_channel]
    image_rgb[~mask] = [255,255,255]
    
    return image_rgb

def plt_plot_cv2(image,ax=plt) :
    """
    Function used to plot images
    """
    ax.imshow(image.max()-image, cmap="Greys")
    
def dilate(img, kernel=np.ones((8, 8), np.uint8)):    
    return cv2.dilate(img, kernel, iterations=1)

def erode(img, kernel=np.ones((8, 8), np.uint8)):
    return cv2.erode(img, kernel, iterations=1)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    
    Input:
        origin: tuple (2D coordinate)
            origin of the rotation
        point: tuple (2D coordinate)
            coordinates of the point to rotate
        angle: float
            angle of the rotation —> should be given in radian
            
    Output:
        qx, qy: float, float
            qx: rotated coordinate in x
            qy: rotated coordinate in y 
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

########################################################################
############## HEATMAP BUILD HELPER ####################################

def get_contour_point_at(img, theta, flag_border):    
    """
    Help using:
    https://stackoverflow.com/questions/4061576/finding-points-on-a-rectangle-at-a-given-angle
    
    """
    
    twoPI = np.pi* 2.0
    PI = np.pi
    
    theta=theta%twoPI
    
    h_anchor, w_anchor = img.shape

    x_anchor_center = w_anchor/2
    y_anchor_center = h_anchor/2

    rectAtan = np.arctan2(h_anchor,w_anchor)
    tanTheta = np.tan(theta)

    xFactor = 1
    yFactor = 1
    
    # determine regions
    if theta > twoPI-rectAtan or theta <= rectAtan:
        region = 1
    elif theta > rectAtan and theta <= PI-rectAtan:
        region = 2
    elif theta > PI - rectAtan and theta <= PI + rectAtan:
        region = 3
        xFactor = -1
        yFactor = -1
    elif theta > PI + rectAtan and theta <= twoPI - rectAtan:
        region = 4
        xFactor = -1
        yFactor = -1
    else:
        print(f"region assign failed : {theta}")
        raise
    
    edgePoint = [0,0]
    ## calculate points
    if (region == 1) or (region == 3):
        edgePoint[0] += xFactor * (w_anchor / 2.)
        edgePoint[1] += yFactor * (w_anchor / 2.) * tanTheta
    else:
        edgePoint[0] += xFactor * (h_anchor / (2. * tanTheta))
        edgePoint[1] += yFactor * (h_anchor /  2.)
        
    edgePoint[0] += w_anchor / 2.
    edgePoint[1] += h_anchor / 2.
    
    # Coordinate point on the border of the image:
    x_target = edgePoint[0]
    y_target = edgePoint[1]

    # define the line joining the center of the anchor and the target point
    N = int(np.hypot(x_target-x_anchor_center, y_target-y_anchor_center))
    x_theta = np.linspace(x_anchor_center, x_target-1, N).astype(np.int32)
    y_theta = np.linspace(y_anchor_center, y_target-1, N).astype(np.int32)

    # compute the image values along the line
    zi_theta = img[y_theta, x_theta].copy()
    
    if flag_border:
        # find last pixel along the line from center to image's edge
        # pixel on the border of the image
        last_edge_theta = np.where(zi_theta==0)[0][-1]
    else:
        # find last edge along the line from center to image's edge
        last_edge_theta = np.where(zi_theta>0)[0][-1]
        
    x_temp, y_temp = x_theta[last_edge_theta], y_theta[last_edge_theta]
    
    return x_temp, y_temp

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    x1, y1 = v1
    x2, y2 = v2
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    return (np.arctan2(det, dot))

def build_heatmap(image):
    """ Build the heatmap of the distance to the outtermost edges of the images.  """

    size_y, size_x = image.shape
    x_img_center = size_x//2
    y_img_center = size_y//2
    
    heatmap = np.zeros(image.shape)

    vec_im_center_x = 1
    vec_im_center_y = 0

    for x in range(size_x):
        for y in range(size_y):

            vec_x = x - x_img_center
            vec_y = y - y_img_center


            angle_patch_transformed = angle_between((vec_im_center_x, vec_im_center_y), (vec_x, vec_y))

            if x==x_img_center and y==y_img_center:
                angle_patch_transformed=0

            contour_point_x, contour_point_y = get_contour_point_at(image.copy(), angle_patch_transformed, True)

            dist = np.linalg.norm((contour_point_x-x, contour_point_y-y))
        
            heatmap[y, x] = dist
    return heatmap

def build_heatmap_plot(image, tresh_dist):
    """ Same as build_heatmap, but with plot figure possibility """

    size_y, size_x = image.shape
    x_img_center = size_x//2
    y_img_center = size_y//2
    
    source=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    source=cv2.circle(source, (int(x_img_center), int(y_img_center)), radius=5, color=(0, 255, 0), thickness=-1)
    
    heatmap = np.zeros(image.shape)

    vec_im_center_x = 1
    vec_im_center_y = 0

    for x in range(size_x):
        for y in range(size_y):

            vec_x = x - x_img_center
            vec_y = y - y_img_center


            angle_patch_transformed = angle_between((vec_im_center_x, vec_im_center_y), (vec_x, vec_y))

            if x==x_img_center and y==y_img_center:
                angle_patch_transformed=0

            contour_point_x, contour_point_y = get_contour_point_at(image.copy(), angle_patch_transformed, True)

            dist = np.linalg.norm((contour_point_x-x, contour_point_y-y))
        
            heatmap[y, x] = dist
            
            #source=cv2.circle(source, (int(contour_point_x), int(contour_point_y)), radius=5, color=(255, 0, 0), thickness=-1)
            if dist < tresh_dist:
                source=cv2.circle(source, (int(x), int(y)), radius=2, color=(31, 118, 180), thickness=-1)
    return heatmap, source