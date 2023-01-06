import numpy as np
import cv2
import imutils

def get_transformed_points(x, y, template, angle, scale):
    """
    
    Refactor code from the Alignment tool.
    
    """

    x_center_init = template.shape[1]//2
    y_center_init = template.shape[0]//2

    rotated_template = imutils.rotate_bound(template, angle)
    
    width = int(rotated_template.shape[1] * scale)
    height = int(rotated_template.shape[0] * scale)
    dim = (width, height)
  
    # resize image
    scaled_template = cv2.resize(rotated_template, dim)
    
    x_center_scaled = scaled_template.shape[1]//2
    y_center_scaled = scaled_template.shape[0]//2

    shifted_x = x-x_center_init
    shifted_y = y-y_center_init

    rot_mat = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))], 
                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

    shifted_x, shifted_y = rot_mat@np.array([shifted_x, shifted_y])
    
    shifted_x = shifted_x*scale
    shifted_y = shifted_y*scale

    shifted_x += x_center_scaled
    shifted_y += y_center_scaled
    
    return shifted_x, shifted_y

def vector_alignment(a1, a2, b1, b2):
    """
    
    Refactor code from the Alignment tool and following:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
    
    """
    
    v1 = a1-a2
    v2 = b1-b2
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Recover the scale
    scale = norm_v1/norm_v2

    v1_normalized = v1 / norm_v1 # normalize 
    v2_normalized = v2 / norm_v2 # normalize
    
    v_cross = np.cross(v1_normalized, v2_normalized) # gives sin(angle)
    v_dot = np.dot(v1_normalized, v2_normalized) # gives cos(angle)

    # Recover the angle
    #angle = -np.arccos(np.clip(v_dot, -1.0, 1.0)) # minus because counterclockwise
    #angle = np.sign((v1-v2)[1])*angle
    
    angle = -np.arctan2(v_cross, v_dot)
    
    scale=1
    # Recover the translation
    b1_scaled = b1*scale # need to scale
    b1_sc = np.array(rotate((0,0), b1_scaled, angle)) # then we rotate top-left corner
    translation_vector = a1-b1_sc
    
    H = np.array([[v_dot, v_cross, translation_vector[0]],
                  [-v_cross, v_dot, translation_vector[1]],
                  [0., 0., 1.]])
    
    return translation_vector, np.rad2deg(angle), scale, H



"""

! FOLLOWING CODE HAS BEEN TAKEN FROM THE ALIGNMENT TOOL PROJECT:

github: https://github.com/noe-d/AlignmentTool
wiki: http://fdh.epfl.ch/index.php/Alignment_of_XIXth_century_cadasters

"""

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

def warpTwoImages(img1, img2, H):
    """
    warp img2 to img1 with homograph H
    
    Code based on:
    from https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    
    Input:
        img1: 2D array (image)
            anchor image
        img2: 2D array (image)
            target image
        H: 2D array (3x3 matrix)
            homography to be applied on `img2` to wwarp it to `img1`
            
    Output:
        result: 2D array (image)
            Image resulting from the homography warping of img2 to img1
    """
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] += img1
    return result