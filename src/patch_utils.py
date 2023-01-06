import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

from images_utils import plt_plot_cv2, get_color_list, pltcolor_to_cv2, get_contour_point_at
import images_combination
from matplotlib import colors as mcolors

def plot_similitude(image1, image2, im1_points, im2_points, score_list, patch_w=500, patch_h=500, thickness=10):
    """
    3 subplots:
        - Extracted patches from image1.
        - Found patches in image2.
        - Loss function score of the matches.
    """
    
    f, axarr = plt.subplots(1,3, figsize=(17, 5))
    
    image1_colored = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

    for i, pos in enumerate(im1_points):

        top_left_corner = tuple(pos)
        bottom_right_corner = (top_left_corner[0]+patch_w, top_left_corner[1]+patch_h)

        color = pltcolor_to_cv2(get_color_list(i))
        
        image1_colored = cv2.rectangle(image1_colored, top_left_corner, bottom_right_corner, color, thickness)
        
    axarr[0].imshow(image1_colored)
        
    image2_colored = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    for i, pos in enumerate(im2_points):

        color = pltcolor_to_cv2(get_color_list(i))
        pt0=pos[0]
        pt1=pos[1]
        pt2=pos[2]
        pt3=pos[3]
        
        
        image2_colored = cv2.line(image2_colored, pt0, pt1, color, thickness)
        image2_colored = cv2.line(image2_colored, pt1, pt2, color, thickness)
        image2_colored = cv2.line(image2_colored, pt2, pt3, color, thickness)
        image2_colored = cv2.line(image2_colored, pt3, pt0, color, thickness)

    axarr[1].imshow(image2_colored)
    
    # Example data
    y_pos = np.arange(len(score_list))
    color = [get_color_list(i) for i in range(len(score_list))]

    axarr[2].barh(y_pos, score_list, color=color)
    #axarr[2].set_yticks([], [])
    axarr[2].set_yticks(y_pos)
    axarr[2].set_ylabel("Patch indices")
    axarr[2].set_xlabel('Similitude scores (Min is best)')
    
    
    axarr[0].set_title('Patches Extracted')
    axarr[1].set_title('Matches Found')

def plot_patches(image, patches_list, patch_w=500, patch_h=500, thickness=10):
    """
    Plot extracted patches in the anchor.
    """
    
    image=np.invert(image)
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i, pos in enumerate(patches_list):

        top_left_corner = tuple(pos)
        bottom_right_corner = (top_left_corner[0]+patch_w, top_left_corner[1]+patch_h)

        color = pltcolor_to_cv2(get_color_list(i))
        
        image_colored = cv2.rectangle(image_colored, top_left_corner, bottom_right_corner, color, thickness)

    plt_plot_cv2(image_colored)
    plt.show()
    
def rotate_scale(image, angle, scale):
    # ROTATE TEMPLATE:
    image = imutils.rotate_bound(image, angle)
    
    # RESIZE TEMPLATE:
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    return cv2.resize(image, dim)
    
def get_patch_composite(image, patch, tx, ty, angle, scale, flag_composite=False):
    """
    Create a composite with the patch rotated, scaled and translated on the image.
    
    """

    patch_transformed = rotate_scale(patch, angle, scale)
    
    # TRANSLATE IMAGE:
    image_section = image[ty:(ty+patch_transformed.shape[0]), tx:(tx+patch_transformed.shape[1])].copy()

    if image_section.shape != patch_transformed.shape:
        return None, None, None
    
    if (flag_composite):
        patch_composite = cv2.addWeighted(image_section, 1, patch_transformed, 1, 0)
        return image_section, patch_transformed, patch_composite
    else:
        return image_section, patch_transformed, None
    
def get_patch_transformed(target, tx, ty, angle, scale, patch_w, patch_h):
    """
    Same as get_patch_composite but the TARGET is rotating.
    
    """
    # ROTATE TEMPLATE:
    target = imutils.rotate_bound(target, angle)
    
    # RESIZE TEMPLATE:
    width = int(target.shape[1] * scale)
    height = int(target.shape[0] * scale)
    dim = (width, height)

    target = cv2.resize(target, dim)
    
    # TRANSLATE IMAGE:
    target_section = target[ty:(ty+patch_h), tx:(tx+patch_w)].copy()
    
    if target_section.shape != (patch_h, patch_w):
        return None

    return target_section

def get_center_and_tl_points(tl_adj, patch_adj_size, center_patch, tl_patch, resize):
    """
    Function that computes:
    
    - the vector between the center of the main patch and the center of an adjacent patch
    
    - the vector between the tl point of the main patch and the tl point of an adjacent patch
    """
    tl_adj1 = tl_adj.copy()
    center_adj1 = tl_adj1 + [patch_adj_size/2, patch_adj_size/2]

    patch_center_to_adj1_center = center_adj1 - center_patch
    patch_center_to_adj1_center = patch_center_to_adj1_center/resize

    tl_patch_to_tl_adj1 = tl_adj1 - tl_patch
    tl_patch_to_tl_adj1 = tl_patch_to_tl_adj1/resize

    tl_adj1_resized = tl_adj1/resize
    
    return patch_center_to_adj1_center, tl_patch_to_tl_adj1, tl_adj1_resized

def get_adj_params(angle,
                   scale,
                   tx,
                   ty,
                   patch_process,
                   patch_adj_process,
                   patch_center_to_adj_center, 
                   tl_patch_to_tl_adj,
                   tl_adj_resized,
                   center_adj_resized,
                   post_patch_center_shape):
    """
    helper to find coordinates of adjacent patch after rotation, translation and scaling of the central patch.
    """
    
    center_of_rotation = [0, 0]
    angle_rad = np.deg2rad(angle)
    
    ######## Find center of patch adj1
    post_adj_center = np.asarray(images_combination.rotate(center_of_rotation, 
                                                                   patch_center_to_adj_center, angle_rad))
    post_adj_center = post_adj_center*scale

    post_patch_center = [tx + post_patch_center_shape[0]/2, ty + post_patch_center_shape[1]/2]

    post_adj_center = post_patch_center + post_adj_center
    
    ######## Find top left point of adj patch after transformation
    post_tl_patch_to_tl_adj = np.asarray(images_combination.rotate(center_of_rotation,
                                                            tl_patch_to_tl_adj, angle_rad))

    post_tl_patch_to_tl_adj = post_tl_patch_to_tl_adj*scale
    
    tl_x, tl_y = images_combination.get_transformed_points(0, 0, patch_process, angle, scale)
    post_tl_patch = [tl_x, tl_y]
    post_tl_patch += np.asarray([tx, ty])
    
    post_tl_adj = post_tl_patch + post_tl_patch_to_tl_adj
    
    post_tl_adj = np.around(post_tl_adj).astype(int)
    
    ######## Find angle rotation for patch adj1
    _,a,s, _  = images_combination.vector_alignment(tl_adj_resized
                                     , center_adj_resized
                                     , post_tl_adj
                                     , post_adj_center
                                    )
    post_adj_shape = rotate_scale(patch_adj_process, -a, s).shape

    post_adj_bound_tl = [post_adj_center[0] - post_adj_shape[0]/2,
                         post_adj_center[1] - post_adj_shape[1]/2]
    
    return a, s, np.around(post_adj_bound_tl).astype(int), post_adj_center
    
###############################################################################
###############################################################################
###############################################################################
    
def get_list_angle(n_angles=10):
    """Uniform distribution of angle."""
    return np.linspace(0, 2*np.pi-(2*np.pi)/n_angles, n_angles)

def extract_templates_wAngle(anchor,
                             nb_patch,
                             template_size):
    """Corrected extraction of multiple patches"""
    # set the angles parametrizing the lines along which to extract the templates
    angles = get_list_angle(nb_patch)
    
    # define lists to store the templates and templates anchoring points
    templates_angles = []
    templates_corners = []
    
    # for each angle: extract template and store it
    for i, theta in enumerate(angles):
        
        template_theta, corner_theta = extract_single_template_wAngle(anchor
                                                 , theta
                                                 , template_w=template_size
                                                 , template_h=template_size
                                                )

        # store template
        templates_angles += [template_theta]
        #templates_corners += [np.array([x_temp-template_w//2, y_temp-template_h//2])]
        templates_corners += [corner_theta]
    
    return templates_angles, templates_corners

def extract_single_template_wAngle(anchor_cadastre
                      , angle
                      , template_w=500
                      , template_h=500
                     ):
    """
    Corrected extraction of one patch from an image.
    
    get_contour_point_at is a helper from images_utils.py
    Otherwise, there is an offset error in the function extract_templates.

    """
    
    x_temp, y_temp = get_contour_point_at(anchor_cadastre.copy(), angle, False)
    
    # extract template
    corner_theta = np.array([x_temp-template_w//2, y_temp-template_h//2])
        
    template_theta = extract_single_template(anchor_cadastre
                                          , top_left_corner_coordinates=corner_theta
                                          , template_w=template_w
                                          , template_h=template_h)
    return template_theta, corner_theta

def extract_adjacent_templates_wAngle(anchor_cadastre
                      , angle
                      , diff
                      , template_w=500
                      , template_h=500
                     ):
    """
    Corrected extraction of a pair of adjacent patches.

    """   
    
    angles = np.asarray([angle-diff, angle+diff])
    
    # define lists to store the templates and templates anchoring points
    templates_angles = []
    templates_corners = []

    # for each angle: extract template and store it
    for i, theta in enumerate(angles):
        
        template_theta, corner_theta = extract_single_template_wAngle(anchor_cadastre
                                                 , theta
                                                 , template_w=template_w
                                                 , template_h=template_h
                                                )
        

        # store template
        templates_angles += [template_theta]
        #templates_corners += [np.array([x_temp-template_w//2, y_temp-template_h//2])]
        templates_corners += [corner_theta]
    
    return templates_angles, templates_corners
    
def extract_adjacent_templates(anchor_cadastre
                      , angle
                      , diff
                      , template_w=500
                      , template_h=500
                     ):
    """
    """
    
    # extract parameters from the anchor
    h_anchor, w_anchor = anchor_cadastre.shape
    # set center
    x_anchor_center = w_anchor//2
    y_anchor_center = h_anchor//2
    # compute diagonal length
    l_diag = np.hypot(x_anchor_center, y_anchor_center)
    
    
    angles = np.asarray([angle-diff, angle+diff])
    
    # define lists to store the templates and templates anchoring points
    templates_angles = []
    templates_corners = []

    # for each angle: extract template and store it
    for i, theta in enumerate(angles):
        # compute the image coordinate of the end of the line parametrized by angle theta
        x_target = np.cos(theta)*l_diag+x_anchor_center
        y_target = np.sin(theta)*l_diag+y_anchor_center

        # make sure the target point is in the image
        x_target = np.min([np.max([0, x_target]), w_anchor])
        y_target = np.min([np.max([0, y_target]), h_anchor])

        # define the line joining the center of the anchor and the target point
        N = int(np.hypot(x_target-x_anchor_center, y_target-y_anchor_center))
        x_theta = np.linspace(x_anchor_center, x_target-1, N).astype(np.int32)
        y_theta = np.linspace(y_anchor_center, y_target-1, N).astype(np.int32)

        # compute the image values along the line
        zi_theta = anchor_cadastre[y_theta, x_theta]
        # find last edge along the line from center to image's edge
        last_edge_theta = np.where(zi_theta>0)[0][-1]
        # set the center of the template to be extracted on the last value of interest
        x_temp, y_temp = x_theta[last_edge_theta], y_theta[last_edge_theta]

        # make sure the template to be extracted is contained within the anchor cadastre
        x_temp = np.min([np.max([template_w//2, x_temp]), w_anchor-template_w//2])
        y_temp = np.min([np.max([template_h//2, y_temp]), h_anchor-template_h//2])

        # extract template
        corner_theta = np.array([x_temp-template_w//2, y_temp-template_h//2])
        
        template_theta = extract_single_template(anchor_cadastre
                                                 , top_left_corner_coordinates=corner_theta
                                                 , template_w=template_w
                                                 , template_h=template_h
                                                )
        
        #template_theta = anchor_cadastre[y_temp-template_h//2:y_temp+template_h//2
         #                                , x_temp-template_w//2:x_temp+template_w//2
          #                              ]
        # store template
        templates_angles += [template_theta]
        #templates_corners += [np.array([x_temp-template_w//2, y_temp-template_h//2])]
        templates_corners += [corner_theta]
    
    return templates_angles, templates_corners

"""

! FOLLOWING CODE HAS BEEN TAKEN FROM THE ALIGNMENT TOOL PROJECT:

github: https://github.com/noe-d/AlignmentTool
wiki: http://fdh.epfl.ch/index.php/Alignment_of_XIXth_century_cadasters

"""

def extract_single_template(anchor_cadastre
                            , top_left_corner_coordinates
                            , template_w=500
                            , template_h=500
                           ):
    """
    Extract a template from an image given its top left corner coordinates, width and height
    
    Input:
        anchor_cadastre: 2D array (image)
            image of the cadastre on which to extract the template
        top_left_corner_coordinates: tuple or corresponding
            coordinate in pixel values of the top-left corner of the template to be extracted
        template_w: int (default value = 500)
            width of the extracted template in pixels
        template_h: int (default value = 500)
            height of the extracted template in pixels
    
    Output:
        template: 2D array (image)
            extracted template of size template_w x template_h 
        
    Example:
    extract_templates(Berney001) will return two lists of dimensions 10, 
    the first one storing the subpart of Berney001 referred to as templates of dimensions 500x500,
    and the second one containing the coordinate on Berney001 of the top-left corners of these templates
    """
    x_corner = top_left_corner_coordinates[0]
    y_corner = top_left_corner_coordinates[1]
    
    template = anchor_cadastre[y_corner:y_corner+template_h
                               , x_corner:x_corner+template_w
                              ]
    return template


def extract_templates(anchor_cadastre
                      , template_w=500
                      , template_h=500
                      , n_angles=10
                     ):
    """
    Extract a choosen number of templates (subpart of the image) from a given cadastre.
    The templates have fixed (given) height and width and are centered around the last 
        edge along lines defined by angles. 
        The angles form a uniform partition of the circle into N sections (given N).

    Input:
        anchor_cadastre: 2D array (binary image)
            image of the edges of the cadastre on which to extract the templates
        template_w: int (default value = 500)
            width of the extracted templates in pixels
        template_h: int (default value = 500)
            height of the extracted templates in pixels
        n_angles: int (default value = 10)
            number of angles to consider => a fortiori number of extracted templates
    
    Output:
        templates_angles, templates_corners: list, list
            templates_angles: stores the extracted templates -> size n_angles
                each element of the list is a subpart of size (template_h, template_w) of the anchor cadastre
            templates_corners: stores the top-left coordinates (in pixel coordinates) of the templates -> size n_angles
                each element of the list is a 2D array representing a point on the anchor cadatsre
        
    Example:
    extract_templates(Berney001) will return two lists of dimensions 10, 
    the first one storing the subpart of Berney001 referred to as templates of dimensions 500x500,
    and the second one containing the coordinate on Berney001 of the top-left corners of these templates

    """
    
    # extract parameters from the anchor
    h_anchor, w_anchor = anchor_cadastre.shape
    # set center
    x_anchor_center = w_anchor//2
    y_anchor_center = h_anchor//2
    # compute diagonal length
    l_diag = np.hypot(x_anchor_center, y_anchor_center)
    
    # set the angles parametrizing the lines along which to extract the templates
    angles = get_list_angle(n_angles)
    
    # define lists to store the templates and templates anchoring points
    templates_angles = []
    templates_corners = []
    
    # for each angle: extract template and store it
    for i, theta in enumerate(angles):
        # compute the image coordinate of the end of the line parametrized by angle theta
        x_target = np.cos(theta)*l_diag+x_anchor_center
        y_target = np.sin(theta)*l_diag+y_anchor_center

        # make sure the target point is in the image
        x_target = np.min([np.max([0, x_target]), w_anchor])
        y_target = np.min([np.max([0, y_target]), h_anchor])

        # define the line joining the center of the anchor and the target point
        N = int(np.hypot(x_target-x_anchor_center, y_target-y_anchor_center))
        x_theta = np.linspace(x_anchor_center, x_target-1, N).astype(np.int32)
        y_theta = np.linspace(y_anchor_center, y_target-1, N).astype(np.int32)

        # compute the image values along the line
        zi_theta = anchor_cadastre[y_theta, x_theta]
        # find last edge along the line from center to image's edge
        last_edge_theta = np.where(zi_theta>0)[0][-1]
        # set the center of the template to be extracted on the last value of interest
        x_temp, y_temp = x_theta[last_edge_theta], y_theta[last_edge_theta]

        # make sure the template to be extracted is contained within the anchor cadastre
        x_temp = np.min([np.max([template_w//2, x_temp]), w_anchor-template_w//2])
        y_temp = np.min([np.max([template_h//2, y_temp]), h_anchor-template_h//2])

        # extract template
        corner_theta = np.array([x_temp-template_w//2, y_temp-template_h//2])
        
        template_theta = extract_single_template(anchor_cadastre
                                                 , top_left_corner_coordinates=corner_theta
                                                 , template_w=template_w
                                                 , template_h=template_h
                                                )
        
        #template_theta = anchor_cadastre[y_temp-template_h//2:y_temp+template_h//2
         #                                , x_temp-template_w//2:x_temp+template_w//2
          #                              ]
        # store template
        templates_angles += [template_theta]
        #templates_corners += [np.array([x_temp-template_w//2, y_temp-template_h//2])]
        templates_corners += [corner_theta]
    
    return templates_angles, templates_corners
