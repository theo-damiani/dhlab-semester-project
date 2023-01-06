###############################################################################
###############################################################################
###############################################################################

from custom_ga import geneticalgorithm as ga
from geneticalgorithm import geneticalgorithm as unrestricted_ga

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, distance_transform_bf, distance_transform_cdt
from tqdm import tqdm
import matplotlib.pyplot as plt

import images_combination
import patch_utils
import images_utils
import loss

import datetime
import time

import csv
import json
import ast
###############################################################################
###############################################################################
###############################################################################

class Model:
    """
    This class represent the model used to align cadastres.
    
    Parameters:
    -----------
    
    - loss: Callable loss function
    - patch_numbers: number of patch to extract in anchors
    - distance_transform: dict for the representation of the images, you can invert or not patches and targets
                          and computes Eucilidian Distance Transform by settings the respective flag to True.
                          dt_function is the callable function to compute EDT.
    - adjacent_patch_process: dict to specify the adjacent patches configuration. Boolean flag to specify
                          if you want adj patches. 'diff_angle_adj' to tuned the offset angle during the extraction
                          of patches, note that you may need to change the threshold in the while loop
                          of the function 'get_adjacent_patch'.
                          'kernel_adj' is used for the dilation of adjacent patch.
    - preprocessing_parameters: dict for the dilation and resizing of patched and targets.
    - search_parameters: ranges for the rotation and scale. Translation cannot be specified because, either classic GA
                         is used and the whole image space is possible or restricted GA is used and then valid translations
                         are computed in the function 'run'.
    - ga_parameters: main genetic algorithm parameters.
    - loss_error: loss value if a patch is out of bounds of the target.
    - classification_top_pop_variance: should be True to get the disruptive selection.
    - flag_custom_ga: True to run with the restricted GA.
    
    For an example of configuration with restricted GA, IoU loss see the notebook RUN-MODEL.

    """

    def __init__(self, list_cadastres_name,
                       list_cadastre,
                       loss,
                       patch_numbers=10, 
                       patch_size=500,
                       distance_transform={'flag_target': False,
                                           'flag_patch': False,
                                           'flag_invert_patch': False,
                                           'flag_invert_target': True,
                                           'dt_function': distance_transform_edt},
                       adjacent_patch_process={'flag_adj_patch': False,
                                               'diff_angle_adj': 0.2,
                                               'patch_adj_size': 300,
                                               'kernel_adj': np.ones((5, 5), np.uint8)},
                       preprocessing_parameters={'kernel_dilate_target': np.ones((7, 7), np.uint8),
                                                 'kernel_dilate_patch': np.ones((8, 8), np.uint8),
                                                 'resize': 6},
                       search_parameters={'angle_range': [0.0, 360.0],
                                          'scale_range': [0.95, 1.05]},
                       ga_parameters={'max_num_iteration': 1000,\
                                       'population_size':200,\
                                       'mutation_probability':0.9,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.8,\
                                       'parents_portion': 0.1,\
                                       'parents_mut_portion': 0.1,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},
                        loss_error=10000,
                        classification_top_pop_variance=True,
                        flag_custom_ga=True,
                        experience_name=""):
        
        #############################################################            
        self.list_cadastre = list_cadastre.copy()
        self.list_cadastres_name= list_cadastres_name.copy()
        
        #############################################################
        assert (callable(loss)),"\n The loss function must be callable."  
        self.loss = loss

        #############################################################
        assert(type(patch_numbers) is int),"\n patch_numbers must 'int'"
        self.patch_numbers = patch_numbers
        self.patch_angle_list = patch_utils.get_list_angle(n_angles=patch_numbers)
        self.patch_angle = patch_utils.get_list_angle(n_angles=patch_numbers)[0]
        
        #############################################################
        assert(type(patch_size) is int and patch_size <= 500 and patch_size >= 50),"\n patch_numbers must 'int' in range [50, 500]"
        self.patch_size = patch_size
        
        #############################################################
        self.preprocessing_parameters = preprocessing_parameters
        self.kernel_dilate_target = preprocessing_parameters['kernel_dilate_target']
        self.kernel_dilate_patch = preprocessing_parameters['kernel_dilate_patch']
        assert(len (row) == len (self.kernel_dilate_target) for row in self.kernel_dilate_target),"\n kernel_dilate_target must be a squared matrix."
            
        assert(len (row) == len (self.kernel_dilate_patch) for row in self.kernel_dilate_patch),"\n kernel_dilate_target must be a squared matrix."
            
        self.resize_preprocessing = preprocessing_parameters['resize']
        assert(type(self.resize_preprocessing) is int),"\n resize must be 'int'"
        
        #############################################################
        # Check done in the geneticalgorithm's library.
        self.ga_parameters = ga_parameters
        
        self.search_parameters = search_parameters
        
        #############################################################
        self.loss_error = loss_error

        assert (callable(distance_transform['dt_function'])),"\n The distance transform function must be callable."  
        self.distance_transform_function = distance_transform['dt_function']
        
        self.flag_dt_target = distance_transform['flag_target']
        self.flag_dt_patch = distance_transform['flag_patch']
        self.flag_inv_patch = distance_transform['flag_invert_patch']
        self.flag_inv_target = distance_transform['flag_invert_target']
        
        self.distance_transform=distance_transform
        
        #############################################################
        ## Adjacent patches extraction config.
        self.adjacent_patch_process=adjacent_patch_process
        self.flag_adj_patch = adjacent_patch_process['flag_adj_patch']
        self.diff_angle_adj = adjacent_patch_process['diff_angle_adj']
        self.patch_adj_size = adjacent_patch_process['patch_adj_size']
        self.kernel_adj = adjacent_patch_process['kernel_adj']
        
        self.classification_top_pop_variance=classification_top_pop_variance
        
        self.flag_custom_ga=flag_custom_ga
        self.experience_name=experience_name
        
        return

               
    def run(self):
        
        best_score_index = None

        date = datetime.datetime.now()
        self.date_ymd = date.date()
        self.date_hm = f"{date.hour}:{date.minute}"

        self.time_start = time.time()
            
        # Always take the first as the anchor.
        anchor = self.list_cadastre[0].copy()
        target = self.list_cadastre[1].copy()
            
        anchor_name = self.list_cadastres_name[0]
        target_name = self.list_cadastres_name[1]
            
        ###################### Preprocessing ########################
        # Dilate and resize target
        target_image_process = images_utils.dilate_resize_im(target,
                                                          self.kernel_dilate_target,
                                                          self.resize_preprocessing)
        
        target_image_process = np.invert(target_image_process)
        _ ,target_image_process = cv2.threshold(target_image_process,200,255,cv2.THRESH_BINARY)
                
        ###################### Extract Patches ########################                                    
        patch_list, patch_points_list = self.get_patch_list(anchor)
            
        ###################### Heatmap ########################  
        heatmap = images_utils.build_heatmap(target_image_process)
            
        self.heatmap=heatmap.copy()
        ret,heatmap_treshold = cv2.threshold(self.heatmap,30,255,cv2.THRESH_BINARY)
        self.heatmap_treshold=heatmap_treshold.copy()
            
        tmp1 = np.argwhere(heatmap<(10))
        tmp2 = np.argwhere(heatmap<(20))
        tmp3 = np.argwhere(heatmap<(30))
        valid_translations_center = np.concatenate((tmp1, tmp2, tmp3))
            
        patch_p_size_h = self.patch_size//self.resize_preprocessing
        patch_p_size_w = self.patch_size//self.resize_preprocessing
        offset_tl_h, offset_tl_w = patch_p_size_h//2, patch_p_size_w//2

        valid_translations_tl=(valid_translations_center-np.asarray([offset_tl_h, offset_tl_w])).copy()

        mask=np.all(valid_translations_tl>=0, axis=1)
        self.valid_translations_tl=valid_translations_tl[mask].copy()
        
        #############################################################

        if self.flag_dt_target:
            #target_image_process = cv2.normalize(target_image_process, None, alpha=0, beta=1, 
            #                                                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Compute distance transform:
            target_image_process = self.distance_transform_function(target_image_process)
            # Normalize again:
            target_image_process = cv2.normalize(target_image_process, None, alpha=0, beta=1,
                                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #############################################################
            
        score_list = []
        patch_full_res = []
            
        # Store population at last iteration for every patch:
        disruptive_score = []
        disruptive_score_avg = []
                                        
        for i in tqdm(range(len(patch_list))):
                
            self.patch_angle = self.patch_angle_list[i].copy()
            
            print(f"\n\nStart GA on patch={i}")
                       
            patch_res, pop_report = self.get_patch_score(anchor,
                                                 target_image_process, 
                                                 patch_list[i].copy(),
                                                 patch_points_list[i].copy(),
                                                )
                
            score_list.append(patch_res['score'].copy())
            patch_full_res.append(patch_res.copy())

            translation_range=20
            if self.classification_top_pop_variance:
                top1_res = pop_report[0]
                topX_res = []
                f_ratio=-1
                    
                for i, val in enumerate(pop_report):
                    cty=val[0]
                    ctx=val[1]

                    cty_tresh = np.abs(top1_res[0]-cty)
                    ctx_tresh = np.abs(top1_res[1]-ctx)
                    if ((ctx_tresh>translation_range) or (cty_tresh>translation_range)):
                        # ratio of score:
                        f_ratio = top1_res[4]/val[4]
                        print(i)
                        break
                disruptive_score.append(f_ratio)
                
                top1_res = pop_report[0][4]
                
                parents_p=self.ga_parameters['parents_portion']*self.ga_parameters['population_size']
                parents_mut_p=self.ga_parameters['parents_mut_portion']*self.ga_parameters['population_size']
                tt_p=int(parents_p+parents_mut_p)-1
                f_ratio_avg = np.mean(pop_report[0:tt_p,4])
                disruptive_score_avg.append(top1_res/f_ratio_avg)
                
                
        #############################################################
        best_score_index = np.argmin(score_list)
                
        #############################################################
        # Save run:
        self.save_result(anchor_name, target_name, patch_points_list, patch_full_res,
                         disruptive_score, disruptive_score_avg)
        self.save_model_parameters(anchor_name, target_name)
                
        return best_score_index# full_matching_result, patches_info
    
    def get_target(self, index_match):
        indices = self.get_cadastres_indices()
        return self.list_cadastre[indices[index_match][1].copy()]
    
    def get_patch_score(self, anchor, target_image_process, patch, top_left_patch):
        
        min_angle=self.search_parameters['angle_range'][0]
        max_angle=self.search_parameters['angle_range'][1]
        min_scale=self.search_parameters['scale_range'][0]
        max_scale=self.search_parameters['scale_range'][1]
        
        patch_process = images_utils.dilate_resize_im(patch,
                                               self.kernel_dilate_patch,
                                               self.resize_preprocessing)
            
        #############################################################
        ############### Adjacent patch configuration ################
        patch_adj1_process=[]
        patch_adj2_process=[]
        patch_points_list_adj=[]
        if self.flag_adj_patch:
            patch_list_adj, patch_points_list_adj = self.get_adjacent_patch(anchor, self.patch_angle, top_left_patch.copy())

            patch_adj1 = patch_list_adj[0].copy()
            patch_adj2 = patch_list_adj[1].copy()

            patch_adj1_process = images_utils.dilate_resize_im(patch_adj1, self.kernel_adj, self.resize_preprocessing)
            _ ,patch_adj1_process = cv2.threshold(patch_adj1_process,200,255,cv2.THRESH_BINARY)

            patch_adj2_process = images_utils.dilate_resize_im(patch_adj2, self.kernel_adj, self.resize_preprocessing)
            _ ,patch_adj2_process = cv2.threshold(patch_adj2_process,200,255,cv2.THRESH_BINARY)
            #############################################################
        
        model = self.define_geneticalgo(target_image_process.copy(),
                                        patch_process.copy(),
                                        top_left_patch.copy(),
                                        patch_adj1_process.copy(),
                                        patch_adj2_process.copy(),
                                        patch_points_list_adj.copy(),
                                        min_angle=min_angle, max_angle=max_angle,
                                        min_scale=min_scale, max_scale=max_scale)
                
        _ = model.run()
                
        res_ty, res_tx, res_angle, res_scale = model.output_dict["variable"].copy()
        score = model.output_dict["function"].copy()
        
        res = {'score': score,
               'tx': int(res_tx),
               'ty': int(res_ty),
               'angle': res_angle,
               'scale': res_scale}
        
        if self.flag_custom_ga:
            if self.classification_top_pop_variance:
                return res, model.pop_report
            else:
                return res, None
        else:
            return res, None
        
        return        

    def get_patch_list(self, cadastre):
        
        patch_list, patch_points_list = patch_utils.extract_templates_wAngle(cadastre,
                                                                             self.patch_numbers,
                                                                             self.patch_size)
        
        return patch_list, patch_points_list
    
    def get_adjacent_patch(self, cadastre, patch_angle, tl_patch_ref):
        
        patch_list_adj, patch_points_list_adj = patch_utils.extract_adjacent_templates(cadastre, 
                                                                              patch_angle,
                                                                              self.diff_angle_adj,              
                                                                              template_w=self.patch_adj_size,
                                                                              template_h=self.patch_adj_size)
        #############################################################
        # Take new adjacent patch if previous one are too far from the main patch center:

        for i, tl_adj_p in enumerate(patch_points_list_adj):
            distance_adj=self.get_distance_center_patch(tl_patch_ref, tl_adj_p, self.patch_size, self.patch_adj_size) 
            diff=self.diff_angle_adj
            while(distance_adj>300):
                diff-=0.02
                new_angle = patch_angle
                if i==0:
                    new_angle -= diff
                else:
                    new_angle += diff
                template_theta, corner_theta = patch_utils.extract_single_template_wAngle(cadastre
                              , new_angle
                              , template_w=self.patch_adj_size
                              , template_h=self.patch_adj_size
                             )
                distance_adj=self.get_distance_center_patch(tl_patch_ref, corner_theta, self.patch_size, self.patch_adj_size) 
                
                patch_points_list_adj[i]=corner_theta
                patch_list_adj[i]=template_theta
        
        return patch_list_adj, patch_points_list_adj
    
    def get_distance_center_patch(self, vec1_p, vec2_a, p_size, a_size):
        """compute distance between the center of a principal patch 'vec1_p' and an adjacent patch 'vec2_a'"""
        p_center = [p_size//2, p_size//2]
        a_center = [a_size//2, a_size//2]
        vec_distance = (vec2_a+a_center)-(vec1_p+p_center) 
        distance = np.linalg.norm(vec_distance)
        return distance
    
    def model_loss(self, X, Y):
        l = self.loss(X,Y)
        
        return l
    
    def compute_adj_patch_loss(self,
                               dest_edt, angle, scale, tx, ty,
                               patch_process, patch_adj1_process, patch_adj2_process,
                               patch_transformed,
                               center_patch,
                               tl_patch,
                               patch_points_list_adj
                               
                              ):
        ############## ADJ 1
        tl_adj = patch_points_list_adj[0].copy()
        patch_adj_process = patch_adj1_process.copy()

        return_vars = patch_utils.get_center_and_tl_points(tl_adj, self.patch_adj_size,
                                                           center_patch, tl_patch, self.resize_preprocessing)
        patch_center_to_adj_center = return_vars[0].copy()
        tl_patch_to_tl_adj = return_vars[1].copy()
        tl_adj_resized = return_vars[2].copy()

        adj_process_h, adj_process_w = patch_adj_process.shape
        center_adj_resized = tl_adj_resized + [adj_process_w/2, adj_process_h/2]

        a, s, post_adj_bound_tl, post_adj_center = patch_utils.get_adj_params(angle, scale, tx, ty,
                                                     patch_process, patch_adj_process,
                                                     patch_center_to_adj_center, tl_patch_to_tl_adj,
                                                     tl_adj_resized, center_adj_resized,
                                                     patch_transformed.shape)

        target_section_adj1, post_adj1_patch, _ = patch_utils.get_patch_composite(dest_edt, patch_adj_process,
                                                                    post_adj_bound_tl[0],
                                                                    post_adj_bound_tl[1],
                                                                    -a, s)
        if target_section_adj1 is None:
            return self.loss_error  

        if self.flag_inv_target:
            target_section_adj1 = np.invert(target_section_adj1)
            
        _ ,post_adj1_patch = cv2.threshold(post_adj1_patch,80,255,cv2.THRESH_BINARY)
        if self.flag_inv_patch:
            post_adj1_patch = np.invert(post_adj1_patch)

        l_adj1 = self.model_loss(target_section_adj1, post_adj1_patch)

        ############## ADJ 2
        tl_adj = patch_points_list_adj[1].copy()
        patch_adj_process = patch_adj2_process.copy()

        return_vars = patch_utils.get_center_and_tl_points(tl_adj, self.patch_adj_size,
                                                           center_patch, tl_patch, self.resize_preprocessing)
        
        patch_center_to_adj_center = return_vars[0].copy()
        tl_patch_to_tl_adj = return_vars[1].copy()
        tl_adj_resized = return_vars[2].copy()

        adj_process_h, adj_process_w = patch_adj_process.shape
        center_adj_resized = tl_adj_resized + [adj_process_w/2, adj_process_h/2]

        a, s, post_adj_bound_tl, post_adj_center = patch_utils.get_adj_params(angle, scale, tx, ty,
                                                         patch_process, patch_adj_process,
                                                         patch_center_to_adj_center, tl_patch_to_tl_adj,
                                                         tl_adj_resized, center_adj_resized,
                                                         patch_transformed.shape)

        target_section_adj2, post_adj2_patch, _ = patch_utils.get_patch_composite(dest_edt, patch_adj_process,
                                                                    post_adj_bound_tl[0],
                                                                    post_adj_bound_tl[1],
                                                                    -a, s)
        if target_section_adj2 is None:
            return self.loss_error 
        
        if self.flag_inv_target:
            target_section_adj2 = np.invert(target_section_adj2)
            
        _ ,post_adj2_patch = cv2.threshold(post_adj2_patch,80,255,cv2.THRESH_BINARY)
        if self.flag_inv_patch:
            post_adj2_patch = np.invert(post_adj2_patch)

        l_adj2 = self.model_loss(target_section_adj2, post_adj2_patch)
        
        return l_adj1+l_adj2
        
        
###############################################################################
###############################################################################
###############################################################################
    
    def define_geneticalgo(self, target_image_process,
                           patch_process, top_left_patch,
                           patch_adj1_process, patch_adj2_process,
                           patch_points_list_adj,
                           min_angle=0, max_angle=360,
                           min_scale=0.95, max_scale=1.05):
        
        tsize_y, tsize_x = target_image_process.shape
        psize_y, psize_x = patch_process.shape
        
        min_tx=0
        max_tx=(tsize_x)-(psize_x)
        min_ty=0
        max_ty=(tsize_y)-(psize_y)
        
        #############################################################
        # Configuration of the geneticalgorithm  

        varbound=np.array([
            [min_ty, max_ty], #value in the range are useless if use of custom_ga
            [min_tx, max_tx], #value in the range are useless if use of custom_ga
            [min_angle, max_angle], #rotation
            [min_scale, max_scale] #rescale percent: keep aspect ratio
        ])
        
        vartype=np.array([['int'],
                          ['int'],
                          ['real'],
                          ['real']])

        if self.flag_custom_ga:
            vartype=np.array([['custom_range'], #ty
                              ['custom_range'], #tx
                              ['real'], # rotation
                              ['real']]) # scale
        
        # Coordinates of the center of the patch,
        # Needed for the adjacent patch configuration:
        center_patch = top_left_patch + [self.patch_size/2, self.patch_size/2] 
        
        #############################################################
        # Define genetic algo loss.
        # Need to be define here so it can be injected into the geneticalgorithm params.
        #   <! --- It takes only one arg! otherwise geneticalgorithm library will not work. --- !>
        def matching_loss(gene):
            tx = int(gene[1].copy())
            ty = int(gene[0].copy())
            angle = gene[2].copy()
            scale = gene[3].copy()
            
            target_section, patch_transformed, _ = patch_utils.get_patch_composite(target_image_process, patch_process,
                                                                            tx, ty, angle, scale)
            
            # im_section can be None, because of the translation performed the patch can be 'outside'
            # the boundary of the target.
            if target_section is None:
                return self.loss_error      
            
            if self.flag_inv_target:
                target_section = np.invert(target_section)
            
            _ ,patch_transformed = cv2.threshold(patch_transformed,80,255,cv2.THRESH_BINARY)
            if self.flag_inv_patch:
                patch_transformed = np.invert(patch_transformed)

            if self.flag_dt_patch :
                patch_transformed = self.distance_transform_function(patch_transformed)
                patch_transformed = cv2.normalize(patch_transformed, None, alpha=0, beta=1,
                                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)            
                
            loss_patch = self.model_loss(target_section, patch_transformed)
            
            if self.flag_adj_patch:
                sum_loss_adj_patch = self.compute_adj_patch_loss(target_image_process, angle, scale, tx, ty,
                                   patch_process, patch_adj1_process, patch_adj2_process,
                                   patch_transformed,
                                   center_patch,
                                   top_left_patch,
                                   patch_points_list_adj)   

                return loss_patch+sum_loss_adj_patch
            else:
                return loss_patch

        #############################################################
        # Creation of the genetic algorithm
        if self.flag_custom_ga:
            model=ga(function=matching_loss,
                     dimension=4,
                     valid_translations=self.valid_translations_tl.copy(),
                     variable_type_mixed=vartype,
                     variable_boundaries=varbound,
                     algorithm_parameters=self.ga_parameters,
                     get_pop_report=self.classification_top_pop_variance,
                     convergence_curve=False)
        else:
            model=unrestricted_ga(function=matching_loss,
                     dimension=4,
                     variable_type_mixed=vartype,
                     variable_boundaries=varbound,
                     algorithm_parameters=self.ga_parameters)

        return model

    def save_result(self, d1_name, d2_name, patch_points_list, patch_res_list, disruptive_score, disruptive_score_avg):        
        # SAVE THE DATA:
        csv_columns = ['anchor','target','tl_pt_patch','transformations','score','true_positive', 'disruptive_score', 'mean_disruptive_score']

        csv_file = f"../results/csv/{self.experience_name}_{d1_name}_{d2_name}_date_{self.date_ymd}_{self.date_hm}.csv"


        dict_data = []

        for j in range(len(patch_res_list)):

            tl_point_patch = patch_points_list[j]

            score=patch_res_list[j]['score']
            tx=patch_res_list[j]['tx']
            ty=patch_res_list[j]['ty']
            angle=patch_res_list[j]['angle']
            scale=patch_res_list[j]['scale']

            transformation_dict = {
                'tx': tx,
                'ty': ty,
                'angle': angle,
                'scale': scale
            }

            true_positive=False
            
            disruptive_s=-42
            disruptive_savg=-42
            if self.classification_top_pop_variance:
                disruptive_s=disruptive_score[j]
                disruptive_savg=disruptive_score_avg[j]

            row_data = {
                'anchor': d1_name,
                'target': d2_name,
                'tl_pt_patch': list(tl_point_patch),
                'transformations': transformation_dict,
                'score': score,
                'true_positive': true_positive,
                'disruptive_score': disruptive_s,
                'mean_disruptive_score': disruptive_savg,
            }

            dict_data.append(row_data)

        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
            
        return
    
    def save_model_parameters(self, d1_name, d2_name):
        time_end = time.time()
        
        hours, rem = divmod(time_end-self.time_start, 3600)
        minutes, seconds = divmod(rem, 60)
        #print()
        print("Total Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        #print()
        elapsed_time="{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        
        # SAVE THE MODEL PARAMETERS:
        csv_file_model = f"../results/csv/{self.experience_name}_{d1_name}_{d2_name}_date_{self.date_ymd}_{self.date_hm}_model.csv"

        csv_columns = ['distance_transform','adjacent_patch_process','preprocessing_parameters',
                       'search_parameters','ga_parameters','loss', 'c_tp_pop_var']

        csv_adjacent_patch_process=self.adjacent_patch_process.copy()
        csv_adjacent_patch_process['kernel_adj']=csv_adjacent_patch_process['kernel_adj'].tolist()
        
        csv_preprocessing_parameters=self.preprocessing_parameters.copy()
        csv_preprocessing_parameters['kernel_dilate_target']=csv_preprocessing_parameters['kernel_dilate_target'].tolist()
        csv_preprocessing_parameters['kernel_dilate_patch']=csv_preprocessing_parameters['kernel_dilate_patch'].tolist()

        csv_distance_transform=self.distance_transform.copy()
        csv_distance_transform['dt_function']=str(self.distance_transform_function)
        loss_function_name = str(self.loss)

        dict_data = {
            'elapsed_time':elapsed_time,
            'patch_size':self.patch_size,
            'patch_numbers':self.patch_numbers,
            'flag_custom_ga': self.flag_custom_ga,
            'distance_transform':csv_distance_transform,
            'adjacent_patch_process':csv_adjacent_patch_process,
            'preprocessing_parameters':csv_preprocessing_parameters,
            'search_parameters':self.search_parameters,
            'ga_parameters':self.ga_parameters,
            'loss':loss_function_name,
            'c_tp_pop_var':self.classification_top_pop_variance,
        }

        try:
            with open(csv_file_model, 'w') as csvfile:
                json.dump(dict_data, csvfile, sort_keys=True, indent=4)
        except IOError:
            print("I/O error")
            
        return