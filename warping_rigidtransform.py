# IMPORT RELEVANT LIBRARIES
import numpy as np
from scipy.interpolate import griddata
import shutil as sh
import matplotlib.pyplot as plt
import os
from random import uniform

"""
Note: Results of the experiment saved in folder named 'task3_results' in the same directory as the python script

""" 

# CREATE CLASS FOR 3D RIGID TRANSFORMATIONS
class RigidTransform:
    
    # constructor function for rotation matrix and translation vector
    def __init__(self, r0_param, t0_param, imsize, flag_composing_ddf=False):
        """
        Constructor function of class RigidTransform for implementing rigid transformation and warping of 3D images using 3 translation
        and 3 rotation parameters. It computes rotation matrix and translation vector for rigid transformation. The compute_ddf() is  
        precomputed by calling the function inside the constructor as well.

        SYNTAX:
            RigidTransform(r0_param,t0_param,imsize)
            RigidTransform(r0_param,t0_param,imsize,flag_composing_ddf=True)

        INPUTS:
            r0_param - List of Rotational parameters from angles -180 to +180 - [r0,r1,r2] - (datatype-list)
            t0_param - Translation Vector in the range from -10 to 10 - [t0,t1,t2] - (datatype-list)
            imsize - size of the image dataset (x,y,z) - (datatype-tuple)
            flag_composing_ddf - Either True or False for composing ddf

        OUTPUTS:
            Returns the following when an object of class RigidTransform is created.
            Stored in self
            
            rotat_mat - Rotation Matrix of order ZYX - (datatype - numpy.ndarray)
            trans_vec - Translation Vector - (datatype - numpy.ndarray)
            ddf_pseudo - Self pre-computed Dense Displacement Field - (datatype - numpy.ndarray)

        """
        
        # flag for different composing of ddfs
        self.flag_composing_ddf = flag_composing_ddf
        
        # define image size
        self.imsize = imsize
        
        # specify translation parameters
        self.t1 = t0_param[0]
        self.t2 = t0_param[1]
        self.t3 = t0_param[2]
        
        # specify rotation parameters converted from degree to radian
        self.r1 = np.deg2rad(r0_param[0])
        self.r2 = np.deg2rad(r0_param[1])
        self.r3 = np.deg2rad(r0_param[2])
        
        # define rotation matrices about each axis (x, y, z)
        self.rotat_x = np.array([[1,                 0,                  0],
                                 [0,                 np.cos(self.r1),    -np.sin(self.r1)],
                                 [0,                 np.sin(self.r1),    np.cos(self.r1)]])
        
        self.rotat_y = np.array([[np.cos(self.r2),   0,                  np.sin(self.r2)],
                                 [0,                 1,                  0],
                                 [-np.sin(self.r2),  0,                  np.cos(self.r2)]])
        
        self.rotat_z = np.array([[np.cos(self.r3),   -np.sin(self.r3),   0],
                                 [np.sin(self.r3),   np.cos(self.r3),    0],
                                 [0,                 0,                  1]])
        
        # define final rotation matrix of order ZYX
        self.rotat_mat = self.rotat_z  @  self.rotat_y  @  self.rotat_x
        
        # define final translation vector
        self.trans_vec = np.array([self.t1, self.t2, self.t3])
        
        # precompute initial ddf for an isotropic pseudo-image
        self.ddf_pseudo = self.compute_ddf(self.imsize)
  
            
        
    # function for calculating the dense displacement field
    def compute_ddf(self, warped_imsize, opt_ddf=0):
        """
        Member Function to compute the value of Dense Displacement Field(DDF) . In this function, a coordinate system is defined
        and forward warping is assumed for warped and original coordinate populations. DDF is computed by taking difference between
        original and warped image.

        SYNTAX:
            compute_ddf(warped_imsize)
            compute_ddf(warped_imsize,[ddf1])
            compute_ddf(warped_imsize,[ddf2])

        INPUTS:
            warped_imsize - Shape of warped image (x,y,z) - (datatype - tuple)
            opt_ddf - DDF passed when flag is true. Set to zero by default unless argument called - (datatype - numpy.ndarray)

        OUTPUTS:
            Output stored in self 
            
            ddf - Dense Displacement Filed = Original Coordinates - Warped Coordinates (datatype - numpy.ndarray)

        Coordinate System - It is defined consistently in Cartesian system with the origin at the centre of the image volume.
        The rotation angle range varies from -180degree to +180degree and translation vector parameters vary from -10 to +10.

        """
        
        # define size dimensions of image
        o_imsize_x = warped_imsize[0]
        o_imsize_y = warped_imsize[1]
        o_imsize_z = warped_imsize[2]
        
        # get coordinates for image
        # coordinate system is always taken with the origin at the centre of the volume
        o_coor_x = np.arange(-(o_imsize_x-1)/2, o_imsize_x/2, dtype=float)
        o_coor_y = np.arange(-(o_imsize_y-1)/2, o_imsize_y/2, dtype=float)
        o_coor_z = np.arange(-(o_imsize_z-1)/2, o_imsize_z/2, dtype=float)

        # specify arrays to store warped and original coordinates
        warped_coor = np.zeros((o_imsize_x*o_imsize_y*o_imsize_z, 3))
        origin_coor = np.zeros((o_imsize_x*o_imsize_y*o_imsize_z, 3))
        
        # populate warped and original coordinate arrays using forward warping
        c1 = 0
        for i in range(o_imsize_x):
            for j in range(o_imsize_y):
                for k in range(o_imsize_z):
                    
                    # defining the original coordinates based on flag set
                    if opt_ddf == 0:
                        origin_coor_vec = np.array([o_coor_x[i], o_coor_y[j], o_coor_z[k]])
                    else:
                        origin_coor_vec = np.array([o_coor_x[i] - opt_ddf[0][c1, 0],\
                                                    o_coor_y[j] - opt_ddf[0][c1, 1], \
                                                    o_coor_z[k] - opt_ddf[0][c1, 2]])

                    # warped Image Coordinate calculation
                    warped_coor_vec = (self.rotat_mat @ origin_coor_vec) + self.trans_vec
                    
                    # original and warped coordinate arrays
                    origin_coor[c1, :] = origin_coor_vec[0], origin_coor_vec[1], origin_coor_vec[2]
                    warped_coor[c1, :] = warped_coor_vec[0], warped_coor_vec[1], warped_coor_vec[2]
                    
                    c1 += 1
   
        # get ddf going from warped to resampled original image
        self.ddf = origin_coor - warped_coor
        
        return self.ddf
    
    
    # function for warping image
    def warp(self, image_vol):
        """
        Member Function for warping an image. Returns a warped image in numoy array. In this function, intensities of the array 
        must be resampled to the rigidly transformed new coordinate system. Here, interpolation is used for resampling the intensities.

        SYNTAX:
            warp(image_vol)

        INPUTS:
            image_vol - Volume image of dimension (x,y,z) - (datatype - numpy.ndarray)
                 
        OUTPUTS:
            image_vol_warp - Warped volume of image with warped image size - (datatype - numpy.ndarray)

        Coordinate System - It is defined consistently in Cartesian system with the origin at the centre of the image volume.
        The rotation angle range varies from -180degree to +180degree and translation vector parameters vary from -10 to +10.

        """ 
        # define original image size dimensions
        imsize_x = image_vol.shape[0]
        imsize_y = image_vol.shape[1]
        imsize_z = image_vol.shape[2]
        
        # get coordinates for original image
        # coordinate system is always taken with the origin at the centre of the volume
        imvol_coor_x = np.arange(-(imsize_x-1)/2, imsize_x/2, dtype=float)
        imvol_coor_y = np.arange(-(imsize_y-1)/2, imsize_y/2, dtype=float)
        imvol_coor_z = np.arange(-(imsize_z-1)/2, imsize_z/2, dtype=float)
        
        # specify arrays to store warped and resampled original coordinates
        warped_coor = np.zeros((imsize_x*imsize_y*imsize_z, 3))
        origin_coor = np.zeros((imsize_x*imsize_y*imsize_z, 3))
        
        # get intensities at warped coordinates
        c2 = 0
        for i in range(imsize_x):
            for j in range(imsize_y):
                for k in range(imsize_z):
                    
                    origin_coor_vec = np.array([imvol_coor_x[i], imvol_coor_y[j], imvol_coor_z[k]])
                    warped_coor_vec = origin_coor_vec - self.ddf[c2, :]
                    
                    warped_coor[c2, :] = warped_coor_vec[0], warped_coor_vec[1], warped_coor_vec[2]
                    origin_coor[c2, :] = origin_coor_vec[0], origin_coor_vec[1], origin_coor_vec[2]
                    
                    c2 += 1
        
        # resampling using interpolation 
        image_vol_warp = griddata((warped_coor[:, 0], warped_coor[:, 1], warped_coor[:, 2]), \
                                   image_vol.flatten(), \
                                   (origin_coor[:, 0], origin_coor[:, 1], origin_coor[:, 2]), \
                                   method="nearest")
            
        image_vol_warp = np.reshape(image_vol_warp, (imsize_x, imsize_y, imsize_z))

        # return warped image volume           
        return image_vol_warp
    
    
    # function for composing two rigid transformations and updating relevant parameters
    def compose(self, r1_param=0, t1_param=0, inp_ddf1=0, inp_ddf2=0):
        """
        Member Function for composing two rigidly transformed image matrices.

        SYNTAX:
            compose(r1_param,t1_param)
            compose(inp_ddf1=ddf1, inp_ddf2=ddf2)

        INPUTS:
            r1_param - Second list of Rotational parameters from angles -180 to +180 - [r0,r1,r2] - (datatype-list)
            t1_param - Second set of translation Vector in the range from -10 to 10 - [t0,t1,t2] - (datatype-list)
            inp_ddf1 - DDF input when composing flag is set to True
            inp_ddf2 - DDF input when composing flag is set to True 

        OUTPUTS:
            Returns an object of class RigidTransform:
            RigidTransform(rotat_tup, trans_tup, self.imsize)
            where rotat_tup - Three-item tuple of rotational angles - (datatype - tuple)
                  trans_tup - Three-item tuple of translation elements - (datatype - tuple)
                  imsize - Updated Image size - (datatype - tuple)
            Stored to self:
            composing_ddfs(inp_ddf1, inp_ddf2) - Function called when flag is set to True. Return a numpy array.

        Coordinate System - It is defined consistently in Cartesian system with the origin at the centre of the image volume.
        The rotation angle range varies from -180degree to +180degree and translation vector parameters vary from -10 to +10.

        """ 
        #Flag condition
        if self.flag_composing_ddf == True:
            
            #Update the sum of ddfs in composing_ddfs()
            self.composing_ddfs(inp_ddf1, inp_ddf2)
            
        else:
            
            # specify translation parameters
            t1 = t1_param[0]
            t2 = t1_param[1]
            t3 = t1_param[2]
            
            # specify rotation parameters
            r1 = np.deg2rad(r1_param[0])
            r2 = np.deg2rad(r1_param[1])
            r3 = np.deg2rad(r1_param[2])
            
            # define rotation matrices about each axis (x, y, z)
            rotat_x2 = np.array([[1,            0,             0],
                                 [0,            np.cos(r1),    -np.sin(r1)],
                                 [0,            np.sin(r1),    np.cos(r1)]])
            
            rotat_y2 = np.array([[np.cos(r2),   0,             np.sin(r2)],
                                 [0,            1,             0],
                                 [-np.sin(r2),  0,             np.cos(r2)]])
            
            rotat_z2 = np.array([[np.cos(r3),   -np.sin(r3),   0],
                                 [np.sin(r3),   np.cos(r3),    0],
                                 [0,            0,             1]])
            
            # define final rotation matrix
            rotat_mat2 = rotat_z2  @  rotat_y2  @  rotat_x2
            
            # define final translation vector
            trans_vec2 = np.array([t1, t2, t3])
            
            # update composed rotation matrix and translation vector
            comp_rotat_mat = rotat_mat2 @ self.rotat_mat
            comp_trans_vec = (rotat_mat2 @ self.trans_vec) + trans_vec2
            
            # get rotation angles
            alpha = np.arctan2(comp_rotat_mat[2,1], comp_rotat_mat[2,2])
            beta = np.arctan2(-comp_rotat_mat[2,0], np.sqrt(comp_rotat_mat[2,1]**2 + comp_rotat_mat[2,2]**2))
            gamma = np.arctan2(comp_rotat_mat[1,0], comp_rotat_mat[0,0])
            
            rotat_tup = (np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma))
            
            # get translation elements
            trans_tup = (comp_trans_vec[0], comp_trans_vec[1], comp_trans_vec[2])

            # return composed RigidTransform object
            return RigidTransform(rotat_tup, trans_tup, self.imsize)
        
        
    # function for composing two ddfs
    def composing_ddfs(self, inp_ddf1, inp_ddf2):
        """
        Member Function for composing two input ddfs

        SYNTAX:
            composing_ddfs(inp_ddf1, inp_ddf2)

        INPUTS:
            inp_ddf1 - DDF input when composing flag is set to True
            inp_ddf2 - DDF input when composing flag is set to True 

        OUTPUTS:
            Returns sum of the two input ddfs stored in self

        """ 
        #Sum of DDF
        self.ddf = inp_ddf1 + inp_ddf2

# FUNCTIONS
def create_dir(folder):
    """
    Function for creating a folder in directory with the same location where python script runs

    SYNTAX:
        create_dir('task3_results')

    INPUTS:
        folder - Name of the folder - (datatype: string)

    OUTPUTS:
        Creates a folder with the specified name for task

    """
    if os.path.exists(folder):
        sh.rmtree(folder)
        os.mkdir(folder)  
    else:
        os.mkdir(folder)  
    
    
def save_planes(folder, plane_num, image, title_pre, title_post=""):
    """
    Function for selecting and saving the image planes along transverse axis for display

    SYNTAX:
        save_planes(folder,plane_num,image,title_pre,title_post)
        eg: save_planes("task2_results/original_image", 5, og_image, "original_image")

    INPUTS:
        folder - Name of the folder - (datatype: string)
        plane_num - Number of the slice - (datatype: int)
        image - image dataset for slice - (datatype: numpy.ndarray)
        title_pre - String for describing the experiment number (datatype: string)
        title_post - String for describing if the image was filtered or not (datatype: string)

    OUTPUTS:
        Save image slice along transverse axis in PNG format 

    """
    total_planes = image.shape[2]
    saved_planes = np.linspace(0, total_planes-1, plane_num)
    
    title_post = "_" + title_post if title_post else title_post
    
    for plane in saved_planes:
        plane_rounded = round(plane)
        title = title_pre + "_z" + str(plane_rounded) + title_post + ".png"
        plt.imsave(folder + "/" + title, image[:, :, plane_rounded], cmap="gray")
    


# -------------------------------MAIN SCRIPT----------------------------------- 
# load image from numpy file and transpose so n_plane is in the z-dim
image = np.load("image_train00.npy").T

# get image size
imsize = (image.shape[0], image.shape[1], image.shape[2])

create_dir("task3_results")

# save 5 planes of original image
create_dir("task3_results/original_image")
save_planes("task3_results/original_image", 5, image, "original_image")


# -------------------------------Experiment 1----------------------------------
num_planes = 5                                                                   


# defined on the basis of cartesian coordinate system of class RigidTransform
# rotation angles are set from a rangle of -180 to +180 - allows rotation upto 180degrees in either directions
# translation parameter set from -10 to +10 - allows tranlsation upto 10 in either directions based on random selection
# specify translation and rotation ranges (+ve rotation angle gives anti-clockwise rotation)
trans_range = [-10, 10]
rotat_range = [-180, 180]

tr_1 = trans_range[0]
tr_2 = trans_range[1]

rr_1 = rotat_range[0]
rr_2 = rotat_range[1]

# sampling of 3 sets of rigid transformations (rounded to 3 d.p.)
T1_trans = (uniform(tr_1, tr_2), uniform(tr_1, tr_2), uniform(tr_1, tr_2))
T1_rotat = (uniform(rr_1, rr_2), uniform(rr_1, rr_2), uniform(rr_1, rr_2))

T2_trans = (uniform(tr_1, tr_2), uniform(tr_1, tr_2), uniform(tr_1, tr_2))
T2_rotat = (uniform(rr_1, rr_2), uniform(rr_1, rr_2), uniform(rr_1, rr_2))

T3_trans = (uniform(tr_1, tr_2), uniform(tr_1, tr_2), uniform(tr_1, tr_2))
T3_rotat = (uniform(rr_1, rr_2), uniform(rr_1, rr_2), uniform(rr_1, rr_2))

T1_trans = [round(T1, 3) for T1 in T1_trans]
T1_rotat = [round(T1, 3) for T1 in T1_rotat]
T2_trans = [round(T2, 3) for T2 in T2_trans]
T2_rotat = [round(T2, 3) for T2 in T2_rotat]
T3_trans = [round(T3, 3) for T3 in T3_trans]
T3_rotat = [round(T3, 3) for T3 in T3_rotat]

print("EXPERIMENT 1 - RIGID TRANSFORMATIONS")
print("\nT1 translation parameters: ", str(T1_trans))
print("T1 rotation parameters: ", str(T1_rotat))

print("\nT2 translation parameters: ", str(T2_trans))
print("T2 rotation parameters: ", str(T2_rotat))

print("\nT3 translation parameters: ", str(T3_trans))
print("T3 rotation parameters: ", str(T3_rotat))

# instantiate RigidTransform objects
T1_RigTrans = RigidTransform(T1_rotat, T1_trans, imsize)
T2_RigTrans = RigidTransform(T2_rotat, T2_trans, imsize)
T3_RigTrans = RigidTransform(T3_rotat, T3_trans, imsize)

# warped images (composed transformations - T1, T1-T2, T1-T2-T3)
T1_T2_RigTrans = T2_RigTrans.compose(T1_rotat, T1_trans)

T2_T3_RigTrans = T3_RigTrans.compose(T2_rotat, T2_trans)
T1_T2_T3_RigTrans = T2_T3_RigTrans.compose(T1_rotat, T1_trans)

T1_compWarp = T1_RigTrans.warp(image)
print("\nFinished warping T1")
T1_T2_compWarp = T1_T2_RigTrans.warp(image)
print("Finished warping T1-T2")
T1_T2_T3_compWarp = T1_T2_T3_RigTrans.warp(image)
print("Finished warping T1-T2-T3")

create_dir("task3_results/exp1_composed_warp")
save_planes("task3_results/exp1_composed_warp", num_planes, T1_compWarp, "exp1_compWarp", "T1")
save_planes("task3_results/exp1_composed_warp", num_planes, T1_T2_compWarp, "exp1_compWarp", "T1-T2")
save_planes("task3_results/exp1_composed_warp", num_planes, T1_T2_T3_compWarp, "exp1_compWarp", "T1-T2-T3")

# warped images (sequential operation - T1+T2, T1+T2+T3)
T3_Warp = T3_RigTrans.warp(image)
T2_Warp = T2_RigTrans.warp(image)

T1_T2_seqWarp = T1_RigTrans.warp(T2_Warp)
print("Finished warping T1+T2")
T2_T3_seqWarp = T2_RigTrans.warp(T3_Warp)
T1_T2_T3_seqWarp = T1_RigTrans.warp(T2_T3_seqWarp)
print("Finished warping T1+T2+T3")

create_dir("task3_results/exp1_sequential_warp")
save_planes("task3_results/exp1_sequential_warp", num_planes, T1_T2_seqWarp, "exp1_seqWarp", "T1+T2")
save_planes("task3_results/exp1_sequential_warp", num_planes, T1_T2_T3_seqWarp, "exp1_seqWarp", "T1+T2+T3")

# ------------------------Comments on Experiment 1-----------------------------

# Warped images from composed transformations look similar to the warped images obtained from applying T2 and T3 sequentially on previously warped image. 
# Coordinate wise, both the images are same.
# The sequentially placed image looks more stretched than composed transformation image due to interpolation occuring twice in sequential operation.
# Also specific rotation and translation parameters obtained from the random sample across the range affects the image.
# These difference alos occur due to different number of steps in interpolation in both composed and sequential methods.


# -------------------------------Experiment 2----------------------------------
print("\n\nEXPERIMENT 2 - COMPOSED DDFs")

# instantiate RigidTransform object (T1-T2)
T2_RigTrans2 = RigidTransform(T2_rotat, T2_trans, imsize)
T1_T2_RigTrans2 = RigidTransform(T1_rotat, T1_trans, imsize, True)

# get first ddf (T1)
ddf1 = T2_RigTrans2.compute_ddf(imsize)

# get second ddf (T1-T2)
ddf2 = T1_T2_RigTrans2.compute_ddf(imsize, [ddf1])

# add ddfs (T1-T2)
T1_T2_RigTrans2.compose(inp_ddf1=ddf1, inp_ddf2=ddf2)

# instantiate RigidTransform object (T1-T2-T3)
T3_RigTrans2 = RigidTransform(T3_rotat, T3_trans, imsize)
T2_T3_RigTrans2 = RigidTransform(T2_rotat, T2_trans, imsize, True)
T1_T2_T3_RigTrans2 = RigidTransform(T1_rotat, T1_trans, imsize, True)

# get first ddf (T3)
ddf1 = T3_RigTrans2.compute_ddf(imsize)

# get second ddf (T2-T3)
ddf2 = T2_T3_RigTrans2.compute_ddf(imsize, [ddf1])

# get third ddf (T1-T2-T3)
ddf3 = T1_T2_T3_RigTrans2.compute_ddf(imsize, [ddf1 + ddf2])

# add ddfs (T1-T2-T3)
# self.composing_ddfs(inp_ddf1, inp_ddf2) if self.flag_composing_ddf == True
# Above condition of (self.flag_composing_ddf == True) present inside compose()
T1_T2_T3_RigTrans2.compose(inp_ddf1=ddf1+ddf2, inp_ddf2=ddf3)

# warp images with composed ddfs (T1-T2, T1-T2-T3)
T1_T2_compWarp2 = T1_T2_RigTrans2.warp(image)
print("\nFinished warping T1-T2")
T1_T2_T3_compWarp2 = T1_T2_T3_RigTrans2.warp(image)
print("Finished warping T1-T2-T3")

create_dir("task3_results/exp2_composed_warp_2")
save_planes("task3_results/exp2_composed_warp_2", num_planes, T1_T2_compWarp2, "exp2_compWarp2", "T1-T2")
save_planes("task3_results/exp2_composed_warp_2", num_planes, T1_T2_T3_compWarp2, "exp2_compWarp2", "T1-T2-T3")

# voxel intensity level difference between warped images (with and without composing_ddfs)
voxel_difference_T1_T2 = abs(T1_T2_compWarp2 - T1_T2_compWarp)
voxel_difference_T1_T2_T3 = abs(T1_T2_T3_compWarp2 - T1_T2_T3_compWarp)

# mean and standard deviation in voxel intensity level differences (with and without composing_ddfs)
mean_T1_T2 = np.mean(abs(T1_T2_compWarp2 - T1_T2_compWarp))
std_T1_T2 = np.std(abs(T1_T2_compWarp2 - T1_T2_compWarp))

mean_T1_T2_T3 = np.mean(abs(T1_T2_T3_compWarp2 - T1_T2_T3_compWarp))
std_T1_T2_T3 = np.std(abs(T1_T2_T3_compWarp2 - T1_T2_T3_compWarp))

print("\nMean difference between composed and non-composed image voxel intensities (T1-T2): %5.4f" %mean_T1_T2)
print("\nStandard deviation in difference between composed and non-composed image voxel intensities (T1-T2): %5.4f" %std_T1_T2)

print("\nMean difference between composed and non-composed image voxel intensities (T1-T2-T3): %5.4f" %mean_T1_T2_T3)
print("\nStandard deviation in difference between composed and non-composed image voxel intensities (T1-T2-T3): %5.4f" %std_T1_T2_T3)

# ------------------------Comments on Experiment 2-----------------------------

# Here, Experiment 1 was repeated with 'flag_composing_ddf=True'
# The composed images obtainted from compose() and composing_ddfs() look same
# The mean and standard deviation values of voxel-level difference between warped images are zero or extremely small value.
