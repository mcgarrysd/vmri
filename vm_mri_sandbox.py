#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:17:02 2023

vm_mri_sandbox

A development script for testing vessel metrics on mri data

@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance # Diameter measurement
import matplotlib.pyplot as plt
import os
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line # just in tortuosity
from bresenham import bresenham # diameter 
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import itertools # fixing skeleton
from math import dist
from aicsimageio import AICSImage
from skimage import data, restoration, util # deprecated preproc
import timeit
from skimage.morphology import white_tophat, black_tophat, disk
import pickle
import vessel_metrics as vm
import dicom2nifti
import nibabel as nib
from nipype import Workflow, Node, MapNode, Function
from nipype.interfaces.fsl import BET, IsotropicSmooth, ApplyMask
import nipype.interfaces.fsl as fsl
import vessel_metrics_MRI as vmri
from skimage.filters import gabor_kernel

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

#################################################################

test_input = '/media/sean/ucalgary/Frayne_lab/data/Development/21177/3D_Ax_Phase_Contrast_7/'
output_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/'
subject_dir = '/media/sean/ucalgary/Frayne_lab/data/Development/21177/'

test_nii = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/7_3d_tof_3_slab.nii.gz'
test_dcm = '/media/sean/ucalgary/Frayne_lab/data/Development/21177/700-COL3D_TOF_3_SLAB/1.2.840.113619.2.408.5554020.7748610.13029.1671199749.258.dcm'

subj_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/'

hdr, vol, aff = vmri.unpack_nii(test_nii)
projection, depth_map = vmri.mip(vol, axis = 2)

output_file = output_dir+'21177/angiography_mip_test.nii.gz'
vmri.make_nii(projection, aff, hdr, output_file)

new_nii = nib.Nifti1Image(projection, aff, hdr)
nib.save(new_nii, output_file)

ds = vmri.load_dicom(test_dcm)


proj_nii = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/10_3d_tof_3_slab_fa10_mip.nii.gz' 

hdr, tof, aff = vmri.unpack_nii(proj_nii)
im = vm.normalize_contrast(tof)
vmri.brain_extraction(proj_nii)

proj_bet = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/10_3d_tof_3_slab_fa10_mip_bet.nii.gz' 
hdr, tof_b, aff = vmri.unpack_nii(proj_bet)

t1_file = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/4_sag_3d-t1.nii.gz'
hdr, t1, aff = vmri.unpack_nii(t1_file)
t1_mip, t1_depth = vmri.mip(t1, axis = 2)

raw_ang_path = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/10_3d_tof_3_slab_fa10.nii.gz' 
hdr, tof_all, aff = vmri.unpack_nii(raw_ang_path)
vmri.brain_extraction(raw_ang_path)

hdr, ang_bet, aff = vmri.unpack_nii('/media/sean/ucalgary/Frayne_lab/data/Processed/21177/10_3d_tof_3_slab_fa10_bet.nii.gz')
new_mip, _ = vmri.mip(ang_bet, axis = 2)

vmri.process_subject_directory(subject_dir, output_dir)
###################################################################
file_list = os.listdir(subj_dir)
tof_ims = [i for i in file_list if 'tof' and 'mip' in i.lower()]
for i in tof_ims:
    path = os.path.join(subj_dir,i)
    _, vol, _ = vmri.unpack_nii(path)
    vm.show_im(vol)
    plt.title(i)

good_im_inds = [0,2,4,8]
tof_ims_r = []
for i in good_im_inds:
    tof_ims_r.append(tof_ims[i])

tof_vols = []
for i in tof_ims_r:
    path = os.path.join(subj_dir,i)
    _, vol, _ = vmri.unpack_nii(path)
    tof_vols.append(vol)
    vm.show_im(vol)


sigma1 = range(1,6,1)
for i in tof_vols:
    enh_sig = vm.jerman(i, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    vm.show_im(enh_sig)
    out = vm.subtract_background(i)
    vm.show_im(out)

this_tof = tof_vols[0]
this_tof = vm.normalize_contrast(this_tof)
tof1 = vm.subtract_background(this_tof)
tof1= cv2.fastNlMeansDenoising(tof1.astype(np.uint8), h = 5)
vm.show_im(tof1)
enh = vm.jerman(tof1, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
enh = vm.normalize_contrast(enh)
vm.show_im(enh)

new_dirs = ['21701', '21702', '21703','21731']
base_dir = '/media/sean/ucalgary/Frayne_lab/data/Development/' 
out_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/'
for i in new_dirs:
    vmri.process_subject_directory(os.path.join(base_dir, i), out_dir)

###############################################################
# test apply mask function
mask_path = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/7_3d_tof_3_slab_bet_mask.nii.gz' 

vol_path = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/8_3d_tof_12_slab.nii.gz'

hdr, vol, aff = unpack_nii(vol_path)
hdr2, mask, aff2 = unpack_nii(mask_path)
masked_vol = vol*mask
out_path = mask_path.split('/')[-1]

mask_proj, dm = vmri.mip(mask, axis = 2)
ang_proj, dm2 = vmri.mip(vol, axis = 2)

file_list = os.listdir(subj_dir)
mask_ims = [i for i in file_list if 'bet_mask' in i.lower()]
mask_list = []
for i in mask_ims:
    path = os.path.join(subj_dir, i)
    _, this_mask, _ = vmri.unpack_nii(path)
    mask_proj, _ = vmri.mip(this_mask, axis = 2)
    mask_list.append(mask_proj)
    
overlay_mask = np.zeros_like(mask_proj)
for i in mask_list:
    overlay_mask= overlay_mask + i
vm.overlay_segmentation(ang_proj, overlay_mask)
###############################################################
sigma1 = range(1,6,1)
enh_sig1 = meijering(tof, sigmas = sigma1, mode = 'reflect', black_ridges = False)

#enh_sig2 = meijering(im_enh, sigmas = sigma1, mode = 'reflect', black_ridges = False)

enh_sig3 = vm.jerman(tof, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')


mask = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/3d_tof_3_slab_fa10_mip_label.nii.gz' 

test = vm.subtract_local_mean(tof, size = 32)
vm.show_im(test)

hdr, label, aff = vmri.unpack_nii(mask)
vm.overlay_segmentation(tof, label)

vm.show_im(enh_sig3)

vm.overlay_segmentation(tof, enh)

tof_path = os.path.join(subj_dir, tof_ims_r[0])
def segment_tof(tof_path):
    hdr, tof_orig, aff = unpack_nii(tof_path)
    tof = vm.normalize_contrast(tof_orig)
    tof = vm.subtract_background(tof)

    tof = cv2.fastNlMeansDenoising(tof.astype(np.uint8), h = 5)

    sigma1 = np.arange(0.8,3,0.2)
    enh_sig1 = meijering(tof, sigmas = sigma1, mode = 'reflect', black_ridges = False)
    
    norm_enh = vm.normalize_contrast(enh_sig1)
    thresh = 5
    mask = np.zeros_like(tof)
    mask[norm_enh>thresh] = 1
    
    brain_mask = np.zeros_like(tof)
    brain_mask[tof>0] = 1
    kernel = np.ones((5,5), np.uint8)
    brain_mask_e = cv2.erode(brain_mask, kernel, iterations = 4)
    _, brain_mask_e = vm.fill_holes(brain_mask_e.astype(np.uint8), 1000)
    vessel_mask = brain_mask_e*mask
    vm.overlay_segmentation(tof_orig, vessel_mask)
    return vessel_mask
##################################################################
    
data_path = '/media/sean/ucalgary/Frayne_lab/data/Processed/' 
file_list = os.listdir(data_path)
tof_vols = []
mask_vols = []
label_vols = []
jacc_list = []
for i in file_list:
    this_file_list = os.listdir(os.path.join(data_path,i))
    tof_files = [i for i in this_file_list if 'mip.' in i]
    for k in tof_files:
        hdr, vol, aff = vmri.unpack_nii(os.path.join(data_path,i,k))
        label_name = k.split('.')[0]+'_label.nii.gz'
        if os.path.exists(os.path.join(data_path,i,label_name)):
            print(i,k)
            bet_mask_name = k.split('.')[0]+'_mask_e.nii.gz'
            hdr, bet_mask, aff = vmri.unpack_nii(os.path.join(data_path,i,bet_mask_name))
            mask = segment_tof(os.path.join(data_path,i,k))
            hdr, label, aff = vmri.unpack_nii(os.path.join(data_path,i,label_name))
            label = label*bet_mask
            jacc = vm.jaccard(label, mask)
            tof_vols.append(vol)
            mask_vols.append(mask)
            label_vols.append(label)
            jacc_list.append(jacc)
            
output_path = '/media/sean/ucalgary/Frayne_lab/smra_2023/diameter/'     
vm.overlay_segmentation(tof_vols[0], mask_vols[0]*2+label_vols[0])
print(np.mean(jacc_list))

for i in file_list:
    this_file_list = os.listdir(os.path.join(data_path,i))
    tof_files = [i for i in this_file_list if 'mip.' in i]
    for k in tof_files:
        label_name = k.split('.')[0]+'_label.nii.gz'
        if os.path.exists(os.path.join(data_path,i,label_name)):
            mask_name = k.split('.')[0]+'_mask.nii.gz'
            hdr, mask, aff = vmri.unpack_nii(os.path.join(data_path,i,k))
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations = 4)
            output_file = k.split('.')[0]+'_mask_e.nii.gz'
            vmri.make_nii(mask, aff, hdr, os.path.join(data_path,i,output_file))


def create_gaborfilter(num_filters = 16, ksize = 50, sigma = 3, lambd = 10, gamma = 0.5, psi = 0):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
    newimage = np.zeros_like(img)
    depth = -1 # remain depth same as original image
     
    for kern in filters:  
        image_filter = cv2.filter2D(img, depth, kern) 
        np.maximum(newimage, image_filter, newimage)
    return newimage

filters = create_gaborfilter(sigma = 4)
test_img = apply_filter(tof_orig, filters)
vm.show_im(test_img)


def seg_gabor(tof_path):
    hdr, tof_orig, aff = unpack_nii(tof_path)
    tof = vm.normalize_contrast(tof_orig)
    tof = vm.subtract_background(tof)
    filters = create_gaborfilter(sigma = 4)
    tof = apply_filter(tof, filters)
    tof = cv2.fastNlMeansDenoising(tof.astype(np.uint8), h = 5)
    sigma1 = np.arange(0.8,2,0.2)
    enh_sig1 = meijering(tof, sigmas = sigma1, mode = 'reflect', black_ridges = False)
    
    norm_enh = vm.normalize_contrast(enh_sig1)
    thresh = 30
    mask = np.zeros_like(tof)
    mask[norm_enh>30] = 1
    
    brain_mask = np.zeros_like(tof)
    brain_mask[tof>0] = 1
    kernel = np.ones((5,5), np.uint8)
    brain_mask_e = cv2.erode(brain_mask, kernel, iterations = 4)
    vessel_mask = brain_mask_e*also_bad_mask
    vm.overlay_segmentation(tof_orig, vessel_mask)
    return vessel_mask


####################################################################
# confocal image for figure for SMRA abstract
    
cpath = '/media/sean/ucalgary/from_home/vm_manuscript/E1_combined/Nov14_DMSO4/img.png' 
im = cv2.imread(cpath,0)
seg = vm.segment_image(im, thresh = 30)
vm.overlay_segmentation(im,seg)


#################################################################
vol_list = []
seg_list = []
tof_path_list = []
label_path_list = []
for i in file_list:
    this_file_list = os.listdir(os.path.join(data_path,i))
    tof_files = [i for i in this_file_list if 'mip.' in i]
    for k in tof_files:
        hdr, vol, aff = vmri.unpack_nii(os.path.join(data_path,i,k))
        label_name = k.split('.')[0]+'_label.nii.gz'
        if os.path.exists(os.path.join(data_path,i,label_name)):
            print(i,k)
            t_path = os.path.join(data_path,i,k)
            tof_path_list.append(t_path)
            l_path = os.path.join(data_path,i,label_name)
            label_path_list.append(l_path)
            seg = segment_tof(t_path)
            seg[seg>0] = 1
            skel, edges, bp = skeletonize_vm(seg)
            edge_count, edge_labels = cv2.connectedComponents(edges)
            vm.overlay_segmentation(vol, edge_labels)
            vol_list.append(vol)
            seg_list.append(seg)
            
D1= [50, 44, 78, 49, 43, 79]
D2 = [65, 43, 87, 49, 52, 78]

D1_out = []
D2_out = []
for v,s,x,y in zip(vol_list, seg_list,D1, D2):
    skel, edges, bp = skeletonize_vm(s)
    edge_count, edge_labels = cv2.connectedComponents(edges)
    test, d1_temp, temp_viz = visualize_vessel_diameter(edge_labels, x, s, v, pad = False)
    D1_out.append(d1_temp)
    _, d2_temp, temp_viz = visualize_vessel_diameter(edge_labels, y, s, v, pad = False)
    D2_out.append(d2_temp)
    
########################################################
confocal_path = '/media/sean/ucalgary/from_home/vm_manuscript/E1_combined/' 
all_files = os.listdir(confocal_path)
im_list_c = []
seg_list_c = []
label_list_c = []
jacc_list_c = []

for i in all_files:
    this_path = os.path.join(confocal_path,i)
    im = cv2.imread(os.path.join(this_path,'img.png'),0)
    label = cv2.imread(os.path.join(this_path,'label.png'),0)
    seg = vm.segment_image(im, thresh = 40)
    j = vm.jaccard(label, seg)
    im_list_c.append(im)
    label_list_c.append(label)
    seg_list_c.append(seg)
    jacc_list_c.append(j)
    
##############################################################
path = '/media/sean/ucalgary/Frayne_lab/data/microangiography/'
groups = os.listdir(path)
output_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/'
for g in groups:
    group_path = os.path.join(path,g)
    subj_list = os.listdir(group_path)
    for s in subj_list:
        subj_path = os.path.join(group_path, s)
        dcm2nii(subj_path,output_dir)
    