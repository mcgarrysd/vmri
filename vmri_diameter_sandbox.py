#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:51:48 2023

vmri_diameter

For experimenting with vessel diameter measurement

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

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

################################################################
# Processing 
dcm_dir = '/media/sean/ucalgary/Frayne_lab/data/microangiography/'
output_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/' 

groups = os.listdir(dcm_dir)
for g in groups:
    subjs = os.listdir(os.path.join(dcm_dir,g))
    for s in subjs:
        dcm2nii(os.path.join(dcm_dir,g,s), output_dir)

base_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/' 
subj_list = os.listdir(output_dir)

for s in subj_list:
    nii_files = os.listdir(os.path.join(base_dir,s))
    tof_files = [i for i in nii_files if 'tof.' in i]
    for t in tof_files:
        nii_path = os.path.join(base_dir,s,t)
        vmri.make_mip_nii(nii_path,bet = True)

for s in subj_list:
    s_dir = os.path.join(output_dir,s)
    vmri.quality_check_niftis(s_dir, '/media/sean/ucalgary/Frayne_lab/data/QA/micro/')
#################################################################
micro_dir = output_dir
tof_list = []
seg_list = []
save_path = '/media/sean/ucalgary/Frayne_lab/data/QA/micro_seg/'
for s in subj_list:
    nii_files = os.listdir(os.path.join(micro_dir,s))
    tof_files = [i for i in nii_files if 'tof_mip' in i]
    for t in tof_files:
        nii_path = os.path.join(micro_dir,s,t)
        _, vol ,_ = vmri.unpack_nii(nii_path)
        tof_list.append(vol)
        seg = vmri.segment_tof(nii_path)
        seg_list.append(seg)
        vmri.overlay_segmentation(vol, seg)
        file_name = s+'.png'
        plt.savefig(os.path.join(save_path,file_name), bbox_inches='tight')
        plt.close('all')


diam_subjs = []
diam_segs = []
diam_list = []

im = tof_list[5]
seg = seg_list[5]
subj = subj_list[5]

skel, edge_labels, bp = vmri.skeletonize_vm(seg)
#vm.overlay_segmentation(im, edge_labels)

test_segment = 25
diam, mean_diam, viz = vmri.single_segment_diameter(edge_labels, test_segment, seg, im, plot = False)
out_path = '/media/sean/ucalgary/Frayne_lab/data/QA/diam_viz/'
im_name = subj+'_segment'+str(test_segment)+'.png'
vm.overlay_segmentation(im, viz)
plt.title(subj_list[0]+ ' segment ' + str(test_segment))
plt.savefig(os.path.join(out_path,im_name), bbox_inches='tight')
diam_subjs.append(subj)
diam_segs.append(test_segment)
diam_list.append(diam)

for s,t in zip(subj_list, tof_list):
    file_name = s+'_orig.png'
    cv2.imwrite(os.path.join(out_path,file_name),t)



mean_diams = []
for i in diam_list:
    mean_diams.append(np.mean(i))

def single_segment_diameter(edge_labels, segment_number, seg, im, use_label = False, pad = True, plot = False):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels,pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im,pad_size)
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_median = segment_midpoint(segment)

    vx,vy = tangent_slope(segment, segment_median)
    bx,by = crossline_slope(vx,vy)
    
    viz = np.zeros_like(seg)
    cross_length = find_crossline_length(bx,by, segment_median, seg)
    
    if cross_length == 0:
        diameter = 0
        mean_diameter = 0
        return diameter, mean_diameter, viz
    
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        this_point = segment_inds[i]
        vx,vy = tangent_slope(segment, this_point)
        bx,by = crossline_slope(vx,vy)
        _, cross_index = make_crossline(bx,by, this_point, cross_length)
        if use_label:
            cross_vals = crossline_intensity(cross_index,seg)
            diam = label_diameter(cross_vals)
        else:
            cross_vals = crossline_intensity(cross_index, im)
            diam = fwhm_diameter(cross_vals, plot = plot)
        if diam == 0:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    
    if pad == True:
        im_shape = edge_labels.shape
        viz = viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]
    return diameter, mean_diameter, viz


def fwhm_diameter(cross_vals, plot = False):
    min_val = np.min(cross_vals)
    cross_vals = cross_vals-min_val
    peak = np.max(cross_vals)
    half_max = np.round(peak/2)
    
    peak_ind = np.where(cross_vals == peak)[0][0]
    before_peak = cross_vals[0:peak_ind]
    after_peak = cross_vals[peak_ind+1:]
    
    try:
        hm_before = np.argmin(np.abs(before_peak - half_max))
        hm_after = np.argmin(np.abs(after_peak - half_max))
    
        # +2 added because array indexing begins at 0 twice    
        diameter = (hm_after+peak_ind) - hm_before +2
    except:
        diameter = 0
    if plot == True:
        plt.figure()
        plt.plot(range(len(cross_vals)), cross_vals)
    return diameter




#################################################
subj_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/21177/'

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
            mask = vmri.segment_tof(os.path.join(data_path,i,k))
            hdr, label, aff = vmri.unpack_nii(os.path.join(data_path,i,label_name))
            label = label*bet_mask
            jacc = vm.jaccard(label, mask)
            tof_vols.append(vol)
            mask_vols.append(mask)
            label_vols.append(label)
            jacc_list.append(jacc)
            
im = tof_vols[5]
seg = mask_vols[5]
label = label_vols[5]

pad_size = 25
edge_labels = np.pad(edge_labels,pad_size)
seg = np.pad(seg, pad_size)
im = np.pad(im,pad_size)

diam_segments = [5, 78, 96, 109, 41, 79, 38, 3, 7]

skel, edges, bp = skeletonize_vm(seg)
edge_count, edge_labels = cv2.connectedComponents(edges)

diam_list = []
full_viz = np.zeros_like(im)
for i in diam_segments:
    _, temp_diam, temp_viz = visualize_vessel_diameter(edge_labels, i, seg, im, pad = False)
    full_viz = full_viz+temp_viz
    diam_list.append(temp_diam)
