#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:20:25 2023

Vessel metrics MRI

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
import pydicom as dicom

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# MRI FUNCTIONS
###################################################################

def dcm2nii(subject_dir, output_dir):
    all_files = os.listdir(subject_dir)
    dir_split = subject_dir.split('/')
    if len(dir_split[-1]) > 0:
        subject_name = dir_split[-1]
    else:
        subject_name = dir_split[-2]
    if os.path.exists(os.path.join(output_dir,subject_name)) == False:
        os.mkdir(os.path.join(output_dir,subject_name))
    new_out_dir = os.path.join(output_dir,subject_name)
    for f in all_files:
        this_dir = os.path.join(subject_dir,f)
        print(f)
        try:
            dicom2nifti.convert_directory(this_dir, new_out_dir)
        except:
            print('file ' + f + ' failed')
    return new_out_dir

def unpack_nii(path):
    nii = nib.load(path)
    hdr = nii.header
    vol = nii.get_fdata()
    aff = nii.affine
    return hdr, vol, aff

def brain_extraction(path):
    input_file = os.path.abspath(path)
    path_split = input_file.split('/')
    in_file_name = path_split[-1].split('.')[0]
    output_file_name = in_file_name+'_bet.nii.gz'
    output_path = '/'
    for i in path_split[0:len(path_split)-1]:
        output_path = os.path.join(output_path,i)
    output_path = os.path.join(output_path,output_file_name)
    result = fsl.BET(in_file = input_file, mask = True, out_file = output_path).run()
    return

def mip(volume,axis=2, mask = None):
    projection = np.max(volume, axis = axis)
    depth_map = np.argmax(volume, axis = axis)
    if mask is not None:
        depth_map = depth_map*mask
    return projection, depth_map

def load_dicom(path):
    ds = dicom.dcmread(path)
    output = ds.pixel_array
    return output

def make_nii(image, affine, header, output_file):
    new_nii = nib.Nifti1Image(image, affine, header)
    nib.save(new_nii, output_file)
    
def make_mip_nii(path_to_nifti, bet = False):
    if bet == True:
        brain_extraction(path_to_nifti)
    bet_file = path_to_nifti.split('.nii.gz')[0]+'_bet.nii.gz'
    hdr, vol, aff = unpack_nii(bet_file)
    projection, depth_math = mip(vol,2)
    mip_file_name = path_to_nifti.split('.nii.gz')[0]+'_mip.nii.gz'
    dm_file_name = path_to_nifti.split('.nii.gz')[0]+'_dm.nii.gz'
    full_path = path_to_nifti.split('/')
    full_path = full_path[0:-1]
    subj_dir = '/'
    for i in full_path:
        subj_dir = os.path.join(subj_dir,i)
    make_nii(projection, aff, hdr, os.path.join(subj_dir, mip_file_name))
    make_nii(projection, aff, hdr, os.path.join(subj_dir, dm_file_name))              
    create_blank_mask(os.path.join(subj_dir, path_to_nifti))


def apply_mask(vol_path, mask_path):
    hdr, vol, aff = unpack_nii(vol_path)
    hdr2, mask, aff2 = unpack_nii(mask_path)
    masked_vol = vol*mask
    out_path = mask_path.split('/')[-1]
    make_nii(masked_vol, aff, hdr,)
    
def create_blank_mask(path):
    path_split = path.split('/')
    file_base = path_split[-1].split('.')[0]
    file_name = file_base+'_label.nii.gz'
    hdr, vol, aff = unpack_nii(path)
    new_vol= np.zeros_like(vol)
    output_path = '/'
    path_split[-1] = file_name
    for i in path_split:
        output_path = os.path.join(output_path,i)
    make_nii(new_vol, aff, hdr, output_path)
    return    

def process_subject_directory(dcm_dir, output_dir):
    subj_dir = dcm2nii(dcm_dir,output_dir)
    make_mip_nii(subj_dir)
    
def quality_check_niftis(subj_dir, output_dir):
    file_list = os.listdir(subj_dir)
    file_list_r = [i for i in file_list if '.nii.gz' in i]
    subj_split = subj_dir.split('/')
    subj_split = [i for i in subj_split if len(i)>0]
    subj = subj_split[-1]
    for i in file_list_r:
        file_name = subj+'_'+i.split('.')[0]+'.png'
        hdr, vol, aff = unpack_nii(os.path.join(subj_dir,i))
        if len(np.unique(vol>1)):
            vol = vm.normalize_contrast(vol)
        print(i)
        if len(vol.shape)<4:
            if len(vol.shape)==2:
                cv2.imwrite(os.path.join(output_dir,file_name),vol)
                print('1d image')
            else:
                middle_slice = np.floor(vol.shape[2]/2).astype(np.uint16)
                out = vol[:,:,middle_slice]
                cv2.imwrite(os.path.join(output_dir,file_name),out)
                print('volume')
        else:
            print(file_name+' has more than 3 dims')
            
            
def overlay_segmentation(im,label, alpha = 0.5, im_cmap = 'gray', label_cmap = 'jet'):
    masked = np.ma.masked_where(label == 0, label)
    plt.figure()
    try:
        plt.imshow(im, im_cmap, interpolation = 'none')
        plt.imshow(masked, label_cmap, interpolation = 'none', alpha = alpha)
    except:
        plt.imshow(im, 'gray', interpolation = 'none')
        plt.imshow(masked, 'jet', interpolation = 'none', alpha = alpha)
    
    plt.show(block = False)
    
# SEGMENTATION FUNCTIONS
###################################################################
            
def segment_tof(tof_path, save = False):
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
    if save == True:
        file_base = path_split[-1].split('.')[0]
        file_name = file_base+'_vm_label.nii.gz'
        output_path = '/'
        path_split = path.split('/')
        path_split[-1] = file_name
        for i in path_split:
            output_path = os.path.join(output_path,i)
        vmri.make_nii(vessel_mask, aff, hdr)
    return vessel_mask

def skeletonize_vm(label):
    skel = skeletonize(label)
    _,_, skel = vm.prune_terminal_segments(skel)
    edges, bp = vm.fix_skel_artefacts(skel)
    new_skel = edges+bp
    _, edge_labels = cv2.connectedComponents(edges)
    return new_skel, edge_labels, bp


###############################################################
# Diameter functions
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

def single_segment_diameter(edge_labels, segment_number, seg, im, use_label = False, pad = True, plot = False):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels,pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im,pad_size)
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_median = vm.segment_midpoint(segment)

    vx,vy = vm.tangent_slope(segment, segment_median)
    bx,by = vm.crossline_slope(vx,vy)
    
    viz = np.zeros_like(seg)
    cross_length = vm.find_crossline_length(bx,by, segment_median, seg)
    
    if cross_length == 0:
        diameter = 0
        mean_diameter = 0
        print('diameter measurement failed on segment ' + str(segment_number))
        return diameter, mean_diameter, viz
    
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        this_point = segment_inds[i]
        vx,vy = vm.tangent_slope(segment, this_point)
        bx,by = vm.crossline_slope(vx,vy)
        _, cross_index = vm.make_crossline(bx,by, this_point, cross_length)
        if use_label:
            cross_vals = vm.crossline_intensity(cross_index,seg)
            diam = vm.label_diameter(cross_vals)
        else:
            cross_vals = vm.crossline_intensity(cross_index, im)
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
