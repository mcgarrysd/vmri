#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:19:46 2023

Preprocess svd data

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
dcm_dir = '/media/sean/ucalgary/Frayne_lab/data/microangiography/'
output_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/' 

subjs = os.listdir(dcm_dir+'Grp3_SVD')
for s in subjs:
    vmri.dcm2nii(os.path.join(dcm_dir,'Grp3_SVD',s), output_dir)

base_dir = '/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/' 
subj_list = os.listdir(output_dir)

subj_list = [s for s in subj_list if not 'G' in s]

for s in subj_list:
    nii_files = os.listdir(os.path.join(base_dir,s))
    tof_files = [i for i in nii_files if 'slab.' in i]
    for t in tof_files:
        nii_path = os.path.join(base_dir,s,t)
        vmri.make_mip_nii(nii_path,bet = True)

for s in subj_list:
    s_dir = os.path.join(output_dir,s)
    vmri.quality_check_niftis(s_dir, '/media/sean/ucalgary/Frayne_lab/data/QA/micro/svd1/')


hdr, vol, aff = vmri.unpack_nii('/media/sean/ucalgary/Frayne_lab/data/Processed/micro_angiography/20230601-230601-21970/6_3d_tof_3_slab_mip.nii.gz' )
