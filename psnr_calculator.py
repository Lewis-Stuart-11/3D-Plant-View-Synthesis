import cv2
import numpy as np
import os
import json
from torchmetrics.image import PeakSignalNoiseRatio
import torch
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM

def segment_imgs(image1, image2, mask):
    """Segments both images using the same mask, only returning pixel data included in the mask
    """
    return image1[mask != 0], image2[mask != 0]

def calculate_psnr(img1, img2, mask):
    """Calculates the PSNR metric of two images, after segmentation by a mask"""
    
    metric = PeakSignalNoiseRatio(data_range=1.0)

    segmented_img1, segmented_img2 = segment_imgs(img1, img2, mask)
    segmented_img1 = segmented_img1/255.0
    segmented_img2 = segmented_img2/255.0

    metric.update(torch.tensor(segmented_img1), torch.tensor(segmented_img2))
    
    return metric.compute().numpy().tolist() 

def calculate_masked_psnr_set(rendered_imgs_dir, gt_img_dir, mask_dir, output_path):
    """Calculates the masked PSNR for a set of images"""
    results = {}
    totals = [0,0,0]
    num_tests = len(os.listdir(rendered_imgs_dir))

    # Iterate through every image and perform the PSNR evaluation
    for filename in os.listdir(rendered_imgs_dir):
        results[filename] = {}

        image1_path = os.path.join(gt_img_dir, filename)
        image2_path = os.path.join(rendered_imgs_dir, filename)
        mask_path = os.path.join(mask_dir, filename + ".png")
        
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

        psnr_value = calculate_psnr(img1, img2, mask)

        results[filename]["psnr"] = psnr_value

        totals[0] += psnr_value

        print(filename, ":", round(psnr_value,3))
        
    print()
    print("Average PSNR: ", (totals[0]/num_tests))

    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
