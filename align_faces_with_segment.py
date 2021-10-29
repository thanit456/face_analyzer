# Face alignment and crop demo
# Uses MTCNN, FaceBoxes or Retinaface as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Retinaface+MobileFaceNet gives the best peformance
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021

from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from common.utils import BBox, drawBbox,drawLandmark,drawLandmark_multiple
from models.pfld_compressed import PFLDInference
from FaceBoxes import FaceBoxes
from PIL import Image
import matplotlib.pyplot as plt
import glob
import time
from utils.align_trans import get_reference_facial_points, warp_and_crop_face

import shapely 
from shapely.geometry import Polygon 
import json 
from collections import defaultdict
import math 


def get_segment(segment_path): 
    json_data = json.load(open(segment_path))
    segment = defaultdict(list)
    for i in range(len(json_data["shapes"])): 
        segment[json_data["shapes"][i]["label"]].append(Polygon(json_data["shapes"][i]["points"]))
    return segment

def get_perimeters_from_polygon(polygon):
    poly_x, poly_y = polygon.exterior.coords.xy
    poly_perimeters = [coord for coord in zip(poly_x.tolist(), poly_y.tolist())]
    return poly_perimeters

def rotate_points(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_polygon(polygon, origin=(0,0), degree=0):
    poly_perimeters = get_perimeters_from_polygon(polygon)
    return Polygon(rotate_points(poly_perimeters, origin, degree))  

def scale_image_and_polygon(img, segment, target_size):
    scaled_segment = defaultdict(list)

    scale_y = float(target_size[0]) / img.shape[1]
    scale_x = float(target_size[1]) / img.shape[0]

    scaled_image = cv2.resize(img, target_size)
    for label, polygons in segment.items(): 
        for polygon in polygons:
            # TODO_1: every points with ratios 
            # TODO_2: translate nose to the target point 
            # scaled_polygon = shapely.affinity.scale(polygon, xfact=scale_x, yfact=scale_y)
            # polygon_center = scaled_polygon.centroid 
            # xoff = polygon_center.x * (1-scale_x)
            # yoff = polygon_center.y * (1-scale_y)
            # scaled_polygon = shapely.affinity.translate(scaled_polygon, xoff=xoff, yoff=yoff)
            coords = list(zip(*polygon.exterior.coords.xy))
            print(f'coords = {coords}')
            scaled_coords = [(x * scale_x, y * scale_y) for (x,y) in coords]
            scaled_segment[label].append(Polygon(scaled_coords))
    return scaled_image, scaled_segment

def polygon_overlay_image(image, segments): 
  overlay = image.copy()
  alpha = 0.5 

  for label, polygons in segments.items(): 
    for polygon in polygons:
      int_coords = lambda x: np.array(x).round().astype(np.int32)
      exterior = [int_coords(polygon.exterior.coords)]

      cv2.fillPoly(overlay, exterior, color=(255, 255, 0))
      cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
  return overlay

def get_bbox_after_aligned(bbox):
    top_left = np.array([bbox.left, bbox.top])
    bottom_right = np.array([bbox.right, bbox.bottom])
    w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
    bottom_left = top_left + np.array([0, h])
    top_right = top_left + np.array([w, 0])
    
    bbox_x = [top_left[0], top_right[0], bottom_left[0], bottom_right[0]]
    bbox_y = [top_left[1], top_right[1], bottom_left[1], bottom_right[1]]
    return BBox([min(bbox_x), max(bbox_x), min(bbox_y), max(bbox_y)])

def create_face_heatmaps(max_width, max_height, segments, patch_size=(100,100)):
    num_y = math.ceil(max_height / patch_size[0])
    num_x = math.ceil(max_width / patch_size[1])

    face_heatmaps = dict() 
    err_polygons = list()
    err_current_rects = list()
    for label in segments.keys():
      face_heatmap = np.zeros((num_y, num_x))
      for i in range(num_y):  
          for j in range(num_x): 
              current_rect = Polygon([
                  [j*patch_size[1], i*patch_size[0]], 
                  [j*patch_size[1] + patch_size[1], i*patch_size[0]],
                  [j*patch_size[1] + patch_size[1], i*patch_size[0] + patch_size[0]],
                  [j*patch_size[1], i*patch_size[0] + patch_size[0]] 
              ]) 
              for polygon in segments[label]: 
                try: 
                  area = current_rect.intersection(polygon).area * 1. / current_rect.area
                except:
                  area = current_rect.intersection(polygon.buffer(0)).area * 1. / current_rect.area
                if area > 0:
                  face_heatmap[i][j] += 1
      face_heatmaps[label] = face_heatmap
    return face_heatmaps


parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='PFLD', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='FaceBoxes', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def load_model():
    if args.backbone=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone') 
    else:
        print('Error: not suppored backbone')    
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':

    aspect_ratios = [] # h/w
    widths = [] 
    heights = [] 
    noses = [] 
    cropped_filenames = [] 
    cropped_imgs = [] 
    cropped_segments= [] 

    filenames=glob.glob("my_data/images/*.jpg")


    if args.backbone=='MobileNet':
        out_size = 224
    else:
        out_size = 112 
    model = load_model()
    model = model.eval()

    for imgname in filenames:
        print(imgname)
        img = cv2.imread(imgname)

        segmentname = imgname.replace('images', 'segments').replace('.jpg', '.json')
        segment = get_segment(segmentname)

        org_img = Image.open(imgname)
        height,width,_=img.shape
       
        if args.detector=='FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(img)
        else:
            print('Error: not suppored detector')        
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        for k, face in enumerate(faces): 
            if face[4]<0.9: # remove low confidence detection
                continue
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h])*1.2)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face/255.0
            if args.backbone=='MobileNet':
                test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input= torch.autograd.Variable(input)
            start = time.time()
            if args.backbone=='MobileFaceNet':
                landmark = model(input)[0].cpu().data.numpy()
            else:
                landmark = model(input).cpu().data.numpy()
            end = time.time()
            print('Time: {:.6f}s.'.format(end - start))
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            img = drawLandmark_multiple(img, landmark)

            # crop and aligned the face
            lefteye_x=0
            lefteye_y=0
            for i in range(36,42):
                lefteye_x+=landmark[i][0]
                lefteye_y+=landmark[i][1]
            lefteye_x=lefteye_x/6
            lefteye_y=lefteye_y/6
            lefteye=np.array([lefteye_x,lefteye_y])

            righteye_x=0
            righteye_y=0
            for i in range(42,48):
                righteye_x+=landmark[i][0]
                righteye_y+=landmark[i][1]
            righteye_x=righteye_x/6
            righteye_y=righteye_y/6
            righteye=np.array([righteye_x,righteye_y])

            between_eye = (righteye + lefteye) / 2 

            nose=np.array(landmark[33])

            nose_to_between_eye = between_eye - nose 
            nose_to_top_edge = np.array([0, -1000])

            print(f'nose = {nose}')
            print(f'between_eye = {between_eye}')

            cv2.line(img, nose.astype(int), between_eye.astype(int), (0,0,255), 30)
            cv2.line(img, nose.astype(int), (nose + nose_to_top_edge).astype(int), (255,0,0), 30)

            norm_nose_to_between_eye = nose_to_between_eye / np.linalg.norm(nose_to_between_eye)
            norm_nose_to_top_edge = nose_to_top_edge / np.linalg.norm(nose_to_top_edge)

            angle = np.arccos(norm_nose_to_between_eye.dot(norm_nose_to_top_edge))

            if norm_nose_to_between_eye[0] < norm_nose_to_top_edge[0]:
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)

            print(f'angle = {angle}')

            # rotate image 
            rotation_mat = cv2.getRotationMatrix2D(nose, angle, 1)
            rotated_img = cv2.warpAffine(img, rotation_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

            # rotate image 
            new_bbox = get_bbox_after_aligned(new_bbox)
            print(new_bbox)

            rotated_img = drawBbox(rotated_img, new_bbox)

            rotated_img = rotated_img[new_bbox.top:new_bbox.bottom+1, new_bbox.left:new_bbox.right+1]
            
            # translate polygons corresponding to cropping face 

            rotated_segments = defaultdict(list)
            for label, polygons in segment.items(): 
                for polygon in polygons: 
                    processed_polygon = rotate_polygon(polygon, nose, angle)
                    processed_polygon = shapely.affinity.translate(processed_polygon, xoff=-new_bbox.left, yoff=-new_bbox.top)
                    rotated_segments[label].append(processed_polygon)
                
            noses.append(nose)

            cropped_filenames.append(imgname)
            cropped_imgs.append(rotated_img)
            cropped_segments.append(rotated_segments)

            h,w = rotated_img.shape[:2]
            aspect_ratios.append(h/w)
            widths.append(w)
            heights.append(h)

            print('----')
            print(rotated_img.shape)
            print('----')

    aspect_ratios = np.array(aspect_ratios)
    widths = np.array(widths)

    med_ar = np.median(aspect_ratios)
    med_width = int(np.median(widths))
    med_height = int(med_ar * med_width)

    all_dict = dict()
    all_dict['cropped_filenames'] = cropped_filenames
    all_dict['cropped_imgs'] = cropped_imgs 
    all_dict['cropped_segments'] = cropped_segments
    all_dict['aspect_ratios'] = aspect_ratios 
    all_dict['widths'] = widths 
    all_dict['heights'] = heights 
    all_dict['noses'] = noses
    import pickle 
    with open(f'results/all_dict.pkl', 'wb') as f:
        pickle.dump(all_dict, f)

    target_size = (med_width,med_height)

    combined_segments = defaultdict(list)

    for cropped_filename, cropped_img, cropped_segment in zip(cropped_filenames, cropped_imgs, cropped_segments):
        scaled_img, scaled_segment = scale_image_and_polygon(cropped_img, cropped_segment, target_size)
        overlay = polygon_overlay_image(scaled_img, scaled_segment)

        # overlay = polygon_overlay_image(cropped_img, cropped_segment)
        
        cv2.imwrite(os.path.join('my_aligned_data',os.path.basename(cropped_filename)), overlay)

        for label, polygons in scaled_segment.items():
            for polygon in polygons:
                combined_segments[label].append(polygon)

    face_heatmaps = create_face_heatmaps(med_width, med_height, combined_segments, patch_size=(100,100))

    import pickle
    with open('results/current_face_heatmaps.pkl', 'wb') as f:
        pickle.dump(face_heatmaps, f)

    

            # leftmouth=landmark[48]
            # rightmouth=landmark[54]
            # facial5points=[righteye,lefteye,nose,rightmouth,leftmouth]
            # warped_face = warp_and_crop_face(np.array(org_img), facial5points, reference, crop_size=(crop_size, crop_size))
            # warped_face = warp_and_crop_face(np.array(org_img), facial5points, reference, crop_size=(3000, 4000))
            # cv2.imwrite(os.path.join('results_aligned',os.path.basename(imgname)), cropped)
            
            
            # img_warped = Image.fromarray(warped_face)
            # save the aligned and cropped faces
            # img_warped.save(os.path.join('results_aligned', os.path.basename(imgname)[:-4]+'_'+str(k)+'.png'))  
            #img = drawLandmark_multiple(img, new_bbox, facial5points)  # plot and show 5 points   
        # save the landmark detections 
        # cv2.imwrite(os.path.join('results',os.path.basename(imgname)),img)

