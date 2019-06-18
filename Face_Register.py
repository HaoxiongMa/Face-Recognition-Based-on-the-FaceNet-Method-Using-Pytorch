# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:30:30 2019

@author: Haoxiong Ma
@the part of the image_processing and file processing are
@from  @Author : panjq 
"""

import numpy as np
from processingfiles import image_processing , file_processing
from MTCNN import Facedetection
import os
import torch
import argparse
from torchvision import transforms
from models import FaceNetModel
from skimage import io   



resize_width = 160
resize_height = 160

#Create the 160x160 face images
def create_face(images_dir, out_face_dir):
    
    
    ##Generate face alignment database, save them to out_face_dir, 
    ##the database will be used to generate embedding database
    image_list,names_list = file_processing.gen_files_labels(images_dir,postfix='jpg')
    face_detect = Facedetection()
    for image_path ,name in zip(image_list, names_list):
        image = image_processing.read_image(image_path, resize_height=0, resize_width=0, normalization=False)
        bounding_box, points = face_detect.detect_face(image)
        bounding_box = bounding_box[:,0:4].astype(int)
        bounding_box = bounding_box[0,:]
        face_image = image_processing.crop_image(image,bounding_box)
        out_path = os.path.join(out_face_dir,name)
        face_image = image_processing.resize_image(face_image, resize_height, resize_width)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        basename = os.path.basename(image_path)
        out_path = os.path.join(out_path,basename)
        image_processing.save_image(out_path,face_image)


def create_embedding(model_path, emb_face_dir, out_emb_path, out_filename):
    
    ##Generate face embedding database to register the face using the trained model
    ##The embedding database is n*128 numpy array and is saved to out_emb_path

    parser = argparse.ArgumentParser(description = 'Face Recognition using Triplet Loss')    
    parser.add_argument('--start-epoch', default = 1, type = int, metavar = 'SE',
                        help = 'start epoch (default: 0)')
    parser.add_argument('--num-epochs', default = 40, type = int, metavar = 'NE',
                        help = 'number of epochs to train (default: 200)')
    parser.add_argument('--num-classes', default = 4000, type = int, metavar = 'NC',
                        help = 'number of clases (default: 10000)')
    parser.add_argument('--num-train-triplets', default = 5000, type = int, metavar = 'NTT',
                        help = 'number of triplets for training (default: 10000)')
    parser.add_argument('--num-valid-triplets', default = 5000, type = int, metavar = 'NVT',
                        help = 'number of triplets for vaidation (default: 10000)')
    parser.add_argument('--embedding-size', default = 128, type = int, metavar = 'ES',
                        help = 'embedding size (default: 128)')
    parser.add_argument('--batch-size', default = 50, type = int, metavar = 'BS',
                        help = 'batch size (default: 128)')
    parser.add_argument('--num-workers', default = 0, type = int, metavar = 'NW',
                        help = 'number of workers (default: 0)')
    parser.add_argument('--learning-rate', default = 0.001, type = float, metavar = 'LR',
                        help = 'learning rate (default: 0.001)')
    parser.add_argument('--margin', default = 0.5, type = float, metavar = 'MG',
                        help = 'margin (default: 0.5)')
    parser.add_argument('--train-root-dir', default = 'TrainingData', type = str,
                        help = 'path to train root dir')
    parser.add_argument('--valid-root-dir', default = 'ValidatingData', type = str,
                        help = 'path to valid root dir')
    parser.add_argument('--train-csv-name', default = 'TrainingData.csv', type = str,
                        help = 'list of training images')
    parser.add_argument('--valid-csv-name', default = 'ValidatingData.csv', type = str,
                        help = 'list of validtion images')

    args    = parser.parse_args()
    device  = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = FaceNetModel(embedding_size = args.embedding_size, num_classes = args.num_classes).to(device)
     
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    emb_face_dir = 'dataset/FaceImgData'
    
    image_list,names_list = file_processing.gen_files_labels(emb_face_dir,postfix='jpg')

    data_transforms = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
    images = []
    for image_path in image_list:
        image_path_os = os.path.join(str(image_path))
        image = io.imread(image_path_os)
        image = data_transforms(image)
        images.append(image)
    images = torch.stack(images).to(device)
    with torch.no_grad():
        compare_emb = model(images).detach().numpy()
    print("Finish resigter!")
    np.save(out_emb_path, compare_emb)

    file_processing.write_data(out_filename, names_list, model='w')


if __name__ == '__main__':
    
    # Creat the Training data using MTCNN, this process has been done

#    images_dir='dataset/images'
#    out_face_dir='TraniningData'
#    create_face(images_dir,out_face_dir)
 
    
    images_dir='dataset/RegisterImages'
    out_face_dir='dataset/FaceImgData'
    create_face(images_dir,out_face_dir)
    print("Finish face alignment processing!")

    model_path = 'checkpoint_epoch34.pth'
    emb_face_dir = 'dataset/FaceImgData'
    out_emb_path = 'dataset/Embedding/faceEmbedding.npy'
    out_filename = 'dataset/Embedding/name.txt'
    create_embedding(model_path, emb_face_dir, out_emb_path, out_filename)