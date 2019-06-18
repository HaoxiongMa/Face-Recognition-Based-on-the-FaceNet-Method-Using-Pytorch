from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from processingfiles import image_processing , file_processing
from MTCNN import Facedetection
import torch
import argparse
from torchvision import transforms
from models import FaceNetModel 
import cv2

resize_width = 160
resize_height = 160

def create_embedding(model_path, face_images):
    
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
    model   = FaceNetModel(embedding_size = args.embedding_size, num_classes = args.num_classes).to(device)
     
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])       

    data_transforms = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
    images = []
    for i in range(face_images.shape[0]):
        image = data_transforms(face_images[i,:,:,:])
        images.append(image)
    images = torch.stack(images).to(device)
    with torch.no_grad():
        emb = model(images).detach().numpy()
    return emb


def face_recognition_image(model_path,dataset_path, filename,image_path):

    dataset_emb,names_list = load_dataset(dataset_path, filename)

    face_detect = Facedetection()

    image=image_processing.read_image(image_path)

    bounding_box, points = face_detect.detect_face(image)
    bounding_box = bounding_box[:,0:4].astype(int)
    face_images = image_processing.get_crop_images(image, bounding_box, resize_height, resize_width, whiten = False)
    pred_emb = create_embedding(model_path, face_images)
    pred_name = compare_embadding(pred_emb, dataset_emb, names_list)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_processing.cv_show_image_text("face_recognition", bgr_image,bounding_box,pred_name)
    cv2.waitKey(0)

def load_dataset(dataset_path,filename):

    compare_emb=np.load(dataset_path)
    names_list=file_processing.read_data(filename)
    return compare_emb,names_list

def compare_embadding(pred_emb, dataset_emb, names_list):

    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
            dist_list.append(dist)
        min_value = min(dist_list)
        print(dist_list)
        if (min_value > 2.3):
            pred_name.append('unknow')
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
    return pred_name


    
if __name__=='__main__':
    model_path = 'checkpoint_epoch34.pth'
    dataset_path = 'dataset/Embedding/faceEmbedding.npy'
    filename ='dataset/Embedding/name.txt'
#    image_path = 'dataset/PredictionImgs/test.jpg'
    image_path = 'dataset/PredictionImgs/test_stranger.jpg'    
    face_recognition_image(model_path, dataset_path, filename,image_path)