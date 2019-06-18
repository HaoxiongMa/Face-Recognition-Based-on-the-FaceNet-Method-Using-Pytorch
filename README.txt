
1.Run 'Train.py' to train the FaceNET model, 
  this process will take hundreds of epochs to reach a good performance 
  and will save all the model files to log.

##I have save my own trained model of epoch34 to the location 'checkpoint_epoch34.pth'##
##for convenience, no need to do the training if dont have time or CUDA.##
  
  For my training, it takes around 5min to run one epoch and one night to run 50 epochs for my NVIDIA GTX1070MQ
  and it ran out of memories at epoch 44

2.Run 'Face_Register.py' to register a new face. To add the face pictures, add the pictures(.img) with single face each 
  to the folder 'dataset/RegisterImages/yournameoftheface' first. More register pictures will give better performance.
  
  This process will save an .npy file with the 128D embedding vectors and a .txt file of the picture labels 
  to the folder\dataset\Embedding
  
  In the folder I already put the pictures of Ben Simmons and LeBron James.
  The register process will perform the embedding process and labal the faces with the names.

3.Run 'Predict.py' to perdict a pricture with faces and label the names. First need to put the prediction photo to the folder
  'dataset\PredictionImgs' and change the path name with your prediction pictures in the code of Predict.py.
  
  I already put two pictures with both Ben and LeBron to the folder and directly run the code will show the prediction.