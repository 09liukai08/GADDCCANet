# GADDCCANet
Requirement:
  Python 3.8;
  Pytorch 1.11.0;
  scipy;
  sklearn;
  yaml.
 
Data preparation:
1.Prepare your RGB image datasets and split them into training data and testing data.
2.Extract the R channel from the RGB in the training data and testing data. The data shape should be [N, H, W], where N is the number of samples, H is the height of an image, and W is the width of an image. Save these as train_view1.npy and test_view1.npy, respectively. Similarly, extract the G channel and save it as train_view2.npy and test_view2.npy.
3.Prepare the corresponding labels for the training data and testing data, saving them as train_labels.npy and test_labels.npy.
4.Place the resulting train_view1.npy, test_view1.npy, train_view2.npy, test_view2.npy, train_labels.npy, and test_labels.npy files in the data folder.


Run the model:
python main.py
