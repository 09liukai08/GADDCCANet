# GADDCCANet
__Requirement:__<br/>
  Python 3.8;<br/>
  Pytorch 1.11.0;<br/>
  scipy;<br/>
  sklearn;<br/>
  yaml.<br/>
 
__Data preparation:__<br/>
1.Prepare your RGB image datasets and split them into training data and testing data.<br/>
<br/>
2.Extract the R channel from the RGB in the training data and testing data. The data shape should be [N, H, W], where N is the number of samples, H is the height of an image, and W is the width of an image. Save these as train_view1.npy and test_view1.npy, respectively. Similarly, extract the G channel and save it as train_view2.npy and test_view2.npy.<br/>
<br/>
3.Prepare the corresponding labels for the training data and testing data, saving them as train_labels.npy and test_labels.npy. The data shape should be [N], where N is the number of samples.<br/>
<br/>
4.Place the resulting train_view1.npy, test_view1.npy, train_view2.npy, test_view2.npy, train_labels.npy, and test_labels.npy files in the data folder.<br/>
<br/>

PS: if you want to have a quick try, you can just download the ORL and ETH data with two views from the Google drive link: <br/>https://drive.google.com/drive/folders/1p_HJj4lNxPi2lZxGiF9E1OrBCfexdisl?usp=drive_link
<br/>

__Set Parameter:__<br/>
You can skip this step if you use the default setting in the config file run.yaml.<br/>
<br/>
You can also try different setting by changing the value of the following part in the config file run.yaml:<br/>
  layers_number: 3  # the number of convolution layers, can be 2 or 3<br/>
  batch_size: 32  # adjust the batch_size according to your GPU's memory<br/>
  layer1_filter_size: 4 # the filter size for 1st convolution layer, Recommended setting is 4-8<br/>
  layer2_filter_size: 4 # the filter size for 2nd convolution layer, Recommended setting is 4-8<br/>
  layer3_filter_size: 4 # the filter size for 3rd convolution layer, Recommended setting is 4-8<br/>
  layer1_patch_size: 7  # the patch size for 1st convolution layer, Recommended setting is 4-7<br/>
  layer2_patch_size: 7  # the patch size for 2nd convolution layer, Recommended setting is 4-7<br/>
  layer3_patch_size: 7  # the patch size for 3rd convolution layer, Recommended setting is 4-7<br/>
  reg_term: 0.00001  # learning rate<br/>
  overlap_ratio: 0.5<br/>
  histblk_size: 7<br/>
  pca_keep_dim: 199 # the feature dimension after reducing by PCA<br/>
  svm_kernel: linear  # the kernel type for SVM classifer, the value can be one of these type: linear   poly   sigmoid   rbf<br/>

<br/>
__Run the model:__<br/>
python main.py<br/>
