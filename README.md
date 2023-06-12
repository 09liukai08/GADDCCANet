# GADDCCANet
__Requirement:__<br/>
  Python 3.8;<br/>
  Pytorch 1.11.0;<br/>
  scipy;<br/>
  sklearn;<br/>
  yaml.<br/>
 
__Data preparation:__<br/>
1.Prepare your image datasets and split them into training data and testing data.<br/>
<br/>
2.Obtain two distinct views from both the training and testing sets. Ensure that the shape of each data view is [N, H, W], where N represents the number of samples, and H and W are two dimensions of the data. Then save the two distinct views of the training data as train_view1.npy and train_view2.npy, respectively. Similarly, save the two distinct views of the testing data as test_view1.npy and test_view2.npy.<br/>
<br/>
3.Prepare the corresponding labels for the training data and testing data, saving them as train_labels.npy and test_labels.npy. The data shape should be [N], where N is the number of samples.<br/>
<br/>
4.Place the resulting train_view1.npy, test_view1.npy, train_view2.npy, test_view2.npy, train_labels.npy, and test_labels.npy files in the data folder.<br/>
<br/>

PS: if you want to have a quick try, you can just download the ORL and ETH data with two views from the Google drive link: <br/>https://drive.google.com/drive/folders/1p_HJj4lNxPi2lZxGiF9E1OrBCfexdisl?usp=drive_link
<br/>
For the two views of the ORL data, the first view is based on the original grayscale image data of ORL, its shape is [N, H, W], where N stands for the number of samples and H, W represent the height and width of the image, respectively. The second view is obtained from the data processed by the Local Binary Pattern (LBP) method, which also has the shape [N, H, W].<br/>
<br/>
For the two views of the ETH data, the first view is extracted from the Red (R) channel of the RGB image, its shape is [N, H, W], where N indicates the number of samples and H, W correspond to the image's height and width. The second view is derived from the Green (G) channel of the RGB image, which also has the [N, H, W] shape.<br/>
<br/>
__Set Parameter:__<br/>
You can skip this step if you use the default setting in the config file run.yaml.<br/>
<br/>
You can also try different setting by changing the value of the following part in the config file run.yaml:<br/><br/>
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
