# Virtual Object Insertion
This repository contains the code to generate the object insertion results in paper [Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image, CVPR 2020](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). Please visit our [project webpage](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/) for more information. 

We hope that the code can help reproduce the photorealistic object insertion results in our paper and compare future lighting estimation methods more easily. Please consider citing the paper if you find the code useful. Please contact [Zhengqin Li](zhl378@eng.ucsd.edu) if you have any questions.

## Results and Comparisons
The original models were trained by extending the SUNCG dataset with an SVBRDF-mapping. Since SUNCG is not available now due to copyright issues, we are not able to release the original models. Instead, we rebuilt a new high-quality synthetic indoor scene dataset and trained our models on it. We will release the new dataset in the near future. Some object insertion results on real images generated using our network trained on the new dataset are shown below.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion_test.png)
The quantitative and qualitative comparisons with prior works on object insertion are shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion.png)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/quantitative_objectInsertion.png)
Compared with our networks trained on SUNCG-based dataset, our networks trained on new dataset can predict more consistent lighting color and intensity. Therefore, it achieves better performance in user study. 

## Prerequisites
To run our code, you will need
* matlab
* python 
* Renderer. Please find the renderer from this [link](https://github.com/lzqsd/OptixRenderer), which an Optix-based renderer implemented for this project.

## Data 
Please download the data from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion.ziphttp://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion_test.zip). It includes the inverse rendering predictions of the 20 examples from Garon et al. 2019 dataset (Example 1-20) and 4 examples (demo0-3) downloaded from the Internet. 

## Instructions
To reproduce the 20 object insertion results from Garon et al. 2019 dataset.
* Download the inverse rendering predictions from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion.ziphttp://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion_test.zip). Unzip the data in the same directory as the code. 
* Open `generateXMLread_ref.m`. On line 3 and 4, write down the path to your renderer and your python program. 
* Run `generateXMLread_ref.m`

The program will read the `im_*.mat` files in each `Example*` directory to generate the object insertion results. 

We also provide the code to generate the object insertion code from scratch. The instructions are as follows:
* Similarly, download the inverse rendering results from this [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion.ziphttp://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion_test.zip)
* Open `generateXML.m`. On line 4 and 5, enter the path to your renderer and your python program. 
* Several parameters from  can be used to adjust the object insertion results
  * `meshNewName`: The mesh used for object insertion. We provide two meshes: `sphere.obj` and `bunny.obj`. You can include your own mesh but you may need to scale and rotate the mesh accordingly so that they will look good. 
  * `r, g, b`: The diffuse color of the inserted object. We set them all to 0.8 in our experiments. 
  * `roughness`: The roughness value of the inserted object. The smaller the roughness value, the more specular the object will be. We set roughness value to 0.2 in all our experiments. 
  * `scale`: The size of the inserted object. The size of the inserted object is decided by the size of the plane you selected from the image (will be explained more detailedly in the following) and the scale parameter. 
  * `isNewMask`: Whether to select new plane or not. Sometimes, after selecting the plane, you may find the rendered object too small or too large. In that case, you can set `isNewMask` to be `false` and adjust `scale`, until you get the satisfactory object insertion result.
