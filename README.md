![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


---------------------------------------

# About This Fork

![Yolo logo](http://guanghan.info/blog/en/wp-content/uploads/2015/12/images-40.jpg)

This fork repository adds additional methods and options to make working with yolo from pjreddie more simple and generic. Several new features include:

   (1) Optional arguments passable via terminal call with less stringent order.
   
   (2) Known categories do not longer need to be modified within the source code but can be passed from an external text file at runtime.  
   
   (3) A help dialog listing calls and options for yolo.  
   
   (4) Additional flags for yolo while training, e.g., arbitrary ending of ground-truth files or selectable number of snapshot iterations.  
   
   (5) Test-mode detection on a given file list
   
   (6) More minor improvements.  

# Examples
0. Help dialog  
<code>./darknet yolo help
</code>  
This will print all supported modes to run yolo and lists possible configuration parameters and default values.  
   
1. Testing
<code>./darknet yolo test cfg/yolo.cfg [path-to-weights]/yolo.weights -c_filename data/dog.jpg -c_classes data/classnames_VOC.txt -draw 1 -write 1 -dest ./bboxes.txt
</code>  
This runs the pre-trained yolo network on the dog image, drawing bounding boxes to the image, and writing results to the file bboxes.txt
   
2. Training
<code>
./darknet yolo train cfg/yolo_finetuning_example.cfg [path-to-weights]/extraction.conv.weights -c_fl_train [path-to-your-data]/filelist_train.txt -c_dir_backup [path-to-store-snapshots-in] -c_classes data/classnames_VOC.txt
</code>  
NOTE: you do not longer need to adapt the source code to specify how many and which categories to use. Everything can be adjusted with the external file -c_classes

# Contact
If you have any suggestions for further enhancements or problems with applying the code to your task, contact me at [alexander.freytag@uni-jena.de ](alexander.freytag@uni-jena.de ).
Complementary,
you might want to check out the [Google Group](https://groups.google.com/forum/#!forum/darknet) to seek feedback from a broader audience.


---------------------------------------
  
# About The Fork by Guanghan Ning

![Yolo logo](http://guanghan.info/blog/en/wp-content/uploads/2015/12/images-40.jpg)

1. This fork repository adds some additional niche in addition to the darknet from pjreddie. e.g.

   (1) Read a video file, process it, and output a video with boundingboxes.
   
   (2) Some util functions like image_to_Ipl, converting the image from darknet back to Ipl image format from OpenCV(C).
   
   (3) Adds some python scripts to label our own data, and preprocess annotations to the required format by darknet.  
   
   ...More to be added

2. This fork repository illustrates how to train a customized neural network with our own data, with our own classes.

   The procedure is documented in README.md.
   
   Or you can read this article: [Start Training YOLO with Our Own Data](http://guanghan.info/blog/en/my-works/train-yolo/).

# DEMOS of YOLO trained with our own data
Yield Sign: [https://youtu.be/5DJVLV3P47E](https://youtu.be/5DJVLV3P47E)

Stop Sign: [https://youtu.be/0CQMb3NGlMk](https://youtu.be/0CQMb3NGlMk)

The cfg that I used is here: [darknet/cfg/yolo_2class_box11.cfg](https://github.com/Guanghan/darknet/blob/master/cfg/yolo_2class_box11.cfg)

The weights that I trained can be downloaded here: (UPDATED 1/13/2016)
[yolo_2class_box11_3000.weights](http://guanghan.info/download/yolo_2class_box11_3000.weights)

The pre-compiled software with source code package for the demo:
[darknet-video-2class.zip](http://guanghan.info/download/darknet-video-2class.zip)

You can use this as an example. In order to run the demo on a video file, just type: 

<code> 
./darknet yolo demo_vid cfg/yolo_2class_box11.cfg model/yolo_2class_box11_3000.weights /video/test.mp4
</code> 

If you would like to repeat the training process or get a feel of YOLO, you can download the data I collected and the annotations I labeled. 

images: [images.tar.gz](http://guanghan.info/download/images.tar.gz)

labels: [labels.tar.gz](http://guanghan.info/download/labels.tar.gz)

The demo is trained with the above data and annotations.

# How to Train With Customized Data and Class Numbers/Labels

1. Collect Data and Annotation
   
   (1) For Videos, we can use video summary, shot boundary detection or camera take detection, to create static images.
   
   (2) For Images, we can use [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool) to label objects. The data I used for the demo was downloaded from [Google Images](https://images.google.com/), and hand-labeled by my intern employees. (Just kidding, I had to label it myself. Damn it...) Since I am training with only two classes, and that the signs have less distortions and variances (compared to person or car, for example), I only trained around 300 images for each class to get a decent performance. But if you are training with more classes or harder classes, I suggest you have at least 1000 images for each class.

2. Create Annotation in Darknet Format 
   
   (1) If we choose to use VOC data to train, use [scripts/voc_label.py](https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py) to convert existing VOC annotations to darknet format.
   
   (2) If we choose to use our own collected data, use [scripts/convert.py](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py) to convert the annotations.

   At this step, we should have darknet annotations(.txt) and a training list(.txt).
   
   Upon labeling, the format of annotations generated by [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool) is:
   
   class_number
   
   box1_x1 box1_y1 box1_width box1_height
   
   box2_x1 box2_y1 box2_width box2_height
   
   ....
   
   After conversion, the format of annotations converted by [scripts/convert.py](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py) is:
   
   class_number box1_x1_ratio box1_y1_ratio box1_width_ratio box1_height_ratio
   
   class_number box2_x1_ratio box2_y1_ratio box2_width_ratio box2_height_ratio
   
   ....
   
   Note that each image corresponds to an annotation file. But we only need one single training list of images. Remember to put the folder "images" and folder "annotations" in the same parent directory, as the darknet code look for annotation files this way (by default). 
   
   You can download some examples to understand the format:
   
   [before_conversion.txt](http://guanghan.info/download/before_conversion.txt)
   
   [after_conversion.txt](http://guanghan.info/download/after_conversion.txt)
   
   [training_list.txt](http://guanghan.info/download/training_list.txt)
   
   
3. Modify Some Code

   (1) In [src/yolo.c](https://github.com/Guanghan/darknet/blob/master/src/yolo.c), change class numbers and class names. (And also the paths to the training data and the annotations, i.e., the list we obtained from step 2. )
   
       If we want to train new classes, in order to display correct png Label files, we also need to moidify and run [data/labels/make_labels] (https://github.com/Guanghan/darknet/blob/master/data/labels/make_labels.py)
   
   (2) In [src/yolo_kernels.cu](https://github.com/Guanghan/darknet/blob/master/src/yolo_kernels.cu), change class numbers.
   
   (3) Now we are able to train with new classes, but there is one more thing to deal with. In YOLO, the number of parameters of the second last layer is not arbitrary, instead it is defined by some other parameters including the number of classes, the side(number of splits of the whole image). Please read [the paper](http://arxiv.org/abs/1506.02640)  
       
       (5 x 2 + number_of_classes) x 7 x 7, as an example, assuming no other parameters are modified.  
       
       Therefore, in [cfg/yolo.cfg](https://github.com/Guanghan/darknet/blob/master/cfg/yolo.cfg), change the "output" in line 218, and "classes" in line 222.
       
   (4) Now we are good to go. If we need to change the number of layers and experiment with various parameters, just mess with the cfg file. For the original yolo configuration, we have the [pre-trained weights](http://pjreddie.com/media/files/extraction.conv.weights) to start from. For arbitrary configuration, I'm afraid we have to generate pre-trained model ourselves.
   
4. Start Training

   Try something like:

<code> 
   ./darknet yolo train cfg/yolo.cfg extraction.conv.weights
</code> 

# Contact
If you find any problems regarding the procedure, contact me at [gnxr9@mail.missouri.edu](gnxr9@mail.missouri.edu).

Or you can join the aforesaid [Google Group](https://groups.google.com/forum/#!forum/darknet); there are many brilliant people asking and answering questions out there.
