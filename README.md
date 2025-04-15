
We use YawDD as our train dataset. 
This dataset contains two video datasets of drivers with various facial characteristics.
The videos are taken in real and varying illumination conditions.

- In the first dataset, a camera is installed under the front mirror of the car.   
- In the second dataset, the camera is installed on the driver’s dash.   

### * Data Proprocessing
There are a list of driver videos contained in dataset. We randomly select some video that contains driver yawning and label the video fragment with 0-Normal, 1-Yawing&&Talking, 2-Yawning.  

You can preprocess the data using yawn_split_video1.py, Key_Frame_use2.py, file_list_generator3.py in the utils folder

We've preprocessed some of the files and put them in the `extracted_face` folder
### * Start Training
The model Settings file is placed under the `model` folder
Change your working directory to `model` and run `bash train.sh`
- Note: If necessary, you should change your `caffe` path and data source path in `train.sh` and `train.prototxt` as well as `test.prototxt`.
