# IOLCIA
This is the official code of Data Constraints? Not anymore: Image-Only Learning with Cross-Institutional Applicability for Ultrasound Video Segmentation (IOLCIA).
## Requirements
This is the basic code environment for this article. Missing packages can be installed directly using pip, depending on the error reported.
If you want to use your own local environment, you can try to match the installation based on the following packages.
- torch                          1.11.0+cu113
- torchvision                    0.12.0+cu113
- pydensecrf                     1.0
- Pillow                         9.1.1
- numpy                          1.23.5
- MedPy                          0.5.1
- matplotlib                     3.5.2
- matplotlib                     3.5.2
- imageio                        2.34.0
- ipykernel                      6.15.0
- ipython                        8.4.0
- ipython-genutils               0.2.0
- opencv-python                  4.9.0.80
- opencv-python-headless         4.9.0.80
- scikit-image                   0.21.0
- scikit-learn                   1.3.2
- scipy                          1.10.1
## File Description <br>
- All the codes are placed in the "./code", where the pth of the best model and testing results are included.<br>
- All codes related to data preprocessing are placed in "./code/Tools". <br>
- All codes related to the data loader are placed in "./code/dataloader/", where native codes for ms and fundus datasets are also included. Dataloader for IOLCIA is preserved in "./code/dataloader/breast_ultrasound". <br>
- All codes related to widgets are placed in "./code/models/", which is the main engine for IOLCIA. <br>
- All codes related to visualization results are placed in "./code/output/IOLCIA/breastultrasound/test-domain[5]/20240419_212429.686261". "Rect" represents that results are optimized using rectangular. <br>
- All codes related to tiny external methods deployed in IOLCIA are placed "./code/utils/".<br>
- The codes for training and testing are "./code/train.py" and "./code/test.py"<br>

## Dataset Description <br>

## SOTA Related
All SOTA models have been debugged and uploaded including codes and results to the cloud.
- Faster R-CNN / VFNet / GFocal / DyHead / Rf-next <br>
These models are all implemented using mmdetection: <br>
https://pan.baidu.com/s/1rc1FH8r6_ZobZHcwjn8fnw?pwd=6g66 
- HPE <br>
https://pan.baidu.com/s/1StSivqNdX3D63Kek7pdfQQ?pwd=dcdy
- PraNet <br>
https://pan.baidu.com/s/1xDSOr0bRqrigFryVmmvIcg?pwd=wzyx
- MMS <br>
https://pan.baidu.com/s/1yESRGpxq-gdh7u_AqrxDtw?pwd=hadi
- STM+ <br>
https://pan.baidu.com/s/1AnOVILeIJQHP4IVjMtn82A?pwd=gsgf
- CVA-Net <br>
https://pan.baidu.com/s/1-MBv8MbR_0nmHPvpqfYWPg?pwd=8sw2

## Visualization related <br>
All visualization results are uploaded to the clound, and it is free for you to download and have a look. <br>
All rectangularised results will be labelled with "rect".<br>
The images chosen for the article were run with the model at its best.<br>
https://pan.baidu.com/s/18itc5GLQ7Gamoqr8uoYBaA?pwd=8ev4

## Others
Remaining content will be added incrementally................................
