# CaSE-NeRF: Camera Settings Editing of Neural Radiance Fields

Perform controlled editing of camera settings in the scene
https://github.com/CPREgroup/CaSE-NeRF/blob/main/gif/gif.mp4

# Training
Run the following command, make sure the path is correct. You also need to change the path inside train.py to your data path.  
`python -m train --gin_file=configs/llff.gin --logtostderr`  

You can also train your own dataset, as long as it confroms to NeRF data format.  
