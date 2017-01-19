# Building the data
## Step 1. Download LFW

- Go to http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz to download aligned LFW images
- Download the attributes file: http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt..
- Extract the zip files.
- Move the data to data/raw

In `data/raw` should have the following folder structure:

    ├──lfw_attributes.txt
    ├── lfw-deepfunneled
        ├──lots of images in folder with person's name

## Step 2. Build HDF5 LFW dataset

`python make_dataset.py`

optional arguments:

    -h, --help            show this help message and exit
    --img_size IMG_SIZE   Desired Width == Height
    --do_plot DO_PLOT     Whether to visualize saved images
    --batch_size BATCH_SIZE
                        Batch size for VGG predictions

## Note:
- Scripts for building black&white images hasn't been provided!
- Modified from [tdeboissiere](https://github.com/tdeboissiere).
