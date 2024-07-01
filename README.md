# Image Processor with NVIDIA NPP and OpenCV

This project is a simple C++ program that parses command line arguments to set an input directory and an output directory. It loads all image files from the input directory using OpenCV, calculates the mean RGB values using NVIDIA NPP, resizes the images to 32x32 pixels, and saves the processed images to the output directory with a `.png` extension.

## Prerequisites

- C++17 compiler
- OpenCV library
- NVIDIA NPP library
- make

## Installation

1. **Install OpenCV**:

```sh
# For Ubuntu, to install opencv developer library:
sudo apt install libopencv-dev
```

2. **Install NVIDIA NPP** (Usually installed with CUDA library)

## Building the Project

### Downloading Images

1. data/ contains scripts that download images from Google's Open Image Dataset (validation) subset

2. subset-val.csv contains information about 100 files
This file was created by `head validation-images-with-rotation.csv -n 100 >subset-val.csv`, where validation-images-with-rotation.csv is from https://storage.googleapis.com/openimages/web/download_v7.html - in Image IDs Validation set.

3. First cd into `cd data`

4. `./csv-to-download-format.sh subset-val.csv >subset.txt` will convert to txt format

5. `python3 downloader.py subset.txt --download_folder ./images/` will download images

### Using CMake

1. Clone the repository:

    ```sh
    git clone https://github.com/mastershin/cuda-2d-image-prep-example/
    cd cuda-2d-image-prep-example
    ```

2. Build the project:

    ```sh
    make
    ```

### Without CMake

Alternatively, you can compile the project using `g++`:

```sh
g++ -std=c++17 -I$(pkg-config --cflags opencv4) -L$(pkg-config --libs opencv4) -L/path/to/npp/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lnppicc -lnppif -o image_processor ../image_processor.cpp
```

## Running

### Just calculating RGB mean values (omit --output)
```
./bin/image_processor --directory ./data/images/
```

### Calculate RGB Means, and process all the images by resizing with width x height, and save to output
```
./bin/image_processor --directory ./data/images/ --output ./output/ --width 32 --height 32
```
