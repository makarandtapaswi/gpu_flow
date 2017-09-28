GPU based optical flow extraction in OpenCV
====================
Forked from [Feichtenhofer] (https://github.com/feichtenhofer/gpu_flow)

### Features:
* OpenCV wrapper for Real-Time optical flow extraction on GPU
* Automatic directory handling using Qt
* Allows saving of optical flow to disk,
** either with clipping large displacements
** or by adaptively scaling the displacements to the radiometric resolution of the output image

### Dependencies
* [OpenCV 2.4] (http://opencv.org/downloads.html)
* [Qt 5.4] (https://www.qt.io/qt5-4/)
* [cmake] (https://cmake.org/)

### Installation
1. `mkdir -p build`
2. `cd build`
3. `cmake ..`
4. `make`

Test that your OpenCV supports ffmpeg, and can load videos. For example, you can use Python, `import cv2`, and check that `cv2.VideoCapture(filename)` returns an object that is valid.

### Configuration:
You should adjust the input and output directories by editing the variables `vid_path`, `out_path` and `out_path_jpeg` in `compute_flow.cpp`. Note that these folders have to exist before executing.

### Usage:
```
./compute_flow [OPTION]...
```

Available options:
* `start_video`: start with video number in `vid_path` directory structure [default 1]
* `gpuID`: use this GPU ID [default 0]
* `type`: use this flow method Brox = 0, TVL1 = 1 [default 1]
* `skip`: the number of frames that are skipped between flow calcuation [default 1]
* `save_jpg`: use this to enable saving jpeg images from the video [default 0]
* `vid_path`: picks up .mp4 | .avi videos from here
* `out_path`: saves optical flow images / bin files to here
* `jpg_path`: if `save_jpg`, stores video frames here


Additional features in `compute_flow.cpp`:
* `float MIN_SZ = 256`: defines the smallest side of the frame for optical flow computation
* `float OUT_SZ = 256`: defines the smallest side of the frame for saving as .jpeg
* `bool clipFlow = true;`: defines whether to clip the optical flow larger than [-20 20] pixels and maps the interval [-20 20] to  [0 255] in grayscale image space. If no clipping is performed the mapping to the image space is achieved by finding the frame-wise minimum and maximum displacement and mapping to [0 255] via an adaptive scaling, where the scale factors are saved as a binary file to `out_path`.

### Example:
```
./compute_flow gpuID=0 type=1 vid_path="" out_path=""
```


