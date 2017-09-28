//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow estimation based on a theory for warping. ECCV 2004.
// [2] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM 2007.
//************************************************************************

#define N_CHAR 500
#define WRITEOUT_OF_IMGS 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <time.h>

#include <QDirIterator>
#include <QFileInfo>
#include <QString>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

float MIN_SZ = 256;
float OUT_SZ = 256;

bool clipFlow = true; // clips flow to [-20 20]
bool resize_img = true;

bool createOutDirs = true;

// Global variables for gpu::BroxOpticalFlow
const float alpha_ = 0.197;
const float gamma_ = 50;
const float scale_factor_ = 0.8;
const int inner_iterations_ = 10;
const int outer_iterations_ = 77;
const int solver_iterations_ = 10;

void converFlowMat(Mat& flowIn, Mat& flowOut,float min_range_, float max_range_)
{
    float value = 0.0f;
    for(int i = 0; i < flowIn.rows; i++)
    {
        float* Di = flowIn.ptr<float>(i);
        char* Ii = flowOut.ptr<char>(i);
        for(int j = 0; j < flowIn.cols; j++)
        {
            value = (Di[j]-min_range_)/(max_range_-min_range_);

            value *= 255;
            value = cvRound(value);

            Ii[j] = (char) value;
        }
    }
}

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
        float lowerBound, float higherBound) {
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flowIn.rows; ++i) {
        for (int j = 0; j < flowIn.cols; ++j) {
            float x = flowIn.at<float>(i,j);
            flowOut.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
        }
    }
    #undef CAST
}

int main( int argc, char *argv[] )
{
    GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1, rgb_out;
    Mat frame0_32, frame1_32, imgU, imgV;
    Mat motion_flow, flow_rgb;

    char cad[N_CHAR];
    struct timeval tod1;
    double t1 = 0.0, t2 = 0.0, tdflow = 0.0, t1fr = 0.0, t2fr = 0.0, tdframe = 0.0;

    int start_with_vid = 1;
    int gpuID = 0;
    int type = 1;  // 0: BroxFlow, 1: TVL1Flow
    int frameSkip = 1;
    int save_jpg = 0;
    std::string vid_path, out_path, jpg_path;

    const char* keys = "{ h  | help         | false | print help message }"
                       "{ v  | start_video  |  1    | start video id }"
                       "{ g  | gpuID        |  0    | use this gpu }"
                       "{ f  | type         |  1    | use this flow method }"
                       "{ s  | skip         |  1    | frame skip }"
                       "{ j  | save_jpg     |  0    | store jpeg video frames }"
                       "{ vp | vid_path     | null  | pick videos from this folder }"
                       "{ op | out_path     | null  | store optical flow images to this folder }"
                       "{ jp | jpg_path     | null  | store video JPEGs to this folder }" ;


    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: brox_optical_flow [options]" << endl;
        cout << "Avaible options: " << endl;
        cmd.printParams();
        return 0;
    }

    if (argc > 1) {
        // parse arguments
        start_with_vid = cmd.get<int>("start_video");
        gpuID = cmd.get<int>("gpuID");
        type = cmd.get<int>("type");
        frameSkip = cmd.get<int>("skip");
        save_jpg = cmd.get<int>("save_jpg");

        // parse and save paths
        vid_path = cmd.get<String>("vid_path");
        out_path = cmd.get<String>("out_path");
        jpg_path = cmd.get<String>("jpg_path");

        // print all arguments
        cout << "start video: " << start_with_vid
             << " | gpuID: " << gpuID
             << " | flow method: "<< type
             << " | frameSkip: " << frameSkip
             << " | store jpeg: " << save_jpg
             << endl;
        cout << " video path: " << vid_path << endl;
        cout << "output path: " << out_path << endl;
        cout << " JPEGs path: " << jpg_path << endl << endl;
    }

    // validate paths
    if (vid_path == "null") { cout << "Video path cannot be empty!" << endl; return 1; }
    if (out_path == "null") { cout << "Output path cannot be empty!" << endl; return 1; }
    if (save_jpg && (jpg_path == "null")) { cout << "JPEG dump path cannot be empty!" << endl; return 1; }

    // GPU device id info
    cv::gpu::setDevice(gpuID);
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    // setup optical flow methods
    cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);
    cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;

    // go through video path, find all files ending with .mp4 | .avi
    QString out_folder_jpeg, out_folder_u, out_folder_v;
    QString vpath = QString::fromStdString(vid_path);
    QDirIterator dirIt(vpath, QDirIterator::Subdirectories);

    int vidID = 0;
    std::string video, outfile_u, outfile_v, outfile_jpeg;

    for (; (dirIt.hasNext()); ) {
        // get next file
        dirIt.next();
        QString file = dirIt.fileName();

        // check file is of type avi | mp4
        if ((QFileInfo(dirIt.filePath()).suffix() == "mp4") || (QFileInfo(dirIt.filePath()).suffix() == "avi"))
            video = dirIt.filePath().toStdString();
        else
            continue;

        // make sure start_video id is lower/equal to this video file
        vidID++;
        if (vidID < start_with_vid)
            continue;

        // get video paths and filenames
        std::string fName(video);
        std::string path(video);
        size_t last_slash_idx = std::string::npos;
        if (!createOutDirs) {
            // Remove directory if present.
            // Do this before extension removal incase directory has a period character.
            cout << "removing directories: " << fName << endl;
            last_slash_idx = fName.find_last_of("\\/");
            if (std::string::npos != last_slash_idx)
            {
                fName.erase(0, last_slash_idx + 1);
                path.erase(last_slash_idx + 1, path.length());
            }
        } else {
            last_slash_idx = fName.find(vid_path);
            fName.erase(0, vid_path.length());
            path.erase(vid_path.length(), path.length());
        }

        // Remove extension if present.
        const size_t period_idx = fName.rfind('.');
        if (std::string::npos != period_idx)
            fName.erase(period_idx);

        // make folder to store "u" component
        out_folder_u = QString::fromStdString(out_path + "u/" + fName);

        // skip if output folder already exists
        bool folder_exists = QDir(out_folder_u).exists();
        if (folder_exists) {
            cout << "already exists: " << out_path << fName << endl;
            continue;
        }

        // skip if folder creation fails for some reason
        bool folder_created = QDir().mkpath(out_folder_u);
        if (!folder_created) {
            cout << "cannot create: " << out_path << fName << endl;
            continue;
        }

        // make folder to store "v" component
        out_folder_v = QString::fromStdString(out_path + "v/" + fName);
        QDir().mkpath(out_folder_v);

        // make folder to store JPEG frames
        if (save_jpg) {
            out_folder_jpeg = QString::fromStdString(jpg_path + fName);
            QDir().mkpath(out_folder_jpeg);
        }

        std::string outfile = out_path + "u/" + fName + ".bin";

        FILE *fx = fopen(outfile.c_str(),"wb");

        // load video
        cout << "Processing video " << video << endl;
        VideoCapture cap;
        try {
            cap.open(video);
            // print num-frames in video
            cout << "Video has " << cap.get(CV_CAP_PROP_FRAME_COUNT) << " frames." << endl;
        }
        catch (std::exception& e) {
            cout << e.what() << endl;
        }

        // confirm that video opened properly
        if( cap.isOpened() == 0 ) {
            cout << "Failed to open video " << video << endl;
            return -1;
        }

        // declare stuff
        int nframes = 0, width = 0, height = 0, width_out = 0, height_out = 0;
        float factor = 0, factor_out = 0;

        //  get first frame
        cap >> frame1_rgb_;

        if( resize_img == true )
        {
            // scale image to MIN_SZ on the lower dimension
            factor = std::max<float>(MIN_SZ/frame1_rgb_.cols, MIN_SZ/frame1_rgb_.rows);

            // make even numbers
            width = std::floor(frame1_rgb_.cols*factor);
            width -= width%2;
            height = std::floor(frame1_rgb_.rows*factor);
            height -= height%2;

            frame1_rgb = cv::Mat(Size(width,height),CV_8UC3);
            width = frame1_rgb.cols;
            height = frame1_rgb.rows;
            cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_CUBIC);

            factor_out = std::max<float>(OUT_SZ/width, OUT_SZ/height);

            rgb_out = cv::Mat(Size(cvRound(width*factor_out),cvRound(height*factor_out)),CV_8UC3);
            width_out = rgb_out.cols;
            height_out = rgb_out.rows;
        }
        else
        {
            frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
            width = frame1_rgb.cols;
            height = frame1_rgb.rows;
            frame1_rgb_.copyTo(frame1_rgb);
        }

        // Allocate memory for the images
        frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);  // 8-bit unsigned int, 3 channels
        flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
        motion_flow = cv::Mat(Size(width,height),CV_8UC3);
        frame0 = cv::Mat(Size(width,height),CV_8UC1);  // 8-bit unsigned int, 1 channel
        frame1 = cv::Mat(Size(width,height),CV_8UC1);
        frame0_32 = cv::Mat(Size(width,height),CV_32FC1);  // 32-bit float, 1 channel
        frame1_32 = cv::Mat(Size(width,height),CV_32FC1);

        // Convert the image to gray and float
        cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
        frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

        // MAIN LOOP OVER ALL VIDEO FRAMES!
        while( frame1.empty() == false )
        {
            // get timers
            gettimeofday(&tod1,NULL);
            t1fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            if( nframes >= 1 )
            {
                gettimeofday(&tod1,NULL);
                //  GetSystemTime(&tod1);
                t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
                switch(type){
                    case 0:
                        // upload frames to GPU
                        frame1GPU.upload(frame1_32);
                        frame0GPU.upload(frame0_32);
                        // Brox Flow
                        dflow(frame0GPU,frame1GPU,uGPU,vGPU);
                    case 1:
                        // upload frames to GPU
                        frame1GPU.upload(frame1);
                        frame0GPU.upload(frame0);
                        // TVL1 Flow
                        alg_tvl1(frame0GPU,frame1GPU,uGPU,vGPU);
                }
                // convert back to CPU
                uGPU.download(imgU);
                vGPU.download(imgV);
                // check timers
                gettimeofday(&tod1,NULL);
                t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
                tdflow = 1000.0*(t2-t1);
            }

            // Save optical flow images
            if ( WRITEOUT_OF_IMGS == true &&  nframes >= 1 ) {
                // resize optical flow images
                if( resize_img == true ) {
                    cv::resize(imgU,imgU,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
                    cv::resize(imgV,imgV,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
                }

                // min-max for u and v OF channels, use to possibly clip flow in range [-20, 20]
                double min_u, max_u;
                cv::minMaxLoc(imgU, &min_u, &max_u);
                float min_u_f = min_u;
                float max_u_f = max_u;

                double min_v, max_v;
                cv::minMaxLoc(imgV, &min_v, &max_v);
                float min_v_f = min_v;
                float max_v_f = max_v;

                // clip flow
                if (clipFlow) {
                    min_u_f = -20;
                    max_u_f = 20;

                    min_v_f = -20;
                    max_v_f = 20;
                }

                // placeholder 8-bit unsigned int 1 channel images
                cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
                cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);
                // convert matrices to images, scale properly
                convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
                convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

                // frame identifier (appended to paths)
                sprintf(cad,"/frame%06d.jpg",nframes);
                // prepare filenames for saving
                outfile_u = out_folder_u.toStdString();
                outfile_v = out_folder_v.toStdString();
                // save optical flow images
                imwrite(outfile_u+cad,img_u);
                imwrite(outfile_v+cad,img_v);

                // write min/max to binary file (scale back when feeding to network)
                fwrite(&min_u_f,sizeof(float),1,fx);
                fwrite(&max_u_f,sizeof(float),1,fx);
                fwrite(&min_v_f,sizeof(float),1,fx);
                fwrite(&max_v_f,sizeof(float),1,fx);
            }

            // Save the video frame as a JPEG image
            if (save_jpg) {
                outfile_jpeg = out_folder_jpeg.toStdString();

                // frame identifier (appended to paths)
                sprintf(cad,"/frame%06d.jpg",nframes + 1);
                // if image had been resized, restore before saving
                if( resize_img == true ) {
                    cv::resize(frame1_rgb,rgb_out,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
                    imwrite(outfile_jpeg+cad,rgb_out);
                }
                else
                    imwrite(outfile_jpeg+cad,frame1_rgb);
            }

            // prepare for next pair of frames
            frame1_rgb.copyTo(frame0_rgb);
            cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
            frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

            nframes++;
            // get next frame, skip some if required
            for (int iskip = 0; iskip<frameSkip; iskip++) {
                cap >> frame1_rgb_;
            }

            // if next raw frame is good (not empty), resize and copy for use
            if( frame1_rgb_.empty() == false ) {
                if( resize_img == true ) {
                    cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_CUBIC);
                }
                else {
                    frame1_rgb_.copyTo(frame1_rgb);
                }

                // convert to gray and float
                cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
                frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);
            }
            else {
                // video is finished!
                break;
            }

            gettimeofday(&tod1,NULL);
            t2fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            tdframe = 1000.0*(t2fr-t1fr);
            cout << "Processing video: " << setw(5) << fName
                 << " | Frame: " << setw(6) << nframes
                 << " | Time: " << fixed << setw(6) << setprecision(2) << tdframe << "ms"
                 << " (flow " << fixed << setw(6) << setprecision(2) << tdflow << "ms)"
                 << endl;
            // printf("Processing video: %5s | Frame: %06d | Time: %.2fms (flow %.2fms)\n", fName, nframes, tdframe, tdflow);
        }
        fclose(fx);
    }

    return 0;
}

