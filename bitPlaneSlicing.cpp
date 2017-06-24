//
//  Bit-Plane Slicing.cpp
//  
//
//  Created by Victor on 8/6/17.
//
//  Implementation of https://doi.org/10.1016/j.protcy.2015.02.107
//  "A Novel Approach for Document Image Binarization Using Bit-plane Slicing" - Karthika, Ajay (2015)

/*  Algorithm:
    1. Convert to grayscale
    2. Bit-plane slicing to 8-bit plane
    3. Calculation if image differences
    4. D-image and E-image
    5. Contrast Stretching
    6. Construction of binary image
    7. Formation of a new edge map
    8. Binarized image construction
    9. Post-processing
 
 Process would be divided as functions in the program
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <climits>

using namespace cv;
using namespace std;

double intensityFactor = 1.2;

void bitSlice(Mat inp, Mat bitPlanes[8]){
    //save bit planes
    for (int i=0; i<8; i++) {
        bitPlanes[i]=Mat(((inp/(1<<i)) & 1) * 255);
    }
}

void d_image(Mat gry, Mat bitIm, Mat toReturn){
    //calculate d_image by subtracting grayscale from bitplane
    toReturn=gry;
    for(int i=0;i<toReturn.rows;i++){
        for(int j=0;j<toReturn.cols;j++){
            toReturn.at<Vec3b>(i,j)[0]=-toReturn.at<Vec3b>(i,j)[0]+bitIm.at<uchar>(i,j);
            toReturn.at<Vec3b>(i,j)[1]=-toReturn.at<Vec3b>(i,j)[1]+bitIm.at<uchar>(i,j);
            toReturn.at<Vec3b>(i,j)[2]=-toReturn.at<Vec3b>(i,j)[2]+bitIm.at<uchar>(i,j);
        }
    }
    
}

void e_image(Mat gry, Mat bitIm, Mat toReturn){
    //calculate d_image by subtracting grayscale to the inverse bitplane weighted by intensityFactor
    bitwise_not(gry, toReturn);
    toReturn=toReturn*intensityFactor;
    for(int i=0;i<toReturn.rows;i++){
        for(int j=0;j<toReturn.cols;j++){
            toReturn.at<Vec3b>(i,j)[0]=toReturn.at<Vec3b>(i,j)[0]-bitIm.at<uchar>(i,j);
            toReturn.at<Vec3b>(i,j)[1]=toReturn.at<Vec3b>(i,j)[1]-bitIm.at<uchar>(i,j);
            toReturn.at<Vec3b>(i,j)[2]=toReturn.at<Vec3b>(i,j)[2]-bitIm.at<uchar>(i,j);
        }
    }
}

void contrastStretch(Mat srcImage, Mat dstImage){
    //contrast stretch through normalization
    
}

void foregroundDetection(Mat src, Mat dst){
    //apply gaussian filter
    GaussianBlur(src, src, Size(49, 49), 0, 0);
    //global thresholding
    //steps:
    //1. produce histogram
    //2. divide histogram
    //3. calculate variance from each division
    //4. get intensity of the smallest variance
    //5. use intensity as global threshold value
    //
    //using 64 hue bins
    int histSize = 64, curMin=0;
    float range[] = { 0, 255 } ;
    const float* histRange = { range };
    
    Mat hist, mean, stddev[histSize-4], min;
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    normalize(hist, hist, 0, hist.rows, NORM_MINMAX, -1, Mat() );
    hist=hist.t();
    for(int i=0;i<histSize-4;i++){
        Mat tempH=hist.colRange(i,i+4);
        meanStdDev(tempH.t(), mean, stddev[i]);
    }
    min=stddev[0];
    for(int i=0;i<histSize-4;i++){
        Mat temp=stddev[i]<min;
        if(cv::countNonZero(temp)>0){
            min=stddev[i];
            curMin=i;
        }
    }
    double thresholdValue=(curMin+1)*4;
    
    threshold(src, dst, thresholdValue, 255, 2);
}

void findEdgePixels(Mat src, Mat dst){
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
}

void cannyDetection(Mat src, Mat dst){
    int edgeThresh = 1;
    int lowThreshold=80;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    Mat detected_edges=src;
    
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo(dst, detected_edges);
}

int main( int argc, char** argv ){
    
    Mat src;
    Mat bitPlanes[8];
    Mat d_images[8];
    Mat e_images[8];
    Mat d_images_sharp[8];
    Mat e_images_sharp[8];
    Mat d_foreground[8];
    Mat d_laplace[8];
    Mat e_laplace[8];
    Mat e_canny[8];

    
    /// Load an image
    src = imread( argv[1], 1 );
    
    //first convert to grayscale
    cvtColor(src, src, CV_BGR2GRAY);
    
    bitSlice(src, bitPlanes);
    
    for(int i=0;i<8;i++){
        d_images[i] = src;
        e_images[i] = src;
        d_laplace[i] = src;
        e_laplace[i] = src;
        d_foreground[i] = src;
        d_images_sharp[i] = src;
        e_images_sharp[i] = src;
        e_canny[i] = src;
    }
    
    for (int i=0; i<8; i++){
        d_image(src, bitPlanes[i], d_images[i]);
        
        imshow("dsa", d_images[i]);
        waitKey(0);
    }
    for (int i=0; i<8; i++){
        e_image(src, bitPlanes[i], e_images[i]);
    }
    
    for (int i=0; i<8; i++){
        
        normalize(d_images[i], d_images_sharp[i], 0, 255, NORM_L1);
        
        
        foregroundDetection(d_images[i], d_foreground[i]);
        
        findEdgePixels(d_images[i], d_laplace[i]);
        findEdgePixels(e_images[i], e_laplace[i]);
        cannyDetection(e_images[i], e_canny[i]);
    }
    
}

