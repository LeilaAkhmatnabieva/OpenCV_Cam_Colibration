#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

cv::Mat frame, gray;
int maxCorners = 4;
cv::RNG rng(12345);
const char* source_window = "Image";
std::vector<cv::Point2f> corners;

void goodFeaturesToTrack_Demo(int, void*);
bool isRotationMatrix(cv::Mat& R);
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R);

int main()
{

    // Creating vector to store vectors of 3D points 
    std::vector<std::vector<cv::Point3f> > objpoints;
    // Creating vector to store vectors of 2D points 
    std::vector<std::vector<cv::Point2f> > imgpoints;
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;

    objp.push_back(cv::Point3f(0, 0, 0));
    objp.push_back(cv::Point3f(0, 0.21, 0));
    objp.push_back(cv::Point3f(0.297, 0.21, 0));// [cm]
    objp.push_back(cv::Point3f(0.297, 0, 0));

    

    cv::String image = "image2.jpeg";

    /*std::cout << "Enter the path:";
    std::cin >> image;*/

    frame = cv::imread(image);
    if (frame.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::imshow("Image", frame);
    // Finding corners
    goodFeaturesToTrack_Demo(0, 0);
    cv::waitKey(0);

    objpoints.push_back(objp);
    imgpoints.push_back(corners);

    cv::destroyAllWindows();
    cv::Mat cameraMatrix, distCoeffs, R, T;
    /*
    * Performing camera calibration by
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the
    * detected corners (imgpoints)
    */
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
    cv::Mat R_mat;//Rotation matrix   
    cv::Rodrigues(R, R_mat);
    cv::Vec3f Angles = rotationMatrixToEulerAngles(R_mat);

    //std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    //std::cout << "distCoeffs : " << distCoeffs << std::endl;
    //std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;
    std::cout << "Angles: " << Angles << std::endl;


    std::ofstream result;
    result.open("Result.txt");
    if (result.is_open())
    {
        result << T << std::endl;
        result << Angles << std::endl;
    }
    return 0;
}

void goodFeaturesToTrack_Demo(int, void*)
{
    maxCorners = MAX(maxCorners, 1);

    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::Mat copy = frame.clone();
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, gradientSize, useHarrisDetector, k);
    std::cout << "** Number of corners detected: " << corners.size() << std::endl;
    int radius = 4;
    for (size_t i = 0; i < corners.size(); i++)//рисуем точки на изображении
    {
        cv::circle(copy, corners[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), cv::FILLED);
    }
    cv::namedWindow(source_window);
    imshow(source_window, copy);
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    cornerSubPix(gray, corners, winSize, zeroZone, criteria);
    for (size_t i = 0; i < corners.size(); i++)
    {
        std::cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << std::endl;
    }
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat& R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-6;
}
// Calculates rotation matrix to euler angles
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6; // If
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}
