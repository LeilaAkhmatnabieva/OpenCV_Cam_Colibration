#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>


cv::Mat frame, gray;
int maxCorners = 4;
cv::RNG rng(12345);
const char* source_window = "Image";


// Creating vector to store vectors of 3D points 
std::vector<std::vector<cv::Point3f> > objpoints;
std::vector<cv::Point3f> objp;
// Creating vector to store vectors of 2D points 
std::vector<std::vector<cv::Point2f> > imgpoints;
std::vector<cv::Point2f> corners;

void goodFeaturesToTrack_Demo(int, void*);
bool isRotationMatrix(cv::Mat& R);
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R);
void SortCorners();


int main()
{
    cv::String image = "image.jpeg";
    std::ofstream result;

   /* std::cout << "Enter the path:";
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
    cv::destroyAllWindows();
    imgpoints.push_back(corners);


    for (int k = 0; k < 2; k++)
    {
        cv::Mat cameraMatrix, distCoeffs, R, T, RMat;
        if (k == 0)
        {
            objp.push_back(cv::Point3f(0, 0, 0));
            objp.push_back(cv::Point3f(0, 29.7, 0));
            objp.push_back(cv::Point3f(21, 29.7, 0));
            objp.push_back(cv::Point3f(21, 0, 0));
        }
        else
        {
            objp.clear();
            objp.push_back(cv::Point3f(0, 0, 0));
            objp.push_back(cv::Point3f(21, 0, 0));
            objp.push_back(cv::Point3f(21, 29.7, 0));
            objp.push_back(cv::Point3f(0, 29.7, 0));
        }

        objpoints.clear();
        objpoints.push_back(objp);

        for (int i = 0; i < 4; i++)
        {
            std::cout << "Object: " << objpoints[0][i].x << "," << objpoints[0][i].y << "," << objpoints[0][i].z << " = ";
            std::cout << "Image: " << imgpoints[0][i].x << "," << imgpoints[0][i].y << std::endl;
        }       

        bool FindCol = true;
        try {
            /*
       * Performing camera calibration by
       * passing the value of known 3D points (objpoints)
       * and corresponding pixel coordinates of the
       * detected corners (imgpoints)
       */
            cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.cols, gray.rows), cameraMatrix, distCoeffs, R, T);
        }
        catch (const std::exception& ex) {
            std::cout << ex.what() << std::endl;
            FindCol = false;
        }
        if (FindCol)
        {
            cv::Rodrigues(R, RMat);//Find rotation matrix
            cv::Vec3f Angles = rotationMatrixToEulerAngles(RMat);

            cv::Mat_<double> CamCoord;
            cv::Mat RInv = -RMat.inv();
            CamCoord = T;
            cv::Mat TTr = T.t();

            for (int i = 0; i < 3; i++)
            {
                CamCoord.at<double>(i) = 0;
                for (int j = 0; j < 3; j++)
                {
                    CamCoord.at<double>(i) += RInv.at<double>(i, j) * TTr.at<double>(j);
                }
            }
            std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
            std::cout << "distCoeffs : " << distCoeffs << std::endl;
            std::cout << "Rotation vector : " << R << std::endl;
            std::cout << "Rotation matrix : " << RMat << std::endl;
            std::cout << "Translation vector : " << T << std::endl;
            std::cout << "Angles: " << Angles << std::endl;
            std::cout << "Camera coordinates: " << CamCoord << std::endl;

            if (result.is_open())
            {
                result << T << std::endl;
                result << Angles << std::endl;
            }
            else
            {
                result.open("Result.txt");
                result << T << std::endl;
                result << Angles << std::endl;
            }
        }
        std::cout << "****************" << std::endl << std::endl;
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
    //std::cout << "** Number of corners detected: " << corners.size() << std::endl;
    int radius = 4;
    for (size_t i = 0; i < corners.size(); i++)
    {
        cv::circle(copy, corners[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), cv::FILLED);
    }
    cv::namedWindow(source_window);
    imshow(source_window, copy);
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    cornerSubPix(gray, corners, winSize, zeroZone, criteria);
    SortCorners();
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

void SortCorners()
{
    cv::Point2f temp[3];
    int MinY = 0;
    //find point with min y
    for (int i = 0; i < corners.size(); i++) {
        if (corners[i].y < corners[MinY].y) {
            MinY = i;
        }
    }

    cv::Point2f origin;
    origin = corners[MinY];

    sort(corners.begin(), corners.end(), [&origin](cv::Point2f p1, cv::Point2f p2) {
        cv::Point2f origin1, origin2;
        origin1.x = p1.x - origin.x;
        origin1.y = p1.y - origin.y;
        origin2.x = p2.x - origin.x;
        origin2.y = p2.y - origin.y;
        float area = origin1.x * origin2.y - origin1.y * origin2.x;
        return area > 0;
        });
}