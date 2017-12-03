// DZ_OpenCV_1.cpp: ���������� ����� ����� ��� ����������� ����������.
//

#include "stdafx.h"

#include "opencv2\core\core.hpp"
#include "opencv2\flann\miniflann.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\photo\photo.hpp"
#include "opencv2\video\video.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\highgui\highgui.hpp"

#include "opencv2\core\core_c.h"
#include "opencv2\highgui\highgui_c.h"
#include "opencv2\imgproc\imgproc_c.h" 

#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
	IplImage* image = 0;
	IplImage* img_sobel1 = 0;
	IplImage* img_sobel2 = 0;
	IplImage* img_gray = 0;
	IplImage* img_canny = 0;
	IplImage* img_disttr = 0;
	IplImage* img_filter1 = 0;

	int xorder = 1;
	int yorder = 1;
	int aperture = 3;//������ ������� ������
	const char* path = "i8.jpg";//���� �� �����������

	// �������� ��������
	image = cvLoadImage(path, 1);
	// ������ ��������
	img_gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	img_canny = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	img_disttr = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	img_filter1 = cvLoadImage(path, 1);
	
	// ���� ��� ����������� ��������
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Canny", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("dist transform", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Filter", CV_WINDOW_AUTOSIZE);

	//� �������� ������
	cvCvtColor(image, img_gray, CV_RGB2GRAY);

	//����� �������� ������������ ������� ����������� ������ ���� 16 - ������(IPL_DEPTH_16S) ��� 8 - ������ �������� �����������.SOBEL
	//��� �������������� ������������� ����������� � 8 - ������ ����� ������������ cvConvertScale() ��� ConvertScaleAbs()		 SOBEL

	img_sobel1 = cvCreateImage(cvSize(img_gray->width, img_gray->height), IPL_DEPTH_16S, img_gray->nChannels);
	img_sobel2 = cvCreateImage(cvSize(img_gray->width, img_gray->height), img_gray->depth, img_gray->nChannels);
	
	// ��������� �������� ������
	cvSobel(img_gray, img_sobel1, xorder, yorder, aperture);
	
	// ����������� ����������� � 8-�������
	cvConvertScale(img_sobel1, img_sobel2);

	// �������� ������� �� ������������//Canny
	cvCanny(img_gray, img_canny, 100, 250, 3);

	//����� ����������
	Mat bw = cv::cvarrToMat(img_canny);
	threshold(bw, bw, 0, 255, THRESH_BINARY_INV);//distancetransform ������� ���������� �� 0 => �������� ������������ �����������
	Mat dist;									 //����� ����������
	distanceTransform(bw, dist, CV_DIST_L2, 5);
	normalize(dist, dist, 1.0, 0.0, NORM_MINMAX);//������������ ����� ��� �����������
	img_disttr = cvCloneImage(&(IplImage)dist);
	threshold(bw, bw, 0, 255, THRESH_BINARY_INV);//��� ����������� canny, �.�. bw = cv::cvarrToMat(img_canny);

	//���������� � ����������� �� dist*k ��� ������������� �����������(������ ������� (col2-col1)*(r2-r1))
	//double k = 1.0;
	//Mat src_img = cv::cvarrToMat(img_filter1);
	//Mat out = src_img;
	//for (int x = 0; x < cvGetSize(image).width; x++)
	//	for (int y = 0; y < cvGetSize(image).height; y++)
	//	{
	//		int r = dist.at<float>(y, x) * 255;
	//		int col1 = x-r * k;
	//		int r1 = y-r * k;
	//		int col2 = x+r * k;
	//		int r2 = y+r * k;
	//		//double coef = 1.0 / ((2.0*r * k + 1.0)*(2.0*r * k+1.0));//���� �� ����� ��������� ����������� � ��������� �������� 0
	//		double coef = 1.0 / (((double)(col2 - col1+1.0))*((double)(r2 - r1+1.0)));
	//		if (col1 < 0) col1 = 0;
	//		if (col2 >= cvGetSize(image).width) col2 = cvGetSize(image).width-1;
	//		if (r1 < 0) r1 = 0;
	//		if (r2 >= cvGetSize(image).height) r2 = cvGetSize(image).height-1;
	//		float R = 0;
	//		float G = 0;
	//		float B = 0;
	//		for (int i = r1; i < r2+1; i++)
	//			for (int j = col1; j < col2 + 1; j++) 
	//			{
	//				R += coef*src_img.at<Vec3b>(i, j)[2];
	//				G += coef*src_img.at<Vec3b>(i, j)[1];
	//				B += coef*src_img.at<Vec3b>(i, j)[0];
	//			}
	//		out.at<Vec3b>(y, x)[0] = (int)(B);
	//		out.at<Vec3b>(y, x)[1] = (int)(G);
	//		out.at<Vec3b>(y, x)[2] = (int)(R);
	//	
	//	}
	//img_filter1=cvCloneImage(&(IplImage)out);

	//���������� � ����������� �� dist*k � ������������ ������������(������ ������� (col2-col1)*(r2-r1))
	double k = 0.5;
	Mat src_img = cv::cvarrToMat(img_filter1);
	Mat out = src_img;
	Mat mat_integral;
	cv::integral(src_img, mat_integral,CV_64F);
	for (int x = 0; x < cvGetSize(image).width; x++)
		for (int y = 0; y < cvGetSize(image).height; y++)
		{
			float r = dist.at<float>(y, x)*255;
			int col1 = x - r * k;
			int r1 = y - r * k;
			int col2 = x + r * k;
			int r2 = y + r * k;
			//double coef = 1.0 / ((2.0*r * k + 1.0)*(2.0*r * k + 1.0));//���� �� ����� ��������� ����������� � ��������� �������� 0
			double coef = 1.0 / (((double)(col2-col1+1.0))*((double)(r2-r1+1.0)));
			if (col1 < 0) col1 = 0;
			if (col2 >= cvGetSize(image).width) col2 = cvGetSize(image).width-1;
			if (r1 < 0) r1 = 0;
			if (r2 >= cvGetSize(image).height) r2 = cvGetSize(image).height-1;
			cv::Vec3d sum = coef*(mat_integral.at<Vec3d>(r2+1,col2+1)- mat_integral.at<Vec3d>(r1,col2+1)- mat_integral.at<Vec3d>(r2+1,col1)+ mat_integral.at<Vec3d>(r1,col1));//���������� ������ �����
			out.at<Vec3b>(y, x) = sum;
		}
	img_filter1 = cvCloneImage(&(IplImage)out);

	

	// ���������� ��������
	cvShowImage("original", image);
	cvShowImage("gray", img_gray);
	cvShowImage("sobel", img_sobel2);
	cvShowImage("Canny", img_canny);
	cvShowImage("dist transform", img_disttr);
	cvShowImage("Filter", img_filter1);

	//����
	waitKey(0);

	// ����������� �������
	cvReleaseImage(&image);
	cvReleaseImage(&img_gray);
	cvReleaseImage(&img_canny);
	cvReleaseImage(&img_sobel1);
	cvReleaseImage(&img_sobel2);
	cvReleaseImage(&img_disttr);
	cvReleaseImage(&img_filter1);

	// ������� ����
	cvDestroyAllWindows();
	delete path;
	return 0;
}
