#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <fstream>
#include <string>
#include <vector>
#include<iostream>
#include <iterator>


#define Pi 3.14


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


struct templatePointVector
{
	vector<int> templatePointX;
	vector<int> templatePointY;
};

struct Contours
{
	Mat dstImage;
	int centerX;
	int centerY;
};

struct offsetResult
{
	int grayDiff;
	int resultX;
	int resultY;
};

templatePointVector GetTemplate(Contours templateContours,double angle)
{
	/*模板点集*/
	templatePointVector templatePoint;

	for (int i = 0; i < templateContours.dstImage.cols; i++)
	{
		for (int j = 0; j < templateContours.dstImage.rows; j++)
		{
			if (templateContours.dstImage.at<uchar>(j,i)!=0)
			{
				//templatePoint.templatePointX.push_back(i);
				//templatePoint.templatePointY.push_back(j);

				/*旋转*/
				int cx = i - templateContours.centerX;
				int cy = j - templateContours.centerY;
				int rx = (int)cx * cos(angle) - cy * sin(angle) + templateContours.centerX;
				int ry = (int)cx * sin(angle) + cy * cos(angle) + templateContours.centerY;
				if (rx > 0 && rx < templateContours.dstImage.cols && ry>0 && ry < templateContours.dstImage.rows)
				{
					templatePoint.templatePointX.push_back(rx);
					templatePoint.templatePointY.push_back(ry);
				}

			}
		}
	}

	return templatePoint;
}


offsetResult TemplateMatch(Mat& image, Mat& imageRGB, templatePointVector& templatePoint, Mat& templateImage)
{
	int pixelStep = 10;
	int patchStep = 64;

	offsetResult result;
	result.grayDiff = 0;

	vector<int> templatePointX = templatePoint.templatePointX;
	vector<int> templatePointY = templatePoint.templatePointY;

	/*寻找模板图像轮廓的凸包矩形的左上角点*/
	vector<int>::iterator minTemplatePointX = min_element(begin(templatePointX), end(templatePointX));
	vector<int>::iterator minTemplatePointY = min_element(begin(templatePointY), end(templatePointY));
	Point minPosition = Point(*minTemplatePointX, *minTemplatePointY);

	for (int col = 0; col < image.cols; col += patchStep)
	{
		for (int row = 0; row < image.rows; row += patchStep)
		{
			cout << "col\\cols:" << col << "\\" << image.cols << endl;

			/*计算当前点与模板的结构性偏差*/
			int minX = minPosition.x;
			int minY = minPosition.y;
			int offsetX = col - minX;
			int offsetY = row - minY;
			//cout << offsetY << endl;
			/*获取当前点(row,col)与模板结构性信息结合之后的grayDiff*/
			double temp = 0;
			for (int i = 0; i < templatePointX.size(); i++)
			{
				/*模板图像中点 - 偏差 = 待测图像中对应的点*/
				Point pt = Point(templatePointX[i] - offsetX, templatePointY[i] - offsetY);

				/*计算当前待测点的邻域内灰度差信息*/
				if (pt.x >= 0 && pt.x < image.cols  && pt.y >= 2 * pixelStep && pt.y < image.rows - 2 * pixelStep)
				{
					double curPointValue = image.at<uchar>(pt.y, pt.x);
					double curPointValue_U1 = image.at<uchar>(pt.y, pt.x - 1 * pixelStep);
					double curPointValue_U2 = image.at<uchar>(pt.y, pt.x - 2 * pixelStep);
					double curPointValue_D1 = image.at<uchar>(pt.y, pt.x + 1 * pixelStep);
					double curPointValue_D2 = image.at<uchar>(pt.y, pt.x + 2 * pixelStep);

					temp += abs(4 * curPointValue - curPointValue_U1 - curPointValue_U2 - curPointValue_D1 - curPointValue_D2);
				}
			}

			if (temp > result.grayDiff)
			{
				//grayDiff = temp;
				result.grayDiff = temp;
				result.resultX = offsetX ;
				result.resultY = offsetY ;
			}
		}
	}

	//for (int i = 0; i < templatePointX.size(); i++)
	//{
	//	double x = templatePointX[i] - resultX;
	//	double y = templatePointY[i] - resultY;
	//	if (x >= 0 && x <= image.cols - 1 && y >= 0 && y <= image.rows - 1)
	//	{

	//		imageRGB.at<Vec3b>(y, x)[0] = 0;
	//		imageRGB.at<Vec3b>(y, x)[1] = 0;
	//		imageRGB.at<Vec3b>(y, x)[2] = 255;
	//	}
	//}
	
	//namedWindow("【匹配结果】", CV_WINDOW_NORMAL);
	//imshow("【匹配结果】", imageRGB);

	return result;
}


void PixelGrow(Mat srcImage, Mat& Curve, Mat& srcClone, Point pt, int Thres, int threshExpect, int LowerBind, int UpperBind)
{
	Point pToGrowing;								    //待生长点位置
	int pGrowValue = 0;                                 //待生长点灰度值
	double pSrcValue = 0;                               //生长起点灰度值
	double pCurValue = 0;                               //当前生长点灰度值

	//生长方向顺序数据
	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	vector<Point> growPtVector;							//生长点栈
	growPtVector.push_back(pt);                         //将生长点压入栈中

	pSrcValue = srcImage.at<uchar>(pt.y, pt.x);         //记录生长点的灰度值

	while (!growPtVector.empty())                       //生长栈不为空则生长
	{
		pt = growPtVector.back();                       //取出一个生长点
		growPtVector.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			pToGrowing.x = pt.x + DIR[i][0];
			pToGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (pToGrowing.x < 0 || pToGrowing.y < 0 || pToGrowing.x >(srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
				continue;
			if (Curve.at<uchar>(pToGrowing.y, pToGrowing.x) != 0)continue;

			pSrcValue = srcImage.at<uchar>(pt.y, pt.x);
			if (pGrowValue == 0)//如果标记点还没有被生长
			{
				pCurValue = srcImage.at<uchar>(pToGrowing.y, pToGrowing.x);
				if (pCurValue <= UpperBind && pCurValue >= LowerBind)
				{
					if (((abs(pCurValue - pSrcValue) < Thres) || (pCurValue > threshExpect)) && (pCurValue > threshExpect) && (pSrcValue > threshExpect)) //在阈值范围内则生长
					{
						srcClone.at<Vec3b>(pToGrowing.y, pToGrowing.x) = Vec3b(0, 0, 255);
						Curve.at<uchar>(pToGrowing.y, pToGrowing.x) = 255;//标记为白色
						growPtVector.push_back(pToGrowing);//将下一个生长点压入栈中
					}
				}
			}
		}
	}
}


Contours CurveComplete(Mat& srcImage, Mat& srcClone)
{
	Contours result;

	int thresholdValue = 50;
	int thresholdMax = 255;
	int Thres = 20;
	int threshExpect = 140;
	int LowerBind = 0; 
	int UpperBind = 255;
	int centerX = 0;
	int centerY = 0;
	int size = 0;

	Mat cannyOutput;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	Canny(srcImage, cannyOutput, 150, 255, 3, false);
	namedWindow("Canny output", CV_WINDOW_NORMAL);
	imshow("Canny output", cannyOutput);

	findContours(cannyOutput, contours, hierachy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dstImg = Mat::zeros(srcImage.size(), CV_8UC1);
	//画轮廓
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		double length = arcLength(contours[i], true);
		if (area > 50 && length > 50)
		{
			drawContours(dstImg, contours, i, Scalar(255), 1, 8, hierachy, 1, Point(0, 0));
			for (int j = 0; j < contours[i].size(); j++)
			{
				srcClone.at<Vec3b>(contours[i][j])[0] = 0;
				srcClone.at<Vec3b>(contours[i][j])[1] = 0;
				srcClone.at<Vec3b>(contours[i][j])[2] = 255;
				centerX += contours[i][j].x;
				centerY += contours[i][j].y;
				size++;
			}
		}
	}
	result.centerX = centerX / size;
	result.centerY = centerY / size;
	result.dstImage = dstImg;

	namedWindow("【轮廓图】", CV_WINDOW_NORMAL);
	imshow("【轮廓图】", dstImg);

	namedWindow("【srcClone】", CV_WINDOW_NORMAL);
	imshow("【srcClone】", srcClone);

	return result;
}


int main(int argc, char *argv[])
{
	Mat maskImage =     imread("E:\\数据集\\彩虹纹图像-0901\\02\\1.bmp", 0);
	Mat maskImageeRGB = imread("E:\\数据集\\彩虹纹图像-0901\\02\\1.bmp", 1);
	Mat image =         imread("E:\\数据集\\彩虹纹图像-0901\\01\\1.bmp", 0);
	Mat imageRGB =      imread("E:\\数据集\\彩虹纹图像-0901\\01\\1.bmp", 1);

	Contours templateContours = CurveComplete(maskImage, maskImageeRGB);
	namedWindow("templateImage", CV_WINDOW_NORMAL);
	imshow("templateImage", templateContours.dstImage);

	if (image.empty()|| maskImage.empty())
	{
		cout << "could not load image...\n" << endl;
	}


	double angle;
	templatePointVector templatePoint, bestTemplatePoint;
	offsetResult curResult, finalResult;
	finalResult.grayDiff = 0;

	for (angle = -30 * Pi / 180; angle < 30 * Pi / 180; angle += Pi / 180)
	{
		templatePoint = GetTemplate(templateContours, -angle);

		offsetResult curResult = TemplateMatch(image, imageRGB, templatePoint, templateContours.dstImage);

		if (curResult.grayDiff > finalResult.grayDiff)
		{
			finalResult.grayDiff = curResult.grayDiff;
			finalResult.resultX = curResult.resultX;
			finalResult.resultY = curResult.resultY;
			bestTemplatePoint = templatePoint;
		}
	}

	for (int i = 0; i < bestTemplatePoint.templatePointX.size(); i++)
	{
		double x = bestTemplatePoint.templatePointX[i] - finalResult.resultX;
		double y = bestTemplatePoint.templatePointY[i] - finalResult.resultY;
		if (x >= 0 && x <= image.cols - 1 && y >= 0 && y <= image.rows - 1)
		{

			imageRGB.at<Vec3b>(y, x)[0] = 0;
			imageRGB.at<Vec3b>(y, x)[1] = 0;
			imageRGB.at<Vec3b>(y, x)[2] = 255;
		}
	}

	cout << "best angle" << angle << endl;

	namedWindow("【匹配结果】", CV_WINDOW_NORMAL);
	imshow("【匹配结果】", imageRGB);

	waitKey(0);
	return 0;
}