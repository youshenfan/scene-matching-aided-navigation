// ConsoleApplication1_pucancha.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

// 谱残差方法只对背景单一，目标突出的图像效果好。不适用于复杂纹理背景的图像。
// https://blog.csdn.net/funsir/article/details/52924706

int _tmain(int argc, char** argv)
{
	//读取图片
	const char *filename = (argc >= 2 ? argv[1] : "ex1.png");
	Mat I = imread(filename);
	if (I.empty())
	{
		return -1;
	}
	//彩色图转成灰色图
	if (I.channels() == 3)
	{
		cvtColor(I, I, CV_RGB2GRAY);
	}
	Mat planes[] = { Mat_<float>(I), Mat::zeros(I.size(), CV_32F) };
	Mat complexI;
	//构造复数双通道矩阵
	merge(planes, 2, complexI);
	//快速傅里叶变换
	dft(complexI, complexI);
	Mat mag, pha, mag_mean;
	Mat Re, Im;
	//分离复数到实部和虚部
	Re = planes[0]; //实部
	split(complexI, planes);
	Re = planes[0]; //实部
	Im = planes[1]; //虚部
	//计算幅值
	magnitude(Re, Im, mag);
	//计算相角
	phase(Re, Im, pha);

	float *pre, *pim, *pm, *pp;
	//对幅值进行对数化
	for (int i = 0; i<mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = log(*pm);
			pm++;
		}
	}
	//对数谱的均值滤波
	blur(mag, mag_mean, Size(5, 5));
	//求取对数频谱残差
	mag = mag - mag_mean;

	for (int i = 0; i<mag.rows; i++)
	{
		pre = Re.ptr<float>(i);
		pim = Im.ptr<float>(i);
		pm = mag.ptr<float>(i);
		pp = pha.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = exp(*pm);
			*pre = *pm * cos(*pp);
			*pim = *pm * sin(*pp);
			pre++;
			pim++;
			pm++;
			pp++;
		}
	}
	Mat planes1[] = { Mat_<float>(Re), Mat_<float>(Im) };
	//重新整合实部和虚部组成双通道形式的复数矩阵
	merge(planes1, 2, complexI);
	// 傅立叶反变换
	idft(complexI, complexI, DFT_SCALE);
	//分离复数到实部和虚部
	split(complexI, planes);
	Re = planes[0];
	Im = planes[1];
	//计算幅值和相角
	magnitude(Re, Im, mag);
	for (int i = 0; i<mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = (*pm) * (*pm);
			pm++;
		}
	}
	GaussianBlur(mag, mag, Size(7, 7), 2.5, 2.5);
	Mat invDFT, invDFTcvt;
	//归一化到[0,255]供显示
	normalize(mag, invDFT, 0, 255, NORM_MINMAX);
	//转化成CV_8U型
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("SpectualResidual", invDFTcvt);
	imshow("Original Image", I);

	waitKey(0);
	return 0;
}
