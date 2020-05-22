// ConsoleApplication1_pucancha.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <iostream>

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

// �ײв��ֻ�Ա�����һ��Ŀ��ͻ����ͼ��Ч���á��������ڸ�����������ͼ��
// https://blog.csdn.net/funsir/article/details/52924706

int _tmain(int argc, char** argv)
{
	//��ȡͼƬ
	const char *filename = (argc >= 2 ? argv[1] : "ex1.png");
	Mat I = imread(filename);
	if (I.empty())
	{
		return -1;
	}
	//��ɫͼת�ɻ�ɫͼ
	if (I.channels() == 3)
	{
		cvtColor(I, I, CV_RGB2GRAY);
	}
	Mat planes[] = { Mat_<float>(I), Mat::zeros(I.size(), CV_32F) };
	Mat complexI;
	//���츴��˫ͨ������
	merge(planes, 2, complexI);
	//���ٸ���Ҷ�任
	dft(complexI, complexI);
	Mat mag, pha, mag_mean;
	Mat Re, Im;
	//���븴����ʵ�����鲿
	Re = planes[0]; //ʵ��
	split(complexI, planes);
	Re = planes[0]; //ʵ��
	Im = planes[1]; //�鲿
	//�����ֵ
	magnitude(Re, Im, mag);
	//�������
	phase(Re, Im, pha);

	float *pre, *pim, *pm, *pp;
	//�Է�ֵ���ж�����
	for (int i = 0; i<mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = log(*pm);
			pm++;
		}
	}
	//�����׵ľ�ֵ�˲�
	blur(mag, mag_mean, Size(5, 5));
	//��ȡ����Ƶ�ײв�
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
	//��������ʵ�����鲿���˫ͨ����ʽ�ĸ�������
	merge(planes1, 2, complexI);
	// ����Ҷ���任
	idft(complexI, complexI, DFT_SCALE);
	//���븴����ʵ�����鲿
	split(complexI, planes);
	Re = planes[0];
	Im = planes[1];
	//�����ֵ�����
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
	//��һ����[0,255]����ʾ
	normalize(mag, invDFT, 0, 255, NORM_MINMAX);
	//ת����CV_8U��
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("SpectualResidual", invDFTcvt);
	imshow("Original Image", I);

	waitKey(0);
	return 0;
}
