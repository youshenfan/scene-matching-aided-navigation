// imread_imwrite.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>



using namespace cv;
using namespace std;


//��������������������������������
//��Ȩ����������ΪCSDN�����������䡹��ԭ�����£���ѭ CC 4.0 BY - SA ��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
//ԭ�����ӣ�https ://blog.csdn.net/Lv0930Hui/article/details/80713303





//��������������������������������
//��Ȩ����������ΪCSDN�����������䡹��ԭ�����£���ѭ CC 4.0 BY - SA ��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
//ԭ�����ӣ�https ://blog.csdn.net/Lv0930Hui/article/details/80713303

int _tmain(int argc, _TCHAR* argv[])
{
	string dir_path = "../data/input/";//����Ŀ¼
	string out_path = "../data/output/";//���Ŀ¼
	vector<String> fileNames;
	cv::glob(dir_path, fileNames);//ȡ������Ŀ¼�е��ļ���
	int i = 0;
	while (i < fileNames.size()){
		cout << dir_path + fileNames[i];
		waitKey(1000000);
    Mat imgIn = imread(fileNames[i], 0);//��ȡ�Ҷ�ͼ��
	namedWindow("test");
	imshow("test", imgIn);
	waitKey(100);
	//please input the ORB process here


	//
	string outFileName = out_path+ fileNames[i].substr(15);
    imwrite(outFileName, imgIn);//������������д������ļ��У�
	i++;
	destroyAllWindows();
	}
	

	
	return 0;
}

