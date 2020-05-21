// imread_imwrite.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>



using namespace cv;
using namespace std;


//————————————————
//版权声明：本文为CSDN博主「歹伤殇」的原创文章，遵循 CC 4.0 BY - SA 版权协议，转载请附上原文出处链接及本声明。
//原文链接：https ://blog.csdn.net/Lv0930Hui/article/details/80713303





//————————————————
//版权声明：本文为CSDN博主「歹伤殇」的原创文章，遵循 CC 4.0 BY - SA 版权协议，转载请附上原文出处链接及本声明。
//原文链接：https ://blog.csdn.net/Lv0930Hui/article/details/80713303

int _tmain(int argc, _TCHAR* argv[])
{
	string dir_path = "../data/input/";//输入目录
	string out_path = "../data/output/";//输出目录
	vector<String> fileNames;
	cv::glob(dir_path, fileNames);//取得输入目录中的文件名
	int i = 0;
	while (i < fileNames.size()){
		cout << dir_path + fileNames[i];
		waitKey(1000000);
    Mat imgIn = imread(fileNames[i], 0);//读取灰度图像
	namedWindow("test");
	imshow("test", imgIn);
	waitKey(100);
	//please input the ORB process here


	//
	string outFileName = out_path+ fileNames[i].substr(15);
    imwrite(outFileName, imgIn);//将处理后的数据写入输出文件夹；
	i++;
	destroyAllWindows();
	}
	

	
	return 0;
}

