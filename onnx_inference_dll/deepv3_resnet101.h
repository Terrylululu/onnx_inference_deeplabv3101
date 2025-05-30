#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>



using namespace std;
using namespace cv;
using namespace Ort;

struct Deep_config
{
	float witdh; // Confidence threshold
	float height;  // Non-maximum suppression threshold
	float numclasses;  //Object Confidence threshold
	string model_path;
	bool IsUseCUDA;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;
class TimerC
{
public:
	TimerC() : beg_(std::chrono::system_clock::now()) {}
	void reset() { beg_ = std::chrono::system_clock::now(); }

	void out(std::string message = "") {
		auto end = std::chrono::system_clock::now();
		std::cout << message << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg_).count() << "ms" << std::endl;
		reset();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	chrono::time_point<std::chrono::system_clock> beg_;
};

class Depplabv3
{
public:
	Depplabv3(Deep_config config);
	cv::Mat detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int num_class;

	float default_witdh;
	float default_height;
	float numclasses;

	Mat resize_image(Mat srcimg);
	vector<float> input_image_;
	void normalize_(Mat img);
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "deeplabv3resnet101");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<std::string> input_names;
	vector<std::string> output_names;

	std::array<const char*, 1> InputNames;
	std::array<const char*, 2> OutNames;  //注意这里的输出个数。std::array 在编译的时候已经确定固定的大小

	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};
