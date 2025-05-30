#include "deepv3_resnet101.h"


Depplabv3::Depplabv3(Deep_config config)
{
	this->default_witdh = config.witdh;
	this->default_height = config.height;
	this->numclasses = config.numclasses;

	string model_path = config.model_path;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());

	////获得支持的执行提供者列表
	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
	//创建CUDA提供者选项
	OrtCUDAProviderOptions cudaOption{};
	//判断是否使用GPU，并检查是否支持CUDA
	if (config.IsUseCUDA && (cudaAvailable == availableProviders.end()))
	{
		std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
		std::cout << "Inference device: CPU" << std::endl;
	}
	else if (config.IsUseCUDA && (cudaAvailable != availableProviders.end()))
	{
		std::cout << "Inference device: GPU" << std::endl;
		//添加CUDA执行提供者
		sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
	}
	else
	{
		std::cout << "Inference device: CPU" << std::endl;
	}


	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto inputname = ort_session->GetInputNameAllocated(i, allocator).get();
		input_names.push_back(inputname);
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}

	for (int i = 0; i < numOutputNodes; i++)
	{
		auto outputname = ort_session->GetOutputNameAllocated(i, allocator).get();
		output_names.push_back(outputname);
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	// 推理的run 函数只认这种数据
	InputNames = { input_names[0].c_str() };
	OutNames = { output_names[0].c_str(),output_names[1].c_str() };

	this->inpHeight = input_node_dims[0][2]==-1? this->default_height : input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3] == -1 ? this->default_witdh : input_node_dims[0][3];
	
}

Mat Depplabv3::resize_image(Mat srcimg)
{
	Mat dstimg;
	cv::resize(srcimg, dstimg, cv::Size(this->inpWidth, this->inpHeight));
	return dstimg;
}

void Depplabv3::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	std::vector<float> mean = { 0.485f, 0.456f, 0.406f };
	std::vector<float> _std = { 0.229f, 0.224f, 0.225f };
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = ((pix / 255.0)- mean[2 - c])/ _std[2-c];
			}
		}
	}
}

cv::Mat Depplabv3::detect(Mat& frame)
{
	Mat dstimg = this->resize_image(frame);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs;
	try 
	{
		ort_outputs = ort_session->Run(RunOptions{ nullptr }, InputNames.data(), &input_tensor_, 1, OutNames.data(), output_names.size());// 开始推理
	}
	catch (const Ort::Exception& e) 
	{
		std::cerr << "Error during inference: " << e.what() << std::endl;
	}
	
	const float* preds = ort_outputs[0].GetTensorMutableData<float>();

	cv::Mat class_mat = cv::Mat(this->inpHeight, this->inpWidth, CV_8UC1, cv::Scalar(0));
	cv::Mat color_mat = cv::Mat(this->inpHeight, this->inpWidth, CV_8UC3, cv::Scalar(0));
	cv::Mat img_bgr = frame;
	cv::resize(img_bgr, img_bgr, cv::Size(this->inpWidth, this->inpHeight));
	for (unsigned int i = 0; i < this->inpHeight; ++i)
	{
		uchar *p_class = class_mat.ptr<uchar>(i);
		cv::Vec3b *p_color = color_mat.ptr<cv::Vec3b>(i);

		for (unsigned int j = 0; j < this->inpWidth; ++j)
		{
			// argmax
			unsigned int max_label = 0;
			float max_conf = preds[this->inpWidth*i + j];

			for (unsigned int l = 0; l < this->numclasses; ++l)
			{
				float conf = preds[this->inpWidth * i + j + this->inpWidth * this->inpHeight*l];
				if (conf > max_conf)
				{
					max_conf = conf;
					max_label = l;
				}
			}

			if (max_label == 0) continue;

			// assign label for pixel(i,j)
			p_class[j] = cv::saturate_cast<uchar>(max_label);
			// assign color for detected class at pixel(i,j).
			p_color[j][0] = cv::saturate_cast<uchar>((max_label % 10) * 20);
			p_color[j][1] = cv::saturate_cast<uchar>((max_label % 5) * 40);
			p_color[j][2] = cv::saturate_cast<uchar>((max_label % 10) * 20);

		}

	}

	cv::Mat out_img;
	cv::addWeighted(img_bgr, 0.5, color_mat, 0.5, 0., out_img);
	return out_img;
}

int main0()
{
	Deep_config yolo_nets = { 512, 512, 21 ,"DeeplabV3ResNet101/weights/deeplabv3_resnet101_coco.onnx" ,true};
	Depplabv3 Depplabv3(yolo_nets);
	string imgpath = "DeeplabV3ResNet101/images/test_lite_deeplabv3_resnet101.png";
	Mat srcimg = imread(imgpath);
	Depplabv3.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}