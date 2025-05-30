#include<windows.h>
#include "deepv3_resnet101.h"
#include"dirent.h"


int read_files_in_dir2(const char* p_dir_name, std::vector<std::string>& file_names) {
	DIR* p_dir = opendir(p_dir_name);
	if (p_dir == nullptr) {
		return -1;
	}

	struct dirent* p_file = nullptr;
	while ((p_file = readdir(p_dir)) != nullptr) {
		if (strcmp(p_file->d_name, ".") != 0 &&
			strcmp(p_file->d_name, "..") != 0) {

			// 只寻找 jpg, bmp, png 格式的文件
			std::string file_name(p_file->d_name);
			std::string extension = file_name.substr(file_name.find_last_of('.') + 1);
			if (extension == "jpg" || extension == "jpeg" || extension == "bmp" || extension == "png") {
				/*std::string cur_file_name(p_dir_name);
				cur_file_name += "/";
				cur_file_name += file_name;*/
				std::string cur_file_name(p_file->d_name);
				file_names.push_back(cur_file_name);
			}
		}
	}

	closedir(p_dir);
	return 0;
}
void gen_train_deeplab_img()
{
	Deep_config deep_nets = { 512, 512, 21 ,"DeeplabV3ResNet101/weights/deeplabv3_resnet101_coco.onnx",true };
	Depplabv3 Depplabv3(deep_nets);

	std::vector<string> files;
	string rootPath = "DeeplabV3ResNet101/images/";
	string dstPath = "DeeplabV3ResNet101/OutputImage/";
	read_files_in_dir2(rootPath.c_str(), files);

	int j = 0;
	for (string file : files)
	{
		std::cout << file << std::endl;
		Mat img = cv::imread(rootPath + file);
		
		TimerC time;
		Mat colorImg =Depplabv3.detect(img);
		time.out("deeplab:");
		int k = 0;
		{
			imwrite(dstPath + file, colorImg);
		}
		j++;
	}
}
int main()
{
	gen_train_deeplab_img();
}