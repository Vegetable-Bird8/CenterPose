#pragma once
#include<vector>
#include<iostream>
#include<algorithm>
#include<numeric>
#include "net.h"

#define NMS_UNION 1
#define NMS_MIN  2

/*
人体识别的信息
信息主要为：
bbox : float [x1,y1,x2,y2]          保存了矩形框的左上和右下的坐标
scores : float                      保存了置信度
hp :float [x1,y1,x2,y2,...,x17,y17] 保存了关节点的坐标
label : int                         保存了类别信息，人体姿态检测仅有一个类别 label=0 
*/

typedef struct ObjInfo {
    std::vector <float> bbox(5);  //初始化bbox 长度为5 分别为 x1,x2,y1,y2,area
	float score;
	std::vector<float> hps(34); //初始化关节点的坐标
	int label;
};
class CenterPoseDecoder {
public:
	CenterPoseDecoder();
	~CenterPoseDecoder();

	int init(std::string model_path);   //初始化加载函数


	//You can change the shape of input image by setting params :resized_w and resized_h
    // 检测并输出人脸信息
	int detect(ncnn::Mat &inblob, std::vector<ObjInfo>&objs, int resized_w,int resized_h,float scoreThresh = 0.3, float nmsThresh = 0.45);
    ncnn::Mat sigmoid(ncnn::Mat &heatmap);
private:
	void dynamicScale(float in_w, float in_h);
	void genIds(float * heatmap, int h, int w,int c, float thresh, std::vector<float> &ids);  //找出大于阈值的点的
	void nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold = 0.3,int type=NMS_MIN);
	void gen_hp_Ids(float * heatmap, int h, int w, int c, float thresh, std::vector<std::vector<float>>& ids);  //找出关节点大于阈值的点
	void decode(ncnn::Mat & heatmap  , ncnn::Mat &wh, ncnn::Mat &reg,ncnn::Mat & hm_hp  , ncnn::Mat &hps, ncnn::Mat &hp_offset,
		std::vector<ObjInfo>&objs, float scoreThresh, float nmsThresh);
private:
	ncnn::Net net;     //ncnn 实例
	//为固定的输出尺寸
	int d_h = 512;
	int d_w = 512;

	// 与标准的512*512大小之间的缩放尺度
	float d_scale_h;
	float d_scale_w;

	float scale_h;
	float scale_w;

	int image_h;
	int image_w;
};