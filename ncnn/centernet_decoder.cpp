#include "centernet_decoder.h"
#include <map>
#include <string>
#include <cmath>
using namespace std;

CenterPoseDecoder::CenterPoseDecoder()
{
}

CenterPoseDecoder::~CenterPoseDecoder()
{
}

ncnn::Mat CenterPoseDecoder::sigmoid(ncnn::Mat& heatmap){

    float *heatmap_ = (float*)(heatmap.data);
    int w = heatmap.w;
    int h = heatmap.h;
    int channels = heatmap.c;
    int size = w * h;

    if (heatmap.empty())
	{
		std::cout << "heatmap is empty,Please check " << std::endl;
		return;
	}

    for (int q=0; q<channels; q++)
    {
        float* ptr = heatmap.channel(q);
 
        // 对每个元素做sigmoid操作
        for (int i=0; i<size; i++)
            ptr[i] = 1.0 / (1 + exp(-ptr[i] * 1.0));
    }
 

    return heatmap;
}

int CenterPoseDecoder::init(std::string model_path)
{
	std::string param = model_path + "mobilenet_v2.param";
	std::string bin= model_path + "mobilenet_v2.bin";
	net.load_param(param.data());
	net.load_model(bin.data());
	return 0;
}

int CenterPoseDecoder::detect(ncnn::Mat & inblob, std::vector<ObjInfo>& objs, int resized_w, int resized_h, float scoreThresh, float nmsThresh)
{
	if (inblob.empty()) {
		 std::cout << "Input is empty ,please check!" << std::endl;
		 return -1;
	}

	image_h = inblob.h;
	image_w = inblob.w;
	int image_c = inblob.c;

	scale_w = (float)image_w / (float)resized_w;
	scale_h = (float)image_h / (float)resized_h;

	ncnn::Mat in;




	//scale 
	dynamicScale(resized_w, resized_h);
	ncnn::resize_bilinear(inblob, in, d_w, d_h);


	float mean_vals_1[3]  = {0.485 * 255 , 0.456 * 255, 0.406 * 255} ;
	float norm_vals_1[3]  = {1.0/0.229/255, 1.0/0.224/255, 1.0/0.225/255} ;
	// float norm_vals_1[3]  = {1.0/127.5,1.0/127.5,1.0/127.5} ;
	
	in.substract_mean_normalize(mean_vals_1, norm_vals_1);

	ncnn::Extractor ex = net.create_extractor();
	ex.input("input", in);
	// 'hm' : 1,'hm_hp':17,'wh':2,'reg':2,'hps':34,'hp_offset':2
	ncnn::Mat heatmap, wh, reg, hm_hp , hps,hp_offset;
	ex.extract("hm", heatmap);
	ex.extract("wh", wh);
	ex.extract("reg", reg);
	ex.extract("hm_hp",hm_hp);
	ex.extract("hps",hps);
	ex.extract("hp_offset",hp_offset);

	
	//解码
	decode(heatmap, wh, reg, hm_hp , hps , hp_offset , objs, scoreThresh,nmsThresh);
	
	return 0;
}
/*
 找到大于阈值的点，并保存其行数、列数、通道数以及值 4个为一组
*/
void CenterPoseDecoder::genIds(float * heatmap, int h, int w, int c, float thresh, std::vector<float>& ids)
{
	if (heatmap==NULL)
	{
		std::cout << "heatmap is nullptr,please check! " << std::endl;
		return;
	}
	for (int id = 0; id < c ;id ++){
		float *temp_heatmap = heatmap + h*w *id;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				if (temp_heatmap[ i*w + j] > thresh  ) {
			
					ids.push_back(i);  //保存了 行数
					ids.push_back(j);  //保存了 列数
					ids.push_back(id); // 保存了 通道
					ids.push_back(temp_heatmap[ i*w + j]); //保存对应所应点的值 
				}
			}
		}
	}
}

void CenterPoseDecoder::gen_hp_Ids(float * heatmap, int h, int w, int c, float thresh, std::vector<vector<float>>& ids)
{

	if (heatmap==NULL)
	{
		std::cout << "heatmap is nullptr,please check! " << std::endl;
		return;
	}
	for (int id = 0;id<c;++id){
		float *temp_heatmap = heatmap + h*w *id;
		vector<float> joint;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (temp_heatmap[ i*w + j] > thresh  ) {
			
					joint.push_back(i);  //保存了 行数
					joint.push_back(j);  //保存了 列数
					// joint.push_back(id); // 保存了 通道
					joint.push_back(temp_heatmap[ i*w + j]); //保存对应所应点的值 
				}
			}
		}
		ids.push_back(joint);  //将当前的关节点放入索引中
	}





}
// 非极大值抑制 ，代替了原来的top_k函数
void CenterPoseDecoder::nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold,int type)
{
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
	[](const ObjInfo& a, const ObjInfo& b)   // 自定义函数 返回的是a<b
		{
			return a.score < b.score;   // 根据score从小到大排列
		});

	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;  // int类型列表

	std::multimap<float, int> vScores;  // 字典类型， key类型为float 值类型为 int
	const int num_boxes = input.size();  //检测到的包围框的数量
	vPick.resize(num_boxes);  // 为vPick分配num_boxes的空间
/* 	根据上面的for循环的语法定义 ++i 和 i++的结果是一样的，都要等代码块执行完毕才能执行语句3，但是性能是不同的。
	在大量数据的时候++i的性能要比i++的性能好原因：
	i++由于是在使用当前值之后再+1，所以需要一个临时的变量来转存。
 	而++i则是在直接+1，省去了对内存的操作的环节，相对而言能够提高性能
 */
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(input[i].score, i));   // vScores = { { scores,i }}
	}
	int nPick = 0;
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;   // rbegin为逆向迭代器 指向容器的最后一个  second指向容器的value first指向容器的key
		vPick[nPick] = last;     // 向vPick中赋值
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			//搜索左上角坐标的最大值，右下角坐标的最小值
			maxX = max(input.at(it_idx).bbox[0], input.at(last).bbox[0]);         // v[i] = v.at(i)  但at函数会检测函数是否越界
			maxY = max(input.at(it_idx).bbox[1], input.at(last).bbox[1]);
			minX = min(input.at(it_idx).bbox[2], input.at(last).bbox[2]);
			minY = min(input.at(it_idx).bbox[3], input.at(last).bbox[3]);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (type==NMS_UNION)
				IOU = IOU / (input.at(it_idx).bbox[4] + input.at(last).bbox[4] - IOU);
			else if (type == NMS_MIN) {
				IOU = IOU / ((input.at(it_idx).bbox[4] < input.at(last).bbox[4]) ? input.at(it_idx).bbox[4] : input.at(last).bbox[4]);
			}
			if (IOU > nmsthreshold) {
				it = vScores.erase(it);  // 删掉之后it自动指向下一个
			}
			else {
				it++;   //保留的话it++
			}
		}
	}

	vPick.resize(nPick);
	output.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		output[i] = input[vPick[i]];   // 只输出最后保留的几个对应的obj信息框
	}
}
/*
decode函数 对输出的图像进行解码
*/
void CenterPoseDecoder::decode(ncnn::Mat & heatmap  , ncnn::Mat &wh, ncnn::Mat &reg,ncnn::Mat & hm_hp  , ncnn::Mat &hps, ncnn::Mat &hp_offset,
  std::vector<ObjInfo>& objs, float scoreThresh, float nmsThresh)
{
	heatmap = sigmoid(heatmap);
	hm_hp = sigmoid(hm_hp);
	// 由于在网络中未进行sigmoid操作，故在解码之前先对其进行sigmoid操作
	// 后期训练网络中可以加入sigmoid操作，省去这步

	// 先对目标框进行decode
	int fea_h = heatmap.h;
	int fea_w = heatmap.w;
	int fea_c = heatmap.c;
	int spacial_size = fea_w*fea_h;  //用来确定指针跳过每个通道的元素数量

	float *heatmap_ = (float*)(heatmap.data);   // 指针指向矩阵头部


	float *wh_w = (float*)(wh.data); // wh_w 第一个通道为包围框的宽
	float *wh_h = wh_w + spacial_size;  // wh_h 为wh的第二个通道，是包围框的高

	float *reg_x = (float*)(reg.data);  // 中心点x的偏移量
	float *reg_y = reg_x + spacial_size; // 中心点y的偏移量

	float *offset_x = (float*)(hp_offset.data);  //关节点x的偏移量
	float *offset_y = offset_x + spacial_size;	//关节点y的偏移量

	int num_joints = hm_hp.c;    //人体关键点数量
	vector<float*> kps_x(num_joints);  //初始化kps指针vector 并分配空间
	vector<float*> kps_y(num_joints);  //初始化kps指针vector 并分配空间
	float *tmp_point = (float*)(hps.data);
	// 初始化hps的指针
	for (int i=0;i<num_joints;++i){
		kps_x[i] = tmp_point + spacial_size * 2*i;
		kps_y[i] = tmp_point + spacial_size *(2*i +1);
	}
	vector<float> ids;
	genIds(heatmap_,fea_h, fea_w,fea_c, scoreThresh, ids);  //生成大于阈值的点的indexs 共有ids.size//4个热力点
	vector<vector<float>> hp_ids;    // 保存了关节点的信息 y x and scores 共有ids.size//3个热力点
	gen_hp_Ids(hm_hp,fea_h,fea_w,num_joints,scoreThresh,hp_ids);   

	std::vector<ObjInfo> objs_tmp;
	vector<float> hm_kps(34);
	for (int i = 0; i < ids.size() / 4; ++i) {
		int id_h = ids[4 * i];
		int id_w = ids[4 * i + 1];
		int cate_id = ids[4 * i + 2];
		float score = ids[4 * i + 3];
		int index = id_h*fea_w + id_w;

		// float s0 = std::exp(scale0[index]) * 4;
		// float s1 = std::exp(scale1[index]) * 4;
		float s0 = wh_w[index] * 4;  // 扩大为原来的四倍
		float s1 = wh_h[index] * 4;

		float o0 = reg_x[index];  // 中心点对应的偏移量
		float o1 = reg_y[index];

		// std::cout << s0 << " " << s1 << " " << o0 << " " << o1 << std::endl;
		// x1，y1 为左上角的坐标 
		float x1 =  (id_w + o1 + 0.5) * 4 - s0 / 2 > 0.f ? (id_w + o1 + 0.5) * 4 - s0 / 2 : 0;
		float y1 =  (id_h + o0 + 0.5) * 4 - s1 / 2 > 0 ? (id_h + o0 + 0.5) * 4 - s1 / 2 : 0;
		float x2 = 0, y2 = 0;
		// x2,y2 为右下角坐标
		x1 = x1 < (float)d_w ? x1 : (float)d_w;
		y1 = y1 < (float)d_h ? y1 : (float)d_h;
		x2 =  x1 + s0 < (float)d_w ? x1 + s0 : (float)d_w;
		y2 = y1 + s1 < (float)d_h ? y1 + s1 : (float)d_h;

		// std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

		ObjInfo objbox;
		objbox.bbox[0] = x1;
		objbox.bbox[1] = y1;
		objbox.bbox[2] = x2;
		objbox.bbox[3] = y2;
		objbox.label = cate_id;  //类别
		objbox.score = score;	//置信分数
		objbox.bbox[4] = (x2-x1)*(y2-y1);


		float box_w = x2 - x1; //=s1?
		float box_h = y2 - y1; //=s0?

		// std::cout << objbox.x1 << " " << objbox.y1 << " " << objbox.x2 << " " << objbox.y2 << " " << objbox.label  << std::endl;

		// 初始化存放关节点坐标的容器
		// hm_kps 代表用人体中心+关节坐标点偏移量计算出来的关键点坐标
		// hp_kps 代表直接计算出来的关节点坐标

		vector<float> hm_kps(34);
		float kp_x,kp_y;  //初始化关节点偏移量的值
		for (int j =0;j<num_joints;++j){
			kp_x = (kps_x[j][index] + id_w) *4 >x1?(kps_x[j][index] + id_w)*4:x1;  //先判断左边界，是否大于左边界
			kp_y = (kps_y[j][index] + id_h) *4 >y1?(kps_y[j][index] + id_h)*4:y1;
			kp_x = kp_x < x2? kp_x:x2;   //判断右边界
			kp_y = kp_y < y2? kp_y:y2;

			vector<float> tmp_joint = hp_ids[j];   // 取出对应的关节点
			// int length = tmp_joint.size()/4;
			float min_dis = 100.0 ;  //初始化最小距离以及最小
			int min_dis_ind = -1;
			// 根据关节点热力图hm_hp计算出来的关节点坐标
			float hp_x = -1.0 ;
			float hp_y = -1.0 ;
			for (int k =0;k<tmp_joint.size()/3;k++){
				int hp_id_h = ids[3 * i];
				int hp_id_w = ids[3 * i + 1];
				float hp_score = ids[3 * i + 2];
				int hp_index = hp_id_h*fea_w + hp_id_w;

				float hp_offset_x = (offset_x[hp_index] + hp_id_w) *4 >0 ? (offset_x[hp_index] + hp_id_w) * 4 : 0;
				float hp_offset_y = (offset_y[hp_index] + hp_id_h) *4 >0 ? (offset_y[hp_index] + hp_id_h) * 4 : 0;
				if (hp_offset_x>x1 && hp_offset_x<x2 && hp_offset_y>y1 && hp_offset_y<y2){      //如果这个点在框内
					float dis = sqrt((kp_x-hp_offset_x)*(kp_x-hp_offset_x) + (kp_y-hp_offset_y)*(kp_x-hp_offset_x));  //计算L2距离
					if (dis<min_dis){
						min_dis = dis;
						min_dis_ind = k;
						hp_x = hp_offset_x;
						hp_y = hp_offset_y;
					}
				}
				else{
					continue;  // 如果点不在窗口内，直接跳过搜索下一个点
				}
			}
			if (min_dis_ind ==-1 | min_dis>3.0){   //如果循环完毕找不到符合条件的点 或者符合条件的点与坐标之间的坐标相差过大
				// 直接取用原来的点
				hm_kps[2*j] = kp_x;
				hm_kps[2*j+1] = kp_y;
			}
			else {
				hm_kps[2*j] = (kp_x + hp_x)/2.0;  	//取平均值
				hm_kps[2*j+1] = (kp_y + hp_y)/2.0;				
			}

		}
		objbox.hps = hm_kps;

		objs_tmp.push_back(objbox);
	}

	nms(objs_tmp, objs, nmsThresh);

	for (int k = 0; k < objs.size(); k++) {
		objs[k].bbox[0] *= d_scale_w*scale_w;
		objs[k].bbox[1] *= d_scale_h*scale_h;
		objs[k].bbox[2] *= d_scale_w*scale_w;
		objs[k].bbox[3] *= d_scale_h*scale_h;

	}

	// 以下为对关节点的解码
	int hp_h = hm_hp.h;
	int hp_w = hm_hp.w;
	int hp_c = hm_hp.c;
	int spacial_size = fea_w*fea_h;  //用来确定指针跳过每个通道的元素数量

	float *heatmap_ = (float*)(heatmap.data);   // 指针指向矩阵头部


	float *wh_w = (float*)(wh.data); // wh_w 第一个通道为包围框的宽
	float *wh_h = wh_w + spacial_size;  // wh_h 为wh的第二个通道，是包围框的高

	float *reg_x = (float*)(reg.data);  // 中心点x的偏移量
	float *reg_y = reg_x + spacial_size; // 中心点y的偏移量
}

