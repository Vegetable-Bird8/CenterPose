#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include"net.h"
void get_dir(const float* const src_point, \
        const float rot_rad,
        float* src_res) {
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);
    src_res[0] = src_point[0] * cs - src_point[1] * sn;
    src_res[1] = src_point[1] * sn + src_point[1] * cs;
}

void get_3rd_point(const cv::Point2f& a,\
        const cv::Point2f& b, cv::Point2f& out) {
    out.x = b.x + b.y - a.y;
    out.y = b.y + a.x - b.x;
}

void get_affine_transform(
        float* mat_data, 
        const float* const center, \
        const float* const scale, \
        const float* const shift, 
        const float rot, \
        const int output_h, const int output_w, 
        const bool inv) {
    float rot_rad = rot * PI / 180.;
    float src_p[2] = {0, scale[0] * -0.5};
    float dst_p[2] = {0, output_w * -0.5};
    float src_dir[2], dst_dir[2];
    get_dir(src_p, rot_rad, src_dir);
    get_dir(dst_p, rot_rad, dst_dir);

    cv::Point2f src[3], dst[3];
    src[0] = cv::Point2f(center[0] + scale[0] * shift[0],
            center[1] + scale[1] * shift[1]);
    src[1] = cv::Point2f(center[0] + src_dir[0] + scale[0] * shift[0], center[1] + src_dir[1] + scale[1] * shift[1]);
    dst[0] = cv::Point2f(output_w * 0.5, output_h * 0.5);
    dst[1] = cv::Point2f(output_w * 0.5 + dst_dir[0], output_h*0.5 + dst_dir[1]);
    get_3rd_point(dst[0], dst[1], dst[2]);
    get_3rd_point(src[0], src[1], src[2]);
    cv::Mat warp_mat;
    if (inv) {
        warp_mat = getAffineTransform(dst, src);
    } else {
        warp_mat = getAffineTransform(src, dst);
    }
    //return warp_mat;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            mat_data[i * 3 + j] = warp_mat.at<double>(i, j);
        }
    }
}

//   def pre_process(self, image, scale, meta=None):   # 图像预处理
//     height, width = image.shape[0:2]     # 读取高和宽
//     new_height = int(height * scale)      # scale 为缩放系数
//     new_width  = int(width * scale)
//     # # fix_res
//     # inp_height, inp_width = self.opt.input_h, self.opt.input_w
//     # c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
//     # s = max(height, width) * 1.0
//     # keep res
    
//     # 这部分代码作用就是通过按位或运算，找到最接近的2的倍数-1作为最终的尺度。
//     inp_height = (new_height | self.opt.pad) + 1  # 等同于 inp_height = new_height+pad +1 if new_height>pad else pad+1
//     inp_width = (new_width | self.opt.pad) + 1
//     c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
//     s = np.array([inp_width, inp_height], dtype=np.float32)

//     trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])   # 根据输入图片获得仿射矩阵
//     resized_image = cv2.resize(image, (new_width, new_height))     # 改变图像大小
//     inp_image = cv2.warpAffine(
//       resized_image, trans_input, (inp_width, inp_height),    # 仿射变换
//       flags=cv2.INTER_LINEAR)
//     inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)  # 归一化

//     images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 通道数变在前 并变为[1,3,h,w]

//     images = torch.from_numpy(images)  # 转为tensor
//     meta = {'c': c, 's': s,
//             'out_height': inp_height // self.opt.down_ratio,
//             'out_width': inp_width // self.opt.down_ratio}
//     return images, meta
int pre_process(ncnn::Mat image,float scale){
    int height = image.h;
    int width = image.w;
    int new_height = int(height*scale);
    int new_width = int(width*scale);
}