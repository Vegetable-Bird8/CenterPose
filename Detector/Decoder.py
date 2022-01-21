from utils.utils_Decode import _nms, _topk,_topk_channel,_transpose_and_gather_feat
from utils.image import transform_preds
import torch
import numpy as np

def multi_pose_decode(heat, wh, kps, reg, hm_hp, hp_offset, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)  # 获得了前K个scores对应的值和索引 类别 以及 x y 

    kps = _transpose_and_gather_feat(kps, inds)
    # 经过索引后，输出的kps为对应x，y点上的34个偏移量
    kps = kps.view(batch, K, num_joints * 2)  # b×K ×（2j）
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    # reg
    # reg与hp用的是同一套索引
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    # wh与hp用的是同一套索引
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)   # b*k*4
    # hm_hp
    hm_hp = _nms(hm_hp)
    thresh = 0.1
    kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous() # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    # hp_offset
    hp_offset = _transpose_and_gather_feat(
        hp_offset, hm_inds.view(batch, -1))  # 获取关节点偏移量对应的索引 
    hp_offset = hp_offset.view(batch, num_joints, K, 2)
    hm_xs = hm_xs + hp_offset[:, :, :, 0]
    hm_ys = hm_ys + hp_offset[:, :, :, 1]


    mask = (hm_score > thresh).float()  # 筛选所有大于阈值的值，大于的地方为1 小于的地方为0
    hm_score = (1 - mask) * -1 + mask * hm_score  # 将所有大于阈值的地方保留原来的值，小于阈值的地方变为-1
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys   # 将hm_x 和hm_y大于阈值的地方保留原来的值，小于阈值的地方变为-10000
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs

    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        2).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
    # 计算两个坐标之间的L2距离
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    # 得到两套坐标距离之间的最小值和对应的索引
    min_dist, min_ind = dist.min(dim=3) # b x J x K
    # 找对最小坐标对应的hm_hp scores
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1

    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)
    l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)  # b*j*k*1
    t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    # 获得边界框的上下左右坐标 
    # 并判断在框内的坐标点
    mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
            (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
            (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
    mask = (mask > 0).float().expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
    return detections
# def multi_pose_decode(hm, wh, hps, reg, hm_hp, hp_offset, K=100):

#     batch, cat, height, width = hm.size()
#     num_joints = hps.shape[1] // 2
#     # hm = torch.sigmoid(hm)
#     # perform nms on hmmaps
#     hm = _nms(hm)
#     scores, inds, clses, ys, xs = _topk(hm, K=K)  # clses 代表最大值来自哪个通道 即哪个类

#     hps = _transpose_and_gather_feat(hps, inds)
#     hps = hps.view(batch, K, num_joints * 2)
#     # hps =[bs,k,34]
#     hps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
#     hps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
#     # hps[...,::2] += xs.unqueeze(-1).repeat(1,1,17)
#     if reg is not None:
#         reg = _transpose_and_gather_feat(reg, inds)
#         reg = reg.view(batch, K, 2)
#         xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
#         ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
#     else:
#         xs = xs.view(batch, K, 1) + 0.5
#         ys = ys.view(batch, K, 1) + 0.5
#     wh = _transpose_and_gather_feat(wh, inds)
#     wh = wh.view(batch, K, 2)
#     clses = clses.view(batch, K, 1).float()
#     scores = scores.view(batch, K, 1)

#     bboxes = torch.cat([xs - wh[..., 0:1] / 2,
#                         ys - wh[..., 1:2] / 2,
#                         xs + wh[..., 0:1] / 2,
#                         ys + wh[..., 1:2] / 2], dim=2)
#     if hm_hp is not None:
#         hm_hp = _nms(hm_hp)
#         thresh = 0.1
#         hps = hps.view(batch, K, num_joints, 2).permute(
#             0, 2, 1, 3).contiguous()  # b x J x K x 2
#         reg_hps = hps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
#         hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
#         if hp_offset is not None:
#             hp_offset = _transpose_and_gather_feat(
#                 hp_offset, hm_inds.view(batch, -1))
#             hp_offset = hp_offset.view(batch, num_joints, K, 2)
#             hm_xs = hm_xs + hp_offset[:, :, :, 0]
#             hm_ys = hm_ys + hp_offset[:, :, :, 1]
#         else:
#             hm_xs = hm_xs + 0.5
#             hm_ys = hm_ys + 0.5

#         mask = (hm_score > thresh).float()
#         hm_score = (1 - mask) * -1 + mask * hm_score
#         # 1-mask将满足条件的部分都变为0 不满足条件的变为1 乘以-10000 该位置的值变为-10000
#         hm_ys = (1 - mask) * (-10000) + mask * hm_ys
#         hm_xs = (1 - mask) * (-10000) + mask * hm_xs
#         hm_hps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
#             2).expand(batch, num_joints, K, K, 2)
#         # 计算由人体中心点+偏移计算得到的关键点坐标 与 直接通过关键点回归出来的坐标之间的距离
#         dist = (((reg_hps - hm_hps) ** 2).sum(dim=4) ** 0.5)
#         min_dist, min_ind = dist.min(dim=3)  # b x num_Joints x K
#         hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
#         min_dist = min_dist.unsqueeze(-1)
#         min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
#             batch, num_joints, K, 1, 2)
#         hm_hps = hm_hps.gather(3, min_ind)
#         hm_hps = hm_hps.view(batch, num_joints, K, 2)
#         l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#         t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#         r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#         b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#         mask = (hm_hps[..., 0:1] < l) + (hm_hps[..., 0:1] > r) + \
#                (hm_hps[..., 1:2] < t) + (hm_hps[..., 1:2] > b) + \
#                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
#         mask = (mask > 0).float().expand(batch, num_joints, K, 2)
#         hps = (1 - mask) * hm_hps + mask * hps
#         hps = hps.permute(0, 2, 1, 3).contiguous().view(
#             batch, K, num_joints * 2)
#     detections = torch.cat([bboxes, scores, hps, clses], dim=2)
#     """    
#     detections shape [batch,K,40]
#     detections[0:4]  为 bboxes
#     detections[4]    为 scores
#     detections[5:39] 为 hps 单数为x 双数为y
#     detections[39]   为 clses 类别
#     """

#     return detections
def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

if __name__ == '__main__':
    import numpy as np
    def numpy2tensor(ndarray):
        return torch.from_numpy(ndarray).float()
    count1 = np.random.randint(1, 100, 128 * 128)
    count2 = np.random.randint(1, 100, 128 * 128 * 2)
    count3 = np.random.randint(1, 100, 128 * 128 * 17)
    count4 = np.random.randint(1, 100, 128 * 128 * 34)
    hm = numpy2tensor(count1).reshape([1,1,128,128])
    reg = numpy2tensor(count2).reshape([1,2,128,128])
    wh = numpy2tensor(count2).reshape([1, 2, 128, 128])
    hps = numpy2tensor(count4).reshape([1,34,128,128])
    hp_hm = numpy2tensor(count3).reshape([1,17,128,128])
    hp_offset = numpy2tensor(count2).reshape([1,2,128,128])

    y = multi_pose_decode(hm, wh, hps, reg, hp_hm, hp_offset, K=100)
    print(type(y),len(y))
    print("+++++++++++++++++++++++++++++++++")
    for x in y:
        print(x,type(x),x.shape)