from mmcv.ops import ball_query
import torch

points_xyz = torch.randn(1,10,3).cuda()
center_xyz = points_xyz[:,[3,9],:]
# import pdb;pdb.set_trace()
idx = ball_query(0., 1.3, 20, points_xyz, center_xyz)
print(points_xyz)
print(center_xyz)
print(idx)