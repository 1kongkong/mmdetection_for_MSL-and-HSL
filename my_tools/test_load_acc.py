import numpy as np
import time
from mmengine.fileio import get
import pdb

pts_filename = "data/SensatUrban/sensaturban_data/birmingham_block_3_point.npy"

t1 = time.time()
data = np.load(pts_filename)
t2 = time.time()
print(t2 - t1)

pts_filename = "data/SensatUrban/points/birmingham_block_3.bin"

t1 = time.time()
pts_bytes = get(pts_filename)
points = np.frombuffer(pts_bytes, dtype=np.float32)
t2 = time.time()
print(t2 - t1)
pdb.set_trace()
