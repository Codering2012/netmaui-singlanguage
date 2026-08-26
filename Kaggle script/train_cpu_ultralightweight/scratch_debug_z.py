import numpy as np
from asl_geometric import DynamicGestureTracker, get_accurate_asl_template

tracker_z = DynamicGestureTracker()
for t in range(24):
    lm = get_accurate_asl_template('D').copy()
    if t < 8:
        lm[8, 0] += t * 0.006
    elif t < 16:
        lm[8, 0] += 0.048 - (t - 8) * 0.007
        lm[8, 1] += (t - 8) * 0.005
    else:
        lm[8, 0] += -0.008 + (t - 16) * 0.006
    tracker_z.update(lm)

pts = np.array(tracker_z.index_tip_history)
total_path = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
n = len(pts)
seg1 = pts[n//3] - pts[0]
seg2 = pts[2*n//3] - pts[n//3]
seg3 = pts[-1] - pts[2*n//3]

print("total_path:", total_path)
print("seg1:", seg1)
print("seg2:", seg2)
print("seg3:", seg3)
print("tracker_z.detect_z_gesture():", tracker_z.detect_z_gesture())
