#
#   itetate trump faces and calculate mean and average
#   for the data loader
#
import cv2
import os
from statistics import mean

b_means, g_means, r_means = [], [], []
b_stds,  g_stds,  r_stds  = [], [], []

for f in os.listdir("data/trump_faces"):
    if f.endswith(".jpg"):
        fname = os.path.join("data/trump_faces/", f)
        img   = cv2.imread(fname)
        m, s  = cv2.meanStdDev(img)
        m = m.flatten()
        s = s.flatten()
        #
        b_means.append(m[0])
        g_means.append(m[1])
        r_means.append(m[2])
        #
        b_stds.append(s[0])
        g_stds.append(s[1])
        r_stds.append(s[2])
        #

#
# (v - min) / (max - min)

b_mu = (mean(b_means) - 0) / (255 - 0)
g_mu = (mean(g_means) - 0) / (255 - 0)
r_mu = (mean(r_means) - 0) / (255 - 0)

b_st = (mean(b_stds) - 0) / (255 - 0)
g_st = (mean(g_stds) - 0) / (255 - 0)
r_st = (mean(r_stds) - 0) / (255 - 0)

print("RGB means {}, {}, {}".format(r_mu, g_mu, b_mu))
print("RGB stds  {}, {}, {}".format(r_st, g_st, b_st))
