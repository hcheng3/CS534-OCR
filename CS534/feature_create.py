import feature
import os
import numpy as np
import imageprocessing


feature_data = []
dir = 'trainningdata/'
# dir = 'train/'
# dir = 'testingdata/'
files = os.listdir(dir)

i = 0
for file in files:
    if not file.startswith('.'):
        print file
        features = feature.feature_extraction(dir+file)
        aim_feature = np.append(features.combine_all().ravel(), [int(file[0])])
        feature_data.append(aim_feature)
        print "%s file done" % i
        i += 1

feature_data = np.array(feature_data)

np.savetxt("features/features_all.csv", feature_data, fmt='%f', delimiter=",")




