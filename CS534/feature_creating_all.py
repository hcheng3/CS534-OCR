import feature
import os
import numpy as np
import imageprocessing


feature_data1 = []
feature_data2 = []
feature_data3 = []
feature_data4 = []
feature_data5 = []
feature_data6 = []
feature_data7 = []
# dir = 'trainningdata/'
dir = 'train/'
# dir = 'testingdata/'
files = os.listdir(dir)

i = 0
for file in files:
    if not file.startswith('.'):
        print file
        features = feature.feature_extraction(dir+file)
        aim_feature1 = np.append(features.combine_all()[0].ravel(), [int(file[0])])
        aim_feature2 = np.append(features.combine_all()[1].ravel(), [int(file[0])])
        aim_feature3 = np.append(features.combine_all()[2].ravel(), [int(file[0])])
        aim_feature4 = np.append(features.combine_all()[3].ravel(), [int(file[0])])
        aim_feature5 = np.append(features.combine_all()[4].ravel(), [int(file[0])])
        aim_feature6 = np.append(features.combine_all()[5].ravel(), [int(file[0])])
        aim_feature7 = np.append(features.combine_all()[6].ravel(), [int(file[0])])

        feature_data1.append(aim_feature1)
        feature_data2.append(aim_feature2)
        feature_data3.append(aim_feature3)
        feature_data4.append(aim_feature4)
        feature_data5.append(aim_feature5)
        feature_data6.append(aim_feature6)
        feature_data7.append(aim_feature7)
        print "%s file done" % i
        i += 1

feature_data1 = np.array(feature_data1)
feature_data2 = np.array(feature_data2)
feature_data3 = np.array(feature_data3)
feature_data4 = np.array(feature_data4)
feature_data5 = np.array(feature_data5)
feature_data6 = np.array(feature_data6)
feature_data7 = np.array(feature_data7)

np.savetxt("features_1/features_all.csv", feature_data1, fmt='%f', delimiter=",")
np.savetxt("features_1/features_profiling_density.csv", feature_data2, fmt='%f', delimiter=",")
np.savetxt("features_1/features_profiling_direction.csv", feature_data3, fmt='%f', delimiter=",")
np.savetxt("features_1/features_density_direction.csv", feature_data4, fmt='%f', delimiter=",")
np.savetxt("features_1/features_profiling.csv", feature_data5, fmt='%f', delimiter=",")
np.savetxt("features_1/features_density.csv", feature_data6, fmt='%f', delimiter=",")
np.savetxt("features_1/features_direction.csv", feature_data7, fmt='%f', delimiter=",")