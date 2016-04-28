#-------------------------------------------------------------------------------
# Name:        feature extraction
# Purpose:
#
# Author:      Hongzhang
#
# Created:     19/04/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import copy
import numpy as np
import imageprocessing
from skimage import feature, io
from skimage.morphology import skeletonize
from skimage.transform import resize


class feature_extraction(object):

    def __init__(self, imagestr):
        self.p = imageprocessing.readImg(imagestr).imageMatrixWhole
        image_ratio = self.preprocessing()
        self.image = image_ratio[0]
        self.ratio = image_ratio[1]
        # print self.image.shape
        # print self.ratio

    def profiling(self):
        image = resize(self.image, (257, 257))
        edges = feature.canny(image)
        edges = skeletonize(edges)
        edges = edges.astype(int)
        edges = edges[1:256, 1:256]

        lt = self.feature_extraction_profiling(edges, 2, 5)
        profiling = lt.ravel()
        return np.append(profiling, [float(self.ratio)/10])

    def direction_skeleton(self):
        edges = feature.canny(self.image)
        edges = skeletonize(edges)
        edges = edges.astype(int)
        edges = edges[1:251, 1:251]
        zones = self.zoning(edges)
        for i in zones:
            self.line_segmentation(i)
        direction = self.direction_transfer(self.count_common(zones))

        return np.append(direction, [float(self.ratio)/10])

    def direction(self):
        edges = skeletonize(self.image)
        edges = edges.astype(int)
        edges = edges[1:251, 1:251]
        zones = self.zoning(edges)
        for i in zones:
            self.line_segmentation(i)
        direction = self.direction_transfer(self.count_common(zones))

        return np.append(direction, [float(self.ratio)/10])
        # return self.count_common(zones)

    def density(self):
        density = []
        zones = self.zoning(self.image)
        for k in zones:
            count = 0
            for i in range(k.shape[0]):
                for j in range(k.shape[1]):
                    if k[i][j] != 0:
                        count += 1
            density.append(float(count)/(k .shape[0]*k.shape[1]))
        return np.append(density, [float(self.ratio)/10])

    def combine_p_ds(self):
        return np.concatenate((self.profiling(), np.array(self.direction_skeleton())))

    def combine_p_d(self):
        return np.concatenate((self.profiling(), np.array(self.direction())))

    def combine_p_density(self):
        return np.concatenate((self.profiling(), np.array(self.density())))

    def combine_all(self):
        a = self.profiling()
        b = self.density()
        c = self.direction()
        return np.concatenate((a, np.array(b), np.array(c))),np.concatenate((a, np.array(b))),np.concatenate((a, np.array(c))),np.concatenate((b, np.array(c))),a,b,c

    def combine_alls(self):
        return np.concatenate((self.profiling(), np.array(self.density()), np.array(self.direction_skeleton())))

    def combine_dd(self):
        return np.concatenate((np.array(self.density()), np.array(self.direction())))

    def count_common(self,zones):
        direction_feature = []
        for i in zones:
            num_2 = list(i.ravel()).count(2)
            num_3 = list(i.ravel()).count(3)
            num_4 = list(i.ravel()).count(4)
            num_5 = list(i.ravel()).count(5)
            list_a = [(num_2,2), (num_3,3), (num_4,4), (num_5,5)]
            list_a.sort()
            if list_a[3][0] != 0:
                direction_feature.append(list_a[3][1])
            else:
                direction_feature.append(0)
            if list_a[2][0] != 0:
                direction_feature.append(list_a[2][1])
            else:
                direction_feature.append(0)
        return direction_feature

    def direction_transfer(self, direction_feature):
        direction = np.zeros(len(direction_feature)*4)
        for i in range(len(direction_feature)):
            if direction_feature[i] == 2:
                direction[4*i] = 1
            if direction_feature[i] == 3:
                direction[4 * i + 1] = 1
            if direction_feature[i] == 4:
                direction[4 * i + 2] = 1
            if direction_feature[i] == 5:
                direction[4 * i + 3] = 1
        return direction

    def preprocessing(self):
        p = resize(self.p, (256, 256))
        image = np.zeros(shape=(258, 258))
        image[1:257, 1:257] = p
        image_ratio = self.universe(image)
        image = image_ratio[0]
        ratio = image_ratio[1]
        image = resize(image, (252, 252))
        return image, ratio

    def universe(self, image):
        up = 0
        bottom = 0
        right = 0
        left = 0

        for i in range(256):
            for j in range(256):
                if image[i][j] != 0:
                    bottom = i
                    break
        i = 0
        j = 0
        for i in reversed(range(256)):
            for j in range(256):
                if image[i][j] != 0:
                    up = i
                    break
        i = 0
        j = 0
        for j in range(256):
            for i in range(256):
                if image[i][j] != 0:
                    right = j
                    break
        i = 0
        j = 0
        for j in reversed(range(256)):
            for i in range(256):
                if image[i][j] != 0:
                    left = j
                    break
        up = up-1
        bottom = bottom+2
        left = left-1
        right = right +2
        crop_image = image[up: bottom, left:right]
        if bottom - up >= right - left:
            a = bottom - up
            result_image = np.zeros(shape=(a,a))
            result_image[0: result_image.shape[0], (a - crop_image.shape[1])/2: (a - crop_image.shape[1])/2+crop_image.shape[1]] = crop_image
        if right - left > bottom - up:
            b = right - left
            result_image = np.zeros(shape=(b, b))
            result_image[(b - crop_image.shape[0])/2: (b - crop_image.shape[0])/2+crop_image.shape[0], 0: result_image.shape[0]] = crop_image
        ratio = float(bottom - up)/(right - left)
        return result_image, ratio

    def zoning(self, image):
        zone_parts = []

        a = image.shape[0] / 3
        b = image.shape[1] / 3

        zone_parts.append(image[0:a, 0:b])
        zone_parts.append(image[0:a, b:2 * b])
        zone_parts.append(image[0:a, 2 * b:image.shape[1]])
        zone_parts.append(image[a:2 * a, 0:b])
        zone_parts.append(image[a:2 * a, b:2 * b])
        zone_parts.append(image[a:2 * a, 2 * b:image.shape[1]])
        zone_parts.append(image[2 * a:image.shape[0], 0:b])
        zone_parts.append(image[2 * a:image.shape[0], b:2 * b])
        zone_parts.append(image[2 * a:image.shape[0], 2 * b:image.shape[1]])
        return zone_parts

    def check_state(self, image, x, y):
        if image[y][x] == 0:
            return None
        adajacent = [((x - 1, y - 1), 1), ((x, y - 1), 2), ((x + 1, y - 1), 3), ((x - 1, y), 4), ((x + 1, y), 6),
                     ((x - 1, y + 1), 7), ((x, y + 1), 8), ((x + 1, y + 1), 9)]
        neighbor_count = 0
        neigbbor_label = []
        for i in range(len(adajacent)):
            if 0 <= adajacent[i][0][0] < image.shape[1] and 0 <= adajacent[i][0][1] < image.shape[0]:
                if image[adajacent[i][0][1]][adajacent[i][0][0]] != 0:
                    neighbor_count += 1
                    neigbbor_label.append(adajacent[i][1])

        if neighbor_count == 1:
            return 'stater'

        elif neighbor_count == 2:
            return 'normal'

        elif neighbor_count == 3:
            if neigbbor_label == [1, 9, 7] or neigbbor_label == [1, 9, 3] or neigbbor_label == [3, 7, 1] or neigbbor_label == [3, 7, 9] or neigbbor_label == [4, 6, 2] or neigbbor_label == [4, 6, 8] or neigbbor_label == [2, 8, 4] or neigbbor_label == [2, 8, 6]:
                return 'intersection'

        elif neighbor_count == 4:
            for i in neigbbor_label:
                if i == 1:
                    if 2 not in neigbbor_label and 4 not in neigbbor_label:
                        return 'intersection'
                if i == 2:
                    if 1 not in neigbbor_label and 3 not in neigbbor_label:
                        return 'intersection'
                if i == 3:
                    if 2 not in neigbbor_label and 6 not in neigbbor_label:
                        return 'intersection'
                if i == 4:
                    if 1 not in neigbbor_label and 7 not in neigbbor_label:
                        return 'intersection'
                if i == 6:
                    if 3 not in neigbbor_label and 9 not in neigbbor_label:
                        return 'intersection'
                if i == 7:
                    if 4 not in neigbbor_label and 8 not in neigbbor_label:
                        return 'intersection'
                if i == 8:
                    if 7 not in neigbbor_label and 9 not in neigbbor_label:
                        return 'intersection'
                if i == 9:
                    if 6 not in neigbbor_label and 8 not in neigbbor_label:
                        return 'intersection'
        elif neighbor_count >= 5:
            return 'intersection'

    def traverse(self, image, intersection, x, y, minor, direction):
        # print "x,y,direction:", x,y,direction
        adajacent = [((x - 1, y - 1), 5), ((x, y - 1), 2), ((x + 1, y - 1), 3), ((x - 1, y), 4), ((x + 1, y), 4),
                     ((x - 1, y + 1), 3), ((x, y + 1), 2), ((x + 1, y + 1), 5)]
        neighbor_count = 0
        neighbors = []
        image[y][x] = direction
        # print image[y][x]
        count = 0
        new_x = -1
        new_y = -1

        for i in range(len(adajacent)):
            if 0 <= adajacent[i][0][0] < image.shape[1] and 0 <= adajacent[i][0][1] < image.shape[0]:
                if image[adajacent[i][0][1]][adajacent[i][0][0]] != 0:
                    neighbor_count += 1
                    neighbors.append(adajacent[i])

        if (x, y) in intersection:
            for j in neighbors:
                if image[j[0][1]][j[0][0]] == 1:
                    minor.append((j[0][0], j[0][1]))
                    return
        # print neigbbors
        for x in neighbors:
            if x[1] != direction and image[x[0][1]][x[0][0]] == 1:
                minor.append((x[0][0], x[0][1]))
            if x[1] == direction:
                count += 1

                if image[x[0][1]][x[0][0]] == 1:
                    image[x[0][1]][x[0][0]] = direction
                    new_x = x[0][0]
                    new_y = x[0][1]
        if count == 1:
            return
        if new_x == -1 or new_y == -1:
            # print x
            return
        self.traverse(image, intersection, new_x, new_y, minor, direction)

        if neighbor_count == 1:
            image[adajacent[i][0][1]][adajacent[i][0][0]] == direction
            return

    def find_line(self, image, staters, minor_staters, intersection):
        # print staters

        for a in staters:

            x = a[0]
            y = a[1]

            # print (x,y)

            next_x = -1
            next_y = -1
            adajacent = [((x - 1, y - 1), 5), ((x, y - 1), 2), ((x + 1, y - 1), 3), ((x - 1, y), 4), ((x + 1, y), 4),
                         ((x - 1, y + 1), 3), ((x, y + 1), 2), ((x + 1, y + 1), 5)]
            for i in range(len(adajacent)):
                if 0 <= adajacent[i][0][0] < image.shape[1] and 0 <= adajacent[i][0][1] < image.shape[0]:
                    if image[adajacent[i][0][1]][adajacent[i][0][0]] > 1:
                        back = adajacent[i][1]
                    if image[adajacent[i][0][1]][adajacent[i][0][0]] == 1:
                        next_x = adajacent[i][0][0]
                        next_y = adajacent[i][0][1]
                        direction = adajacent[i][1]
                        break
            if next_x == -1 or next_y == -1:
                image[y][x] = back

                continue
            # print next_x, next_y,direction
            image[y][x] = direction
            self.traverse(image, intersection, next_x, next_y, minor_staters, direction)

    def line_segmentation(self, image):
        staters = []
        intersection = []
        minor_staters = []

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_type = self.check_state(image, j, i)

                if pixel_type == 'stater':
                    staters.append((j, i))

                if pixel_type == 'intersection':
                    intersection.append((j, i))

        self.find_line(image, staters, minor_staters, intersection)
        minor_staters = set(minor_staters)
        minor_staters = list(minor_staters)

        while len(minor_staters):
            staters = copy.deepcopy(minor_staters)
            del minor_staters[:]

            self.find_line(image, staters, minor_staters, intersection)
            minor_staters = set(minor_staters)
            minor_staters = list(minor_staters)

    def feature_extraction_profiling(self, image, max, samples):

        lt_lr = np.zeros(shape=(image.shape[0], max))

        left_to_right = np.zeros(shape=(image.shape[0] / samples, max))

        lt_rl = np.zeros(shape=(image.shape[0], max))

        right_to_left = np.zeros(shape=(image.shape[0] / samples, max))

        lt_td = np.zeros(shape=(image.shape[1], max))

        top_to_down = np.zeros(shape=(image.shape[1] / samples, max))

        lt_dt = np.zeros(shape=(image.shape[1], max))

        down_to_top = np.zeros(shape=(image.shape[1] / samples, max))

        for i in range(image.shape[0]):
            n = 0
            for j in range(image.shape[1]):
                if j - 1 >= 0:
                    if image[i][j] == 1 and image[i][j - 1] != 1:
                        if n < max:
                            lt_lr[i][n] = float(j) / image.shape[1]
                            n += 1
        a = 0
        b = 0
        while a + samples <= 250:
            left_to_right[b] = np.mean(lt_lr[a:a + samples], axis=0)
            a += samples
            b += 1

        i = 0
        j = 0
        for i in range(image.shape[0]):
            n = 0
            for j in reversed(range(image.shape[1])):
                if j + 1 < image.shape[1]:
                    if image[i][j] == 1 and image[i][j + 1] != 1:
                        if n < max:
                            lt_rl[i][n] = float(j) / image.shape[1]
                            n += 1
        a = 0
        b = 0
        while a + samples <= 250:
            right_to_left[b] = np.mean(lt_rl[a:a + samples], axis=0)
            a += samples
            b += 1

        i = 0
        j = 0
        for i in range(image.shape[1]):
            n = 0
            for j in range(image.shape[0]):
                if j - 1 >= 0:
                    if image[j][i] == 1 and image[j - 1][i] != 1:
                        if n < max:
                            lt_td[i][n] = float(j) / image.shape[0]
                            n += 1
        a = 0
        b = 0
        while a + samples <= 150:
            top_to_down[b] = np.mean(lt_td[a:a + samples], axis=0)
            a += samples
            b += 1

        i = 0
        j = 0
        for i in range(image.shape[1]):
            n = 0
            for j in reversed(range(image.shape[0])):
                if j + 1 < image.shape[0]:
                    if image[j][i] == 1 and image[j + 1][i] != 1:
                        if n < max:
                            lt_dt[i][n] = float(j) / image.shape[0]
                            n += 1
        a = 0
        b = 0
        while a + samples <= 150:
            down_to_top[b] = np.mean(lt_dt[a:a + samples], axis=0)
            a += samples
            b += 1

        # return lt_td
        return np.concatenate((left_to_right, right_to_left, top_to_down, down_to_top))




# asd = feature_extraction("trainningdata/0-6.bmp")
# # asds= asd.zoning(asd.image)
#
#
# io.imshow(asd.image)
# io.show()
#
#

# print asd.density()
#  np.savetxt('33',asd.direction(),fmt='%d')
#  print type(asd.profiling())
#  print type(asd.direction_skeleton())
# np.savetxt('profiling',asd.profiling(),fmt='%f')
#  np.savetxt('direction_skeleton',asd.direction_skeleton(),fmt='%f')
# np.savetxt('direction', asd.direction(), fmt='%f')
# np.savetxt('density',asd.density(),fmt='%f')
#  np.savetxt('combine_pd',asd.combine_p_d(),fmt='%f')
#  np.savetxt('combine_pden',asd.combine_p_density(),fmt='%f')
#  np.savetxt('combine_pds',asd.combine_p_ds(),fmt='%f')
# np.savetxt('direction',asd.direction(),fmt='%f')
# np.savetxt('combine_density_direction',asd.combine_dd(),fmt='%f')
# np.savetxt('combine_all',asd.combine_all(),fmt='%f')


# np.savetxt('profiling',asd.image,fmt='%f')
