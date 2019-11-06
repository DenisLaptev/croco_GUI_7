import cv2
import csv
import random
import math
import sys
from copy import deepcopy
import numpy as np


def read_contours(csv_filename):
    contours = list()

    with open(csv_filename) as csv_file:
        fs = csv.reader(csv_file, delimiter=',')
        for textline in fs:
            contour = np.zeros((0, 2), dtype='int32')
            if textline == '':
                break
            for i in range(0, len(textline) - 1, 2):
                x = int(textline[i])
                y = int(textline[i + 1])
                contour = np.append(contour, [[x, y]], axis=0)
            contours.append(contour)

    return contours


# ----------------------------------------------------------------------------------------------------------------------


INT_MAX = 2147483647


class Cdata:
    ratio = 0
    direction = 0
    status = False
    center = (1, 2)
    pts = list()
    vn = list()
    h = 0


class Triplet:
    v1 = 0
    v2 = 0
    v3 = 0

    alpha1 = 0
    alpha2 = 0
    alpha3 = 0

    beta1 = 0
    beta2 = 0
    beta3 = 0

    ratio12 = 0
    ratio21 = 0
    ratio23 = 0
    ratio32 = 0
    ratio13 = 0
    ratio31 = 0

    s12 = 0
    s13 = 0
    s23 = 0

    ratio1 = 0
    ratio2 = 0
    ratio3 = 0

    orientation = 0


class Match:
    id1 = -1
    id2 = -1
    dist = -1

    def __init__(self, _id1=-1, _id2=-1, _dist=-1):
        self.id1 = _id1
        self.id2 = _id2
        self.dist = _dist


# ----------------------------------------------------------------------------------------------------------------------


def draw_matched_contours(img1, c1, img2, c2, t1, t2, matches):
    # ***size[0] is height
    # ***size[1] is width

    img1size = img1.shape
    img2size = img2.shape
    size = (max(img1size[0], img2size[0]), img1size[1] + img2size[1], img1size[2])  # height,width,channels

    out_img = np.zeros(size, img1.dtype)

    out_img[0:img1size[0], 0:img1size[1]] = img1
    out_img[0:img2size[0], img1size[1]:size[1]] = img2

    radius = 5

    for m in range(len(matches)):
        print(m)
        i1 = matches[m].id1
        i2 = matches[m].id2

        # if best_j = -1 (there was no matching with i-th triplet)
        if i1 < 0 or i2 < 0:
            continue

        # select color
        # loop over ellipses
        clmin = 192
        cldef = 255 - 192

        # determining random color
        clmd = random.randint(0, 6)  # color mode (6 modes available)
        clmul = float(1 / (1 + random.randint(0, 10)))
        intensity = clmin + int(clmul * cldef)
        if clmd == 0:
            color = (intensity, 0, 0)
        elif clmd == 1:
            color = (0, intensity, 0)
        elif clmd == 2:
            color = (0, 0, intensity)
        elif clmd == 3:
            color = (intensity, intensity, 0)
        elif clmd == 4:
            color = (intensity, 0, intensity)
        elif clmd == 5:
            color = (0, intensity, intensity)
        elif clmd == 6:
            color = (intensity, intensity, intensity)
        else:
            color = (0, 0, 0)

        out_img = cv2.drawContours(out_img, [c1[t1[i1].v1].pts], 0, color, -1)
        out_img = cv2.drawContours(out_img, [c1[t1[i1].v2].pts], 0, color, -1)
        out_img = cv2.drawContours(out_img, [c1[t1[i1].v3].pts], 0, color, -1)

        out_img = cv2.drawContours(out_img, [c2[t2[i2].v1].pts], 0, color, -1, maxLevel=INT_MAX,
                                   offset=(img1size[1], 0))
        out_img = cv2.drawContours(out_img, [c2[t2[i2].v2].pts], 0, color, -1, maxLevel=INT_MAX,
                                   offset=(img1size[1], 0))
        out_img = cv2.drawContours(out_img, [c2[t2[i2].v3].pts], 0, color, -1, maxLevel=INT_MAX,
                                   offset=(img1size[1], 0))

        out_img = cv2.circle(out_img, c1[t1[i1].v1].center, radius, color, 2, cv2.LINE_AA)
        out_img = cv2.circle(out_img, c1[t1[i1].v2].center, radius, color, 2, cv2.LINE_AA)
        out_img = cv2.circle(out_img, c1[t1[i1].v3].center, radius, color, 2, cv2.LINE_AA)

        out_img = cv2.circle(out_img, tuple(map(sum, zip(c2[t2[i2].v1].center, (img1size[1], 0)))), radius, color, 2,
                             cv2.LINE_AA)
        out_img = cv2.circle(out_img, tuple(map(sum, zip(c2[t2[i2].v2].center, (img1size[1], 0)))), radius, color, 2,
                             cv2.LINE_AA)
        out_img = cv2.circle(out_img, tuple(map(sum, zip(c2[t2[i2].v3].center, (img1size[1], 0)))), radius, color, 2,
                             cv2.LINE_AA)

        out_img = cv2.line(out_img, c1[t1[i1].v1].center, c1[t1[i1].v2].center, color, 3, cv2.LINE_AA)  # 1-2 in img1
        out_img = cv2.line(out_img, c1[t1[i1].v2].center, c1[t1[i1].v3].center, color, 3, cv2.LINE_AA)  # 2-3 in img1
        out_img = cv2.line(out_img, c1[t1[i1].v3].center, c1[t1[i1].v1].center, color, 3, cv2.LINE_AA)  # 3-1 in img1

        out_img = cv2.line(out_img, tuple(map(sum, zip(c2[t2[i2].v1].center, (img1size[1], 0)))),
                           tuple(map(sum, zip(c2[t2[i2].v2].center, (img1size[1], 0)))), color, 3,
                           cv2.LINE_AA)  # 1-2 in img2
        out_img = cv2.line(out_img, tuple(map(sum, zip(c2[t2[i2].v2].center, (img1size[1], 0)))),
                           tuple(map(sum, zip(c2[t2[i2].v3].center, (img1size[1], 0)))), color, 3,
                           cv2.LINE_AA)  # 2-3 in img2
        out_img = cv2.line(out_img, tuple(map(sum, zip(c2[t2[i2].v1].center, (img1size[1], 0)))),
                           tuple(map(sum, zip(c2[t2[i2].v3].center, (img1size[1], 0)))), color, 3,
                           cv2.LINE_AA)  # 1-3 in img2

        out_img = cv2.line(out_img, c1[t1[i1].v1].center, tuple(map(sum, zip(c2[t2[i2].v1].center, (img1size[1], 0)))),
                           color, 3, cv2.LINE_AA)  # 1(img1)-1(img2)
        out_img = cv2.line(out_img, c1[t1[i1].v2].center, tuple(map(sum, zip(c2[t2[i2].v2].center, (img1size[1], 0)))),
                           color, 3, cv2.LINE_AA)  # 2(img1)-2(img2)
        out_img = cv2.line(out_img, c1[t1[i1].v3].center, tuple(map(sum, zip(c2[t2[i2].v3].center, (img1size[1], 0)))),
                           color, 3, cv2.LINE_AA)  # 3(img1)-3(img2)

    return out_img


# ----------------------------------------------------------------------------------------------------------------------


def find_triplet_neighbours(c, ref_id, min_k, max_k):
    vn = list()

    cref = c[ref_id].pts#получили контур из объекта Cdata
    refcenter = c[ref_id].center
    low_bounds, high_bounds = np.zeros((0, 2), dtype='int32'), np.zeros((0, 2), dtype='int32')

    # prepare search area
    for k in range(len(cref)):
        low_bounds = np.append(low_bounds, [(cref[k] - refcenter) * min_k + refcenter], axis=0)
        high_bounds = np.append(high_bounds, [(cref[k] - refcenter) * max_k + refcenter], axis=0)
    low_bounds = low_bounds.astype(int)
    high_bounds = high_bounds.astype(int)

    for k in range(len(c)):
        if cv2.pointPolygonTest(high_bounds, c[k].center, True) > 0 > cv2.pointPolygonTest(low_bounds, c[k].center, True):
            vn.append(k)
    c[ref_id].vn = deepcopy(vn)
    return c


# ----------------------------------------------------------------------------------------------------------------------


def get_contours_data(contours):
    c = [Cdata()] * len(contours)

    for i in range(len(contours)):
        ci = Cdata()
        # minAreaRect Finds a rotated rectangle of the minimum area enclosing the input 2D point set
        # rr = ( center (x,y), (width, height), angle of rotation )
        rr = cv2.minAreaRect(contours[i])
        ratio = rr[1][0] / rr[1][1]

        if ratio < 1:
            ci.direction = math.fmod(2.0 * np.pi + (rr[2] * np.pi / 180.0), np.pi)
        else:
            ci.direction = math.fmod(2.5 * np.pi + (rr[2] * np.pi / 180.0), np.pi)
        ci.h = min(rr[1][1], rr[1][0])
        ci.ratio = max(ratio, 1 / ratio)
        ci.center = tuple([round(rr[0][0]), round(rr[0][1])])
        ci.pts = contours[i]
        if ci.ratio > 1.3:
            ci.status = True
        c[i] = deepcopy(ci)
    return c


# ----------------------------------------------------------------------------------------------------------------------


def init_triplet(c, p1, p2, p3):
    # sort vertices
    if p1 > p2:
        p1, p2 = p2, p1
    if p2 > p3:
        p2, p3 = p3, p2
    # if p1 > p2:
    #     p1, p2 = p2, p1

    if p1 > p3:
        p1, p3 = p3, p1

    temp = Triplet()

    # индексы (в списке c объектов Cdata) первого, второго и третьего контура в триплете
    temp.v1 = p1
    temp.v2 = p2
    temp.v3 = p3

    # get points
    # центры трёх контуров в триплете
    pt1 = c[p1].center
    pt2 = c[p2].center
    pt3 = c[p3].center

    # get edges
    # ptt = np.array(pt2) - np.array(pt1)
    l1 = cv2.norm(np.array(pt2) - np.array(pt1))  # расстояние между центрами контуров 1 и 2
    l2 = cv2.norm(np.array(pt3) - np.array(pt2))  # расстояние между центрами контуров 2 и 3
    l3 = cv2.norm(np.array(pt3) - np.array(pt1))  # расстояние между центрами контуров 1 и 3

    # get angles
    # квадраты расстояний между центрами контуров в триптете
    l1sq = l1 * l1
    l2sq = l2 * l2
    l3sq = l3 * l3

    temp.alpha1 = np.arccos((l1sq + l3sq - l2sq) / (2 * l1 * l3))  # myalpha2
    temp.alpha2 = np.arccos((l1sq + l2sq - l3sq) / (2 * l1 * l2))  # myalpha3
    temp.alpha3 = np.arccos((l2sq + l3sq - l1sq) / (2 * l2 * l3))  # myalpha1

    # temp.alpha1 = np.arccos((l2sq + l3sq - l1sq) / (2 * l2 * l3))
    # temp.alpha2 = np.arccos((l1sq + l3sq - l2sq) / (2 * l1 * l3))
    # temp.alpha3 = np.arccos((l1sq + l2sq - l3sq) / (2 * l1 * l2))

    return temp


# ----------------------------------------------------------------------------------------------------------------------

# Проверяем,чтобы углы треугольника были больше минимального угла (min_angle) и меньше максимального угла (max_angle).
def is_good_triplet(tr, min_angle, max_angle):
    min_alpha = min(tr.alpha1, min(tr.alpha2, tr.alpha3))
    max_alpha = max(tr.alpha1, max(tr.alpha2, tr.alpha3))

    # check angles
    if min_alpha < min_angle or max_alpha < max_angle:
        return False

    return True


# ----------------------------------------------------------------------------------------------------------------------


def orientation_triplet(pt1, pt2, pt3):
    # To find orientation of ordered triplet (pt1, pt2, pt3).
    # The function returns following values
    # 0 --> p, q and r are colinear
    # 1 --> Clockwise
    # 2 --> Counterclockwise

    val = (pt2[1] - pt1[1]) * (pt3[0] - pt2[0]) - (pt2[0] - pt1[0]) * (pt3[1] - pt2[1])

    # if val == 0:
    #     return 0  # colinear
    # return 1 if val > 0 else 2

    result = 0
    if val == 0:
        result = 0  # colinear
    elif val > 0:
        result = 1  # Clockwise
    else:
        result = 2  # Counterclockwise

    return result


# ----------------------------------------------------------------------------------------------------------------------


def get_triplet_features(tr, c):
    # sort vertices by angles
    if tr.alpha1 > tr.alpha2:
        tr.alpha1, tr.alpha2 = tr.alpha2, tr.alpha1
        tr.v1, tr.v2 = tr.v2, tr.v1
    if tr.alpha2 > tr.alpha3:
        tr.alpha2, tr.alpha3 = tr.alpha3, tr.alpha2
        tr.v2, tr.v3 = tr.v3, tr.v2
    if tr.alpha1 > tr.alpha2:
        tr.alpha1, tr.alpha2 = tr.alpha2, tr.alpha1
        tr.v1, tr.v2 = tr.v2, tr.v1

    # get points
    #centers of contours in triplet
    pt1 = c[tr.v1].center
    pt2 = c[tr.v2].center
    pt3 = c[tr.v3].center

    # sort vertices in clockwise order
    if orientation_triplet(pt1, pt2, pt3) == 2:
        pt2, pt3 = pt3, pt2
        tr.alpha2, tr.alpha3 = tr.alpha3, tr.alpha2
        tr.v2, tr.v3 = tr.v3, tr.v2

    # set triangle direction
    tr.orientation = math.fmod(2 * np.pi + np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]), 2 * np.pi)

    # set vertices direction in triangle coord-s
    tr.beta1 = math.fmod(2 * np.pi + c[tr.v1].direction - tr.orientation, 2 * np.pi)
    tr.beta2 = math.fmod(2 * np.pi + c[tr.v2].direction - tr.orientation, 2 * np.pi)
    tr.beta3 = math.fmod(2 * np.pi + c[tr.v3].direction - tr.orientation, 2 * np.pi)

    # set area aspect ratios
    s1 = cv2.contourArea(c[tr.v1].pts)
    s2 = cv2.contourArea(c[tr.v2].pts)
    s3 = cv2.contourArea(c[tr.v3].pts)

    tr.s12 = s1 / s2
    tr.s23 = s2 / s3
    tr.s13 = s1 / s3

    # set ellipses aspect ratio
    tr.ratio1 = c[tr.v1].ratio
    tr.ratio2 = c[tr.v2].ratio
    tr.ratio3 = c[tr.v3].ratio

    # set aspect ratio min(ell h, ell w)/ l1
    l1 = cv2.norm(np.array(pt2) - np.array(pt1))
    l2 = cv2.norm(np.array(pt3) - np.array(pt2))
    l3 = cv2.norm(np.array(pt3) - np.array(pt1))

    tr.ratio12 = c[tr.v1].h / l1  # ratio of cell size 1 to distance to 2 cell
    tr.ratio21 = c[tr.v2].h / l1  # ratio of cell size 2 to distance to 1 cell
    tr.ratio23 = c[tr.v2].h / l2  # ratio of cell size 2 to distance to 3 cell
    tr.ratio32 = c[tr.v3].h / l2  # ratio of cell size 3 to distance to 2 cell
    tr.ratio13 = c[tr.v1].h / l3  # ratio of cell size 1 to distance to 3 cell
    tr.ratio31 = c[tr.v3].h / l3  # ratio of cell size 3 to distance to 1 cell

    return tr


# ----------------------------------------------------------------------------------------------------------------------


def extract_triplets(c):
    triplets = list()

    if not c:
        return triplets

    # select good neighbours for triplets
    for i in range(len(c)):
        # check vertice status
        if not c[i].status:
            continue
        find_triplet_neighbours(c, i, 15.0, 20.0)

        # check number of detected neighbor vertices
        if not c[i].vn:
            c[i].status = False

    # select vertices for triplets
    for i in range(len(c)):
        print('Triplets extraction... ', i, 'of', len(c))
        p1 = i
        # check vertice status
        if not c[p1].status:
            continue

        #Пробегаемся по соседям контура p1
        for j in range(len(c[p1].vn)):
            p2 = c[p1].vn[j]
            # check vertice status
            if not c[p2].status:
                continue

            # Пробегаемся по соседям контура p2
            for k in range(len(c[p2].vn)):
                p3 = c[p2].vn[k]
                # check vertice status
                if not c[p3].status:
                    continue
                if p1 == p2 or p1 == p3 or p2 == p3:
                    continue

                # compute angles
                temp = init_triplet(c, p1, p2, p3)

                # validate triplet
                if not is_good_triplet(temp, 0.5, 1.57):
                    continue

                is_unique = True

                # check uniqueness of triplet
                for tripl in triplets:
                    if tripl.v1 == temp.v1 and tripl.v2 == temp.v2 and tripl.v3 == temp.v3:
                        is_unique = False
                        break

                if not is_unique:
                    continue

                triplets.append(temp)

    print(len(triplets))

    # extract features for triplets
    for i in range(len(triplets)):
        get_triplet_features(triplets[i], c)

    return triplets


# ----------------------------------------------------------------------------------------------------------------------


def match_triplets(t1, t2):
    score = 0

    # check triangle angles
    d1 = abs(t1.alpha1 - t2.alpha1) + abs(t1.alpha2 - t2.alpha2) + abs(t1.alpha3 - t2.alpha3)

    # check cells direction
    d2 = abs(t1.beta1 - t2.beta1) + abs(t1.beta2 - t2.beta2) + abs(t1.beta3 - t2.beta3)

    # check area ratio
    d3 = abs(t1.s12 - t2.s12) + abs(t1.s13 - t2.s13) + abs(t1.s23 - t2.s23)

    # check cells rect ratio
    d4 = abs(t1.ratio1 - t2.ratio1) + abs(t1.ratio2 - t2.ratio2) + abs(t1.ratio3 - t2.ratio3)

    # check ratio of rect height to triplet edges
    d5 = abs(t1.ratio12 - t2.ratio12) + abs(t1.ratio13 - t2.ratio13) + abs(t1.ratio21 - t2.ratio21) + \
         abs(t1.ratio23 - t2.ratio23) + abs(t1.ratio31 - t2.ratio31) + abs(t1.ratio32 - t2.ratio32)

    #score = d1 + d2 + d4 + d5
    score = d1 + d2 + d3+ d4 + d5

    return score
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
