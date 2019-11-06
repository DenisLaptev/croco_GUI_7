import numpy as np
import cv2
import matcher

# --------------------------------------------CONSTANTS--------------------------------------------
# FILE_TYPES
FILE_TYPE_CSV = 'csv'
FILE_TYPE_PKL = 'pkl'

PATH_TO_FOLDER_IMAGE_INITIAL = '../resources/images_initial/'
PATH_TO_FOLDER_IMAGE_GROUND_TRUTH = '../resources/images_photoshop/'
PATH_TO_FOLDER_FILE_CSV = '../resources/csv/'
PATH_TO_FOLDER_FILE_PKL = '../resources/pkl/'

# -------------------------csv_processing_util--------------------------------


# --------------------------------------------INPUT--------------------------------------------
file_type = FILE_TYPE_CSV

# --------------------------------------------OUTPUT--------------------------------------------
PATH_TO_FOLDER_OUTPUT = '../output/'
PATH_TO_FOLDER_OUTPUT_GT_CONTOURS = '../output/GT_contours/'


def d_nets_method(image1_number, image2_number):
    if image1_number == 1:
        FILE_NAME_IMAGE_INITIAL_1 = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells1.jpg'
        FILE_NAME_FILE_CSV_1 = 'csv1.txt'
        FILE_NAME_FILE_PKL_1 = ''
    elif image1_number == 2:
        FILE_NAME_IMAGE_INITIAL_1 = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV_1 = 'csv2.txt'
        FILE_NAME_FILE_PKL_1 = 'cells1.pkl'
    elif image1_number == 3:
        FILE_NAME_IMAGE_INITIAL_1 = 'small_image3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH = 'cells3_20160630_160547.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im3_modif.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'small_image3_GT.jpg'
        # FILE_NAME_FILE_CSV_1 = 'cells3_1.csv'
        # FILE_NAME_FILE_CSV_1 = 'mycsv3.csv'
        FILE_NAME_FILE_CSV_1 = 'first_image_for_triplets.csv'
        FILE_NAME_FILE_PKL_1 = 'cells2_2016-03-01_21.42.11.pkl'
    elif image1_number == 4:
        FILE_NAME_IMAGE_INITIAL_1 = 'small_image4.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH = 'cells4_20160630_160548.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im4.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'small_image4_GT.jpg'
        FILE_NAME_FILE_CSV_1 = 'mycsv4.csv'
        FILE_NAME_FILE_PKL_1 = 'cells4_20160630_160548.pkl'
    elif image1_number == 5:
        FILE_NAME_IMAGE_INITIAL_1 = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV_1 = 'csv5.txt'
        FILE_NAME_FILE_PKL_1 = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    if image2_number == 1:
        FILE_NAME_IMAGE_INITIAL_2 = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells1.jpg'
        FILE_NAME_FILE_CSV_2 = 'csv1.txt'
        FILE_NAME_FILE_PKL_2 = ''
    elif image2_number == 2:
        FILE_NAME_IMAGE_INITIAL_2 = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV_2 = 'csv2.txt'
        FILE_NAME_FILE_PKL_2 = 'cells1.pkl'
    elif image2_number == 3:
        FILE_NAME_IMAGE_INITIAL_2 = 'small_image3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells3_20160630_160547.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'im3.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'small_image3_GT.jpg'
        FILE_NAME_FILE_CSV_2 = 'mycsv3.csv'
        FILE_NAME_FILE_PKL_2 = 'cells2_2016-03-01_21.42.11.pkl'
    elif image2_number == 4:
        FILE_NAME_IMAGE_INITIAL_2 = 'small_image4.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells4_20160630_160548.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'im4.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'small_image4_GT.jpg'
        # FILE_NAME_FILE_CSV_2 = 'cells4_1_full.csv'
        FILE_NAME_FILE_CSV_2 = 'mycsv4.csv'
        FILE_NAME_FILE_PKL_2 = 'cells4_20160630_160548.pkl'
    elif image2_number == 5:
        FILE_NAME_IMAGE_INITIAL_2 = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV_2 = 'csv5.txt'
        FILE_NAME_FILE_PKL_2 = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    PATH_TO_IMAGE_INITIAL_1 = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL_1
    PATH_TO_IMAGE_INITIAL_2 = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL_2

    PATH_TO_IMAGE_GROUND_TRUTH_1 = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH_1
    PATH_TO_IMAGE_GROUND_TRUTH_2 = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH_2

    PATH_TO_FILE_CSV_1 = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV_1
    PATH_TO_FILE_CSV_2 = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV_2

    PATH_TO_FILE_PKL_1 = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL_1
    PATH_TO_FILE_PKL_2 = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL_2

    # read images
    # img1 = cv2.imread('data/gt/im3.jpg')
    # img2 = cv2.imread('data/gt/im4.jpg')

    img1 = cv2.imread(PATH_TO_IMAGE_GROUND_TRUTH_1)
    img2 = cv2.imread(PATH_TO_IMAGE_GROUND_TRUTH_2)

    img1_copy = img1.copy()
    img2_copy = img2.copy()

    # read contours
    # contours1 = matcher.read_contours('data/gt/cells3_1.csv')
    # contours2 = matcher.read_contours('data/gt/cells4_1.csv')

    path_to_file_csv_from_gui = '../resources/csv/contours3_from_gui.csv'
    contours1 = matcher.read_contours(path_to_file_csv_from_gui)
    contours2 = matcher.read_contours('../resources/csv/mycsv4_all.csv')
    # contours2 = matcher.read_contours('../resources/csv/mycsv4_all_50.csv')

    for cnt1 in contours1:
        img1_copy_with_cnts = cv2.drawContours(img1_copy, [cnt1], 0, 255, 2)
    # cv2.namedWindow('img1_copy_with_cnts', cv2.WINDOW_NORMAL)
    # cv2.imshow("img1_copy_with_cnts", img1_copy_with_cnts)
    # cv2.waitKey(0)

    for cnt2 in contours2:
        img2_copy_with_cnts = cv2.drawContours(img2_copy, [cnt2], 0, 255, 2)
    # cv2.namedWindow('img2_copy_with_cnts', cv2.WINDOW_NORMAL)
    # cv2.imshow("img2_copy_with_cnts", img2_copy_with_cnts)
    # cv2.waitKey(0)

    print('len(contours1)=', str(len(contours1)))
    # scale contours
    for it in contours1:
        # print('it=',it)
        for p in it:
            # p[0] = int(p[0] * 2.66)
            # p[1] = int(p[1] * 2.66)
            p[0] = int(p[0] * 1)
            p[1] = int(p[1] * 1)

    for it in contours2:
        for p in it:
            # p[0] = int(p[0] * 2.66)
            # p[1] = int(p[1] * 2.66)
            p[0] = int(p[0] * 1)
            p[1] = int(p[1] * 1)

    # draw contours
    viz1 = img1.copy()
    viz2 = img2.copy()

    for i in range(len(contours1)):
        viz1 = cv2.drawContours(viz1, contours1, i, (0, 0, 255), 2)

    for i in range(len(contours2)):
        viz2 = cv2.drawContours(viz2, contours2, i, (0, 0, 255), 2)

    # convert contours to contours data
    c1 = matcher.get_contours_data(contours1)
    c2 = matcher.get_contours_data(contours2)

    # get triplets from contours data
    triplets1 = matcher.extract_triplets(c1)
    triplets2 = matcher.extract_triplets2(c2)

    # match triplets
    matches = list()
    for i in range(len(triplets1)):

        best_score = 1e10
        best_j = -1

        matching_dict = {}

        for j in range(i, len(triplets2)):
            score = matcher.match_triplets(triplets1[i], triplets2[j])

            # matching_dict[score]=j
            # sorted_matching_dict=dict(sorted(matching_dict.items()))
            # print(sorted_matching_dict)

            #     if score < best_score:
            #         best_score = score
            #         best_j = j
            #
            # matches.append(matcher.Match(i, best_j, best_score))
            matches.append(matcher.Match(i, j, score))

    # sort matches
    matches.sort(key=lambda x: x.dist, reverse=False)
    # matches.sort(key=lambda x: x.dist, reverse=True)

    # get best 10 best matched triplets
    matches = matches[0:10]

    print('--------------------')
    for match in matches:
        print('match.dist=', match.dist)

    print('--------------------')
    MATCHER_LIMIT = matches[0].dist+matches[0].dist * 0.26
    print('MATCHER_LIMIT=', MATCHER_LIMIT)
    new_matches = []
    for match in matches:
        if match.dist <= MATCHER_LIMIT:
            new_matches.append(match)
            print('match.dist=', match.dist)

    # draw matches
    match_im = matcher.draw_matched_contours(img1, c1, img2, c2, triplets1, triplets2, new_matches)
    cv2.imwrite(PATH_TO_FOLDER_OUTPUT + 'matches' + '_image_' + str(image1_number) + '_and_image_' + str(
        image2_number) + '.png', match_im)

    return match_im, img1_copy_with_cnts, img2_copy_with_cnts


def dnets_get_triplets1_and_triplets2(image1_number, image2_number):
    if image1_number == 1:
        FILE_NAME_IMAGE_INITIAL_1 = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells1.jpg'
        FILE_NAME_FILE_CSV_1 = 'csv1.txt'
        FILE_NAME_FILE_PKL_1 = ''
    elif image1_number == 2:
        FILE_NAME_IMAGE_INITIAL_1 = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV_1 = 'csv2.txt'
        FILE_NAME_FILE_PKL_1 = 'cells1.pkl'
    elif image1_number == 3:
        FILE_NAME_IMAGE_INITIAL_1 = 'small_image3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH = 'cells3_20160630_160547.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im3_modif.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'small_image3_GT.jpg'
        # FILE_NAME_FILE_CSV_1 = 'cells3_1.csv'
        FILE_NAME_FILE_CSV_1 = 'mycsv3.csv'
        # FILE_NAME_FILE_CSV_1 = 'first_image_for_triplets.csv'
        FILE_NAME_FILE_PKL_1 = 'cells2_2016-03-01_21.42.11.pkl'
    elif image1_number == 4:
        FILE_NAME_IMAGE_INITIAL_1 = 'small_image4.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH = 'cells4_20160630_160548.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'im4.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'small_image4_GT.jpg'
        FILE_NAME_FILE_CSV_1 = 'mycsv4.csv'
        FILE_NAME_FILE_PKL_1 = 'cells4_20160630_160548.pkl'
    elif image1_number == 5:
        FILE_NAME_IMAGE_INITIAL_1 = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_1 = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV_1 = 'csv5.txt'
        FILE_NAME_FILE_PKL_1 = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    if image2_number == 1:
        FILE_NAME_IMAGE_INITIAL_2 = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells1.jpg'
        FILE_NAME_FILE_CSV_2 = 'csv1.txt'
        FILE_NAME_FILE_PKL_2 = ''
    elif image2_number == 2:
        FILE_NAME_IMAGE_INITIAL_2 = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV_2 = 'csv2.txt'
        FILE_NAME_FILE_PKL_2 = 'cells1.pkl'
    elif image2_number == 3:
        FILE_NAME_IMAGE_INITIAL_2 = 'small_image3.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells3_20160630_160547.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'im3.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'small_image3_GT.jpg'
        FILE_NAME_FILE_CSV_2 = 'mycsv3.csv'
        FILE_NAME_FILE_PKL_2 = 'cells2_2016-03-01_21.42.11.pkl'
    elif image2_number == 4:
        FILE_NAME_IMAGE_INITIAL_2 = 'small_image4.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells4_20160630_160548.jpg'
        # FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'im4.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'small_image4_GT.jpg'
        # FILE_NAME_FILE_CSV_2 = 'cells4_1_full.csv'
        FILE_NAME_FILE_CSV_2 = 'mycsv4.csv'
        FILE_NAME_FILE_PKL_2 = 'cells4_20160630_160548.pkl'
    elif image2_number == 5:
        FILE_NAME_IMAGE_INITIAL_2 = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH_2 = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV_2 = 'csv5.txt'
        FILE_NAME_FILE_PKL_2 = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    PATH_TO_IMAGE_INITIAL_1 = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL_1
    PATH_TO_IMAGE_INITIAL_2 = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL_2

    PATH_TO_IMAGE_GROUND_TRUTH_1 = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH_1
    PATH_TO_IMAGE_GROUND_TRUTH_2 = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH_2

    PATH_TO_FILE_CSV_1 = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV_1
    PATH_TO_FILE_CSV_2 = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV_2

    PATH_TO_FILE_PKL_1 = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL_1
    PATH_TO_FILE_PKL_2 = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL_2

    # read images
    # img1 = cv2.imread('data/gt/im3.jpg')
    # img2 = cv2.imread('data/gt/im4.jpg')

    img1 = cv2.imread(PATH_TO_IMAGE_GROUND_TRUTH_1)
    img2 = cv2.imread(PATH_TO_IMAGE_GROUND_TRUTH_2)

    img1_copy = img1.copy()
    img2_copy = img2.copy()

    # read contours
    # contours1 = matcher.read_contours('data/gt/cells3_1.csv')
    # contours2 = matcher.read_contours('data/gt/cells4_1.csv')

    contours1 = matcher.read_contours(PATH_TO_FILE_CSV_1)
    contours2 = matcher.read_contours(PATH_TO_FILE_CSV_2)

    # for cnt1 in contours1:
    # img1_copy_with_cnts = cv2.drawContours(img1_copy, [cnt1], 0, 255, 2)
    # cv2.namedWindow('img1_copy_with_cnts', cv2.WINDOW_NORMAL)
    # cv2.imshow("img1_copy_with_cnts", img1_copy_with_cnts)
    # cv2.waitKey(0)

    for cnt2 in contours2:
        img2_copy_with_cnts = cv2.drawContours(img2_copy, [cnt2], 0, 255, 2)
    # cv2.namedWindow('img2_copy_with_cnts', cv2.WINDOW_NORMAL)
    # cv2.imshow("img2_copy_with_cnts", img2_copy_with_cnts)
    # cv2.waitKey(0)

    print('len(contours1)=', str(len(contours1)))
    # scale contours
    for it in contours1:
        # print('it=',it)
        for p in it:
            # p[0] = int(p[0] * 2.66)
            # p[1] = int(p[1] * 2.66)
            p[0] = int(p[0] * 1)
            p[1] = int(p[1] * 1)

    for it in contours2:
        for p in it:
            # p[0] = int(p[0] * 2.66)
            # p[1] = int(p[1] * 2.66)
            p[0] = int(p[0] * 1)
            p[1] = int(p[1] * 1)

    # draw contours
    viz1 = img1.copy()
    viz2 = img2.copy()

    for i in range(len(contours1)):
        viz1 = cv2.drawContours(viz1, contours1, i, (0, 0, 255), 2)

    for i in range(len(contours2)):
        viz2 = cv2.drawContours(viz2, contours2, i, (0, 0, 255), 2)

    # convert contours to contours data
    c1 = matcher.get_contours_data(contours1)
    c2 = matcher.get_contours_data(contours2)

    print('c1[0].pts=', c1[5].pts)
    print('contours1[0]=', contours1[5])
    img1_copy_with_cnts = cv2.drawContours(img1_copy, [c1[5].pts], 0, [0, 255, 255], 2)
    # cv2.namedWindow('img2_copy_with_cnts', cv2.WINDOW_NORMAL)
    cv2.imshow("img1_copy_with_cnts", img1_copy_with_cnts)
    cv2.waitKey(0)

    # get triplets from contours data
    triplets1 = matcher.extract_triplets(c1)
    triplets2 = matcher.extract_triplets(c2)

    return triplets1, triplets2


def dnets_get_matches(triplets1, triplets2):
    # match triplets
    matches = list()
    for i in range(len(triplets1)):

        best_score = 1e10
        best_j = -1

        for j in range(i, len(triplets2)):

            score = matcher.match_triplets(triplets1[i], triplets2[j])

            if score < best_score:
                best_score = score
                best_j = j

        matches.append(matcher.Match(i, best_j, best_score))

    # sort matches
    matches.sort(key=lambda x: x.dist, reverse=False)

    # get best 10 best matched triplets
    matches = matches[0:10]

    return matches
