import numpy as np
import cv2
import matplotlib.pyplot as plt

import tkinter
import tkinter.messagebox

import PIL.Image, PIL.ImageTk

import pickle

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

def generate_contours_list_from_file(path_to_file, file_type=FILE_TYPE_CSV):
    contours_from_file = []

    if file_type == FILE_TYPE_CSV:
        lines = open(path_to_file).read().split('\n')
        for line in lines:
            firstPointFlag = True
            contour_points = np.empty((1, 2), dtype=np.int32)
            numbers = line.split(' ')
            i = 0
            while i < len(numbers) - 4:
                if numbers[i] != '':
                    x = int(numbers[i].strip())
                    y = int(numbers[i + 1].strip())
                    if firstPointFlag == True:
                        contour_points[0] = [x, y]
                        firstPointFlag = False
                    else:
                        contour_points = np.append(contour_points, [[x, y]], axis=0)
                i += 2
            contours_from_file.append(contour_points)
    elif file_type == FILE_TYPE_PKL:
        # with open(r'./resources/pkl/20160630_160547.pkl', 'rb') as file:
        with open(path_to_file, 'rb') as file:
            contours = pickle.load(file)

        # print(contours)
        contours_from_file = contours

    return contours_from_file


def generate_contours_list_from_GROUND_TRUTH_file(image, title, path_to_image_initial):
    image_copy = image.copy()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 20, 255, 0)

    kernel = np.ones((2, 2), np.uint8)

    dilation = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours.append(cnt)
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                image_copy = cv2.ellipse(image_copy, ellipse, (255, 0, 0), 2)

    PATH_TO_SAVE_FILE = PATH_TO_FOLDER_OUTPUT + str(title) + '_' + file_type + '.png'
    cv2.imwrite(PATH_TO_SAVE_FILE, image_copy)

    ##################Display_cnts_on_initial_image
    im_init = cv2.imread(path_to_image_initial)  # PATH_TO_IMAGE_INITIAL
    for cnt in new_contours:
        cv2.drawContours(im_init, [cnt], 0, (255, 0, 0), 2)
    cv2.namedWindow('initial_image', cv2.WINDOW_NORMAL)
    cv2.imshow('initial_image', im_init)
    # cv2.imwrite('./img5_with_GT_cnts.png', im_init)
    ##################Display_cnts_on_initial_image

    return new_contours


def convert_color(color):
    if color == 'B':
        cnt_color = (255, 0, 0)
    elif color == 'G':
        cnt_color = (0, 255, 0)
    elif color == 'R':
        cnt_color = (0, 0, 255)
    elif color == 'Y':
        cnt_color = (0, 255, 255)

    return cnt_color


def draw_contours_on_image(contours, color, image, title, isFilled=False):
    image_copy = image.copy()

    cnt_color = convert_color(color)

    for cnt in contours:
        if isFilled == False:
            image_copy = cv2.drawContours(image_copy, [cnt], 0, cnt_color, 2)
        if isFilled == True:
            image_copy = cv2.drawContours(image_copy, [cnt], 0, cnt_color, -1)


    PATH_TO_SAVE_FILE = PATH_TO_FOLDER_OUTPUT + str(title) + '_' + file_type + '.png'
    cv2.imwrite(PATH_TO_SAVE_FILE, image_copy)
    return image_copy


def draw_contours_on_image_from_list_of_dict(list_of_contours_dict, color, image, title):
    image_copy = image.copy()

    cnt_color = convert_color(color)

    for cnt_dict in list_of_contours_dict:
        if cnt_dict['is_found'] == True:
            cnt = cnt_dict['cnt']
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, -1)
        else:
            cnt = cnt_dict['cnt']
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, 2)


def find_cnt_center(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        cx = int(M['m10'] / (M['m00'] + 0.000001))
        cy = int(M['m01'] / (M['m00'] + 0.000001))
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_list_of_cnt_data_dict(contours, color):
    list = []
    id = 0
    for cnt in contours:
        dict = {}

        dict['cnt_id'] = id
        dict['cnt_color'] = convert_color(color)
        dict['cnt_center'] = find_cnt_center(cnt)
        dict['cnt_area'] = cv2.contourArea(cnt)
        dict['is_found'] = False
        dict['cnt'] = cnt

        list.append(dict)
        id += 1

    return list


def get_resized_image_for_validation(path_to_file, path_to_image_initial):
    image_init_1 = cv2.imread(path_to_image_initial)  # PATH_TO_IMAGE_INITIAL
    # image_init_1 = cv2.resize(image_init_1, None, fx=0.5, fy=0.5)
    height1, width1, _ = image_init_1.shape

    image_cells = cv2.imread(path_to_file)
    image_cells = cv2.resize(image_cells, (width1, height1))

    return image_cells


def compare_contours_lists(reference_contours_list_of_dict,
                           test_contours_list_of_dict,
                           image,
                           title,
                           cnts_center_displacement,
                           cnts_area_difference_factor):
    image_copy = image.copy()
    common_contours_list_of_dict = []
    fail_csv_contours_list_of_dict = []
    fail_GT_contours_list_of_dict = []
    common_contours_list = []

    for cnt1_dict in reference_contours_list_of_dict:

        cx_1, cy_1 = cnt1_dict['cnt_center']
        cnt1_area = cnt1_dict['cnt_area']
        for cnt2_dict in test_contours_list_of_dict:

            cx_2, cy_2 = cnt2_dict['cnt_center']
            cnt2_area = cnt2_dict['cnt_area']
            if cnt1_dict['is_found'] == False and \
                    cnt2_dict['is_found'] == False and \
                    abs(cx_1 - cx_2) < cnts_center_displacement * np.sqrt(cnt1_area) and \
                    abs(cy_1 - cy_2) < cnts_center_displacement * np.sqrt(cnt1_area) and \
                    abs(cnt1_area - cnt2_area) < cnts_area_difference_factor * cnt1_area:
                cnt1_dict['is_found'] = True
                cnt2_dict['is_found'] = True

                common_contours_list_of_dict.append(cnt1_dict)
                common_contours_list.append(cnt1_dict['cnt'])

    for cnt1_dict in reference_contours_list_of_dict:
        if cnt1_dict['is_found'] == False:
            fail_GT_contours_list_of_dict.append(cnt1_dict)

    for cnt2_dict in test_contours_list_of_dict:
        if cnt2_dict['is_found'] == False:
            fail_csv_contours_list_of_dict.append(cnt2_dict)

    image_with_common_cnts=draw_contours_on_image(contours=common_contours_list, color='G', image=image_copy, title=title, isFilled=True)

    return common_contours_list_of_dict, fail_GT_contours_list_of_dict, fail_csv_contours_list_of_dict,image_with_common_cnts


def calculate_area_of_contours(contours_list_of_dict):
    area = 0
    for cnt_dict in contours_list_of_dict:
        cnt_area = cnt_dict['cnt_area']
        area += cnt_area

    return area


def make_csv_from_cnts_list(path_to_save_csv_file, cnts_list):
    with open(path_to_save_csv_file, 'a') as file:
        for i in range(len(cnts_list)):
            cnt = cnts_list[i]
            number_of_points = len(cnt)
            # print('number_of_points=',number_of_points)
            print('cnt=', cnt)
            for j in range(number_of_points):
                print('cnt[' + str(j) + '][0][0]=', cnt[j][0][0])
                print('cnt[' + str(j) + '][0][1]=', cnt[j][0][1])
                # print('cnt['+str(j)+'][1]=',cnt[j][1])
                file.write(str(cnt[j][0][0]) + ', ' + str(cnt[j][0][1]) + ', ')
            file.write('\n')


# -------------------------csv_processing_util--------------------------------


def make_modified_image(src_image):
    src_image_copy = src_image.copy()
    result_image = cv2.ellipse(src_image_copy, (256, 256), (100, 50), 0, 0, 360, 255, -1)

    return result_image

################IMPORTANT_METHOD###########################
def get_GT_image(image_number):
    if image_number == 1:
        FILE_NAME_IMAGE_INITIAL = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells1.jpg'
        FILE_NAME_FILE_CSV = 'csv1.txt'
        FILE_NAME_FILE_PKL = ''
    elif image_number == 2:
        FILE_NAME_IMAGE_INITIAL = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV = 'csv2.txt'
        FILE_NAME_FILE_PKL = 'cells1.pkl'
    elif image_number == 3:
        #FILE_NAME_IMAGE_INITIAL = 'img3_20160630_160547.jpg'
        FILE_NAME_IMAGE_INITIAL = 'small_image3.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'cells3_20160630_160547.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'im3.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'small_image3_GT.jpg'
        #FILE_NAME_FILE_CSV = 'cells3_1.csv'
        FILE_NAME_FILE_CSV = 'mycsv3.csv'
        FILE_NAME_FILE_PKL = 'cells2_2016-03-01_21.42.11.pkl'
    elif image_number == 4:
        #FILE_NAME_IMAGE_INITIAL = 'img4_20160630_160548.jpg'
        FILE_NAME_IMAGE_INITIAL = 'small_image4.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'cells4_20160630_160548.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'im4.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'small_image4_GT.jpg'
        #FILE_NAME_FILE_CSV = 'cells4_1.csv'
        FILE_NAME_FILE_CSV = 'mycsv4.csv'
        FILE_NAME_FILE_PKL = 'cells4_20160630_160548.pkl'
    elif image_number == 5:
        FILE_NAME_IMAGE_INITIAL = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV = 'csv5.txt'
        FILE_NAME_FILE_PKL = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    PATH_TO_IMAGE_INITIAL = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL
    PATH_TO_IMAGE_GROUND_TRUTH = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH
    PATH_TO_FILE_CSV = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV
    PATH_TO_FILE_PKL = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL

    GT_image = cv2.imread(PATH_TO_IMAGE_GROUND_TRUTH)

    return GT_image


def get_image_with_cnts_from_csv(image_number):
    if image_number == 1:
        FILE_NAME_IMAGE_INITIAL = 'img1.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells1.jpg'
        FILE_NAME_FILE_CSV = 'csv1.txt'
        FILE_NAME_FILE_PKL = ''
    elif image_number == 2:
        FILE_NAME_IMAGE_INITIAL = 'img2_2016-03-01 21.42.11.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells2_2016-03-01 21.42.11.jpg'
        FILE_NAME_FILE_CSV = 'csv2.txt'
        FILE_NAME_FILE_PKL = 'cells1.pkl'
    elif image_number == 3:
        #FILE_NAME_IMAGE_INITIAL = 'img3_20160630_160547.jpg'
        FILE_NAME_IMAGE_INITIAL = 'small_image3.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'cells3_20160630_160547.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'small_image3_GT.jpg'
        #FILE_NAME_FILE_CSV = 'cells3_1.csv'
        FILE_NAME_FILE_CSV = 'mycsv3.csv'
        FILE_NAME_FILE_PKL = 'cells2_2016-03-01_21.42.11.pkl'
    elif image_number == 4:
        #FILE_NAME_IMAGE_INITIAL = 'img4_20160630_160548.jpg'
        FILE_NAME_IMAGE_INITIAL = 'small_image4.jpg'
        #FILE_NAME_IMAGE_GROUND_TRUTH = 'cells4_20160630_160548.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'small_image4_GT.jpg'
        #FILE_NAME_FILE_CSV = 'cells4_1.csv'
        FILE_NAME_FILE_CSV = 'mycsv4.csv'
        FILE_NAME_FILE_PKL = 'cells4_20160630_160548.pkl'
    elif image_number == 5:
        FILE_NAME_IMAGE_INITIAL = 'img5_croc82.jpg'
        FILE_NAME_IMAGE_GROUND_TRUTH = 'cells5_croc82_GT.png'
        FILE_NAME_FILE_CSV = 'csv5.txt'
        FILE_NAME_FILE_PKL = 'cells4_20160630_160548.pkl'
    else:
        print('INCORRECT NUMBER!')

    PATH_TO_IMAGE_INITIAL = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL
    PATH_TO_IMAGE_GROUND_TRUTH = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH
    PATH_TO_FILE_CSV = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV
    PATH_TO_FILE_PKL = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL

    image_cells = get_resized_image_for_validation(path_to_file=PATH_TO_IMAGE_GROUND_TRUTH,
                                                   path_to_image_initial=PATH_TO_IMAGE_INITIAL)

    image_cells_copy = image_cells.copy()

    contours_from_file = generate_contours_list_from_file(path_to_file=PATH_TO_FILE_CSV,
                                                          file_type=FILE_TYPE_CSV)

    contours_from_image_GROUND_TRUTH = generate_contours_list_from_GROUND_TRUTH_file(image=image_cells,
                                                                                     title='Ellipses_' + FILE_NAME_IMAGE_INITIAL[
                                                                                                         :-4],
                                                                                     path_to_image_initial=PATH_TO_IMAGE_INITIAL)

    # ----------------------------------VALIDATION--------------------------------------------
    list_of_file_cnt_data_dict = get_list_of_cnt_data_dict(contours=contours_from_file, color='Y')

    list_of_GT_1_cnt_data_dict = get_list_of_cnt_data_dict(contours=contours_from_image_GROUND_TRUTH, color='G')



    # ------------------------------DRAW_RESULTS--------------------------------------------
    result_image = draw_contours_on_image(contours=contours_from_file,
                                          color='G',
                                          image=image_cells,
                                          title='method_cnts_' + FILE_NAME_IMAGE_INITIAL[:-4])

    draw_contours_on_image_from_list_of_dict(list_of_contours_dict=list_of_GT_1_cnt_data_dict,
                                             color='Y',
                                             image=image_cells,
                                             title='GT_cnts_' + FILE_NAME_IMAGE_INITIAL[:-4])
    return result_image


def generate_contours_list_from_GROUND_TRUTH_file(image):
    image_copy = image.copy()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 20, 255, 0)

    kernel = np.ones((2, 2), np.uint8)

    dilation = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours.append(cnt)
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                image_copy = cv2.ellipse(image_copy, ellipse, (255, 0, 0), 2)


    return new_contours


def make_csv_from_cnts_list(path_to_save_csv_file, cnts_list):
    with open(path_to_save_csv_file, 'a') as file:
        for i in range(len(cnts_list)):
            cnt = cnts_list[i]
            number_of_points = len(cnt)
            # print('number_of_points=',number_of_points)
            # print('cnt=', cnt)
            for j in range(number_of_points):
                #print('cnt[' + str(j) + '][0][0]=', cnt[j][0][0])
                #print('cnt[' + str(j) + '][0][1]=', cnt[j][0][1])
                # print('cnt['+str(j)+'][1]=',cnt[j][1])
                file.write(str(cnt[j][0][0]) + ', ' + str(cnt[j][0][1]) + ', ')
            file.write('\n')