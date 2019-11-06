import tkinter
import tkinter.messagebox
import cv2
import PIL.Image, PIL.ImageTk
import os
import numpy as np

from controller_image_transforms import *
from controller_dnets import *

image1_number = 3
image2_number = 4

PATH_TO_IMAGE_3 = "../resources/images_initial/small_image3.jpg"
PATH_TO_IMAGE_4 = "../resources/images_initial/small_image4.jpg"

PATH_TO_IMAGE_1 = PATH_TO_IMAGE_3  # "../resources/images_initial/img1.jpg"
PATH_TO_IMAGE_2 = PATH_TO_IMAGE_3  # "../resources/images_initial/img2_2016-03-01 21.42.11.jpg"
PATH_TO_IMAGE_3 = PATH_TO_IMAGE_3  # "../resources/images_initial/img3_20160630_160547.jpg"
PATH_TO_IMAGE_4 = PATH_TO_IMAGE_4
PATH_TO_IMAGE_5 = PATH_TO_IMAGE_3  # "../resources/images_initial/img5_croc82.jpg"

if image1_number == 1:
    path_to_image1 = PATH_TO_IMAGE_1
elif image1_number == 2:
    path_to_image1 = PATH_TO_IMAGE_2
elif image1_number == 3:
    path_to_image1 = PATH_TO_IMAGE_3
elif image1_number == 4:
    path_to_image1 = PATH_TO_IMAGE_4
elif image1_number == 5:
    path_to_image1 = PATH_TO_IMAGE_5
else:
    path_to_image1 = PATH_TO_IMAGE_1

if image2_number == 1:
    path_to_image2 = PATH_TO_IMAGE_1
elif image2_number == 2:
    path_to_image2 = PATH_TO_IMAGE_2
elif image2_number == 3:
    path_to_image2 = PATH_TO_IMAGE_3
elif image2_number == 4:
    path_to_image2 = PATH_TO_IMAGE_4
elif image2_number == 5:
    path_to_image2 = PATH_TO_IMAGE_5
else:
    path_to_image2 = PATH_TO_IMAGE_1


def recreate_output_file():
    if os.path.exists("../resources/csv/first_image_for_triplets.csv"):
        os.remove("../resources/csv/first_image_for_triplets.csv")


def is_unique_contour(test_cnts_list, contour):
    result = True
    for cnt in test_cnts_list:
        if cnt == contour:
            result = False
            break

    return result


def about_method():
    global cv_img1
    global cv_img2
    global cv_img3

    global gt_image1
    global gt_image2

    global photo1
    global photo2
    global photo3

    global test_cnts_list

    if len(test_cnts_list) == 3:

        tkinter.messagebox.showinfo(title="Welcome",
                                    message="Run App!\n" + "contours number=" + str(len(test_cnts_list)))

        list_of_contours_dict_1 = create_list_of_contours_dict(test_cnts_list)
        # print(list_of_contours_dict_1)

        print('--------------------------------------')

        contours2 = find_contours_of_image_2()
        list_of_contours_dict_2 = create_list_of_contours_dict(contours2)
        # print(list_of_contours_dict_2)

        # cv_img2 = cv2.drawContours(cv_img2, [list_of_contours_dict_2[350]['cnt']], 0, [0, 0, 255], -1)
        # photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
        # canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)
        #
        # print(list_of_contours_dict_2[350]['area'], list_of_contours_dict_1[0]['area'])

        alpha12_linear = np.sqrt(182 / 147)
        print('alpha12_linear=', alpha12_linear)

        cx1, cy1 = list_of_contours_dict_1[0]['center']
        cx2, cy2 = list_of_contours_dict_1[1]['center']
        cx3, cy3 = list_of_contours_dict_1[2]['center']

        distance12_test = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        distance13_test = np.sqrt((cx1 - cx3) ** 2 + (cy1 - cy3) ** 2)
        distance23_test = np.sqrt((cx2 - cx3) ** 2 + (cy2 - cy3) ** 2)

        vx12_test = cx1 - cx2
        vy12_test = cy1 - cy2

        vx13_test = cx1 - cx3
        vy13_test = cy1 - cy3

        vx23_test = cx2 - cx3
        vy23_test = cy2 - cy3

        ex12_test = vx12_test / distance12_test
        ey12_test = vy12_test / distance12_test

        ex13_test = vx13_test / distance13_test
        ey13_test = vy13_test / distance13_test

        ex23_test = vx23_test / distance23_test
        ey23_test = vy23_test / distance23_test

        # print('ex_test=',ex_test)
        # print('ey_test=',ey_test)
        # print('e_abs=',ex_test**2+ey_test**2)

        list_of_good_contours_dict1 = []
        list_of_good_contours_dict2 = []
        list_of_good_contours_dict3 = []

        for contour_dict_2 in list_of_contours_dict_2:

            if abs(contour_dict_2['area'] - list_of_contours_dict_1[0]['area'] * (alpha12_linear) ** 2) <= 0.2 * \
                    list_of_contours_dict_1[0]['area'] * (alpha12_linear) ** 2 and abs(
                    contour_dict_2['perimeter'] - list_of_contours_dict_1[0]['perimeter'] * (
                    alpha12_linear) ** 1) <= 0.2 * list_of_contours_dict_1[0]['perimeter'] * (alpha12_linear) ** 1:
                # print(contour_dict_2['area'],list_of_contours_dict_1[0]['area']*(alpha12_linear)**2 )
                cv_img2 = cv2.drawContours(cv_img2, [contour_dict_2['cnt']], 0, [0, 0, 255], -1)
                photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
                canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

                list_of_good_contours_dict1.append(contour_dict_2)

            if abs(contour_dict_2['area'] - list_of_contours_dict_1[1]['area'] * (alpha12_linear) ** 2) <= 0.2 * \
                    list_of_contours_dict_1[1]['area'] * (alpha12_linear) ** 2 and abs(
                    contour_dict_2['perimeter'] - list_of_contours_dict_1[1]['perimeter'] * (
                    alpha12_linear) ** 1) <= 0.2 * list_of_contours_dict_1[1]['perimeter'] * (alpha12_linear) ** 1:
                print(contour_dict_2['area'], list_of_contours_dict_1[1]['area'] * (alpha12_linear) ** 2)
                cv_img2 = cv2.drawContours(cv_img2, [contour_dict_2['cnt']], 0, [0, 255, 0], -1)
                photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
                canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

                list_of_good_contours_dict2.append(contour_dict_2)

            if abs(contour_dict_2['area'] - list_of_contours_dict_1[2]['area'] * (alpha12_linear) ** 2) <= 0.2 * \
                    list_of_contours_dict_1[2]['area'] * (alpha12_linear) ** 2 and abs(
                    contour_dict_2['perimeter'] - list_of_contours_dict_1[2]['perimeter'] * (
                    alpha12_linear) ** 1) <= 0.2 * list_of_contours_dict_1[2]['perimeter'] * (alpha12_linear) ** 1:
                print(contour_dict_2['area'], list_of_contours_dict_1[2]['area'] * (alpha12_linear) ** 2)
                cv_img2 = cv2.drawContours(cv_img2, [contour_dict_2['cnt']], 0, [255, 0, 0], -1)
                photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
                canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

                list_of_good_contours_dict3.append(contour_dict_2)

        for contour_dict1 in list_of_good_contours_dict1:
            for contour_dict2 in list_of_good_contours_dict2:
                for contour_dict3 in list_of_good_contours_dict3:

                    print('Hi')
                    cx1, cy1 = contour_dict1['center']
                    cx2, cy2 = contour_dict2['center']
                    cx3, cy3 = contour_dict3['center']

                    distance12_image2 = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                    distance13_image2 = np.sqrt((cx1 - cx3) ** 2 + (cy1 - cy3) ** 2)
                    distance23_image2 = np.sqrt((cx2 - cx3) ** 2 + (cy2 - cy3) ** 2)

                    vx12_image2 = cx1 - cx2
                    vy12_image2 = cy1 - cy2

                    vx13_image2 = cx1 - cx3
                    vy13_image2 = cy1 - cy3

                    vx23_image2 = cx2 - cx3
                    vy23_image2 = cy2 - cy3

                    ex12_image2 = vx12_image2 / distance12_image2
                    ey12_image2 = vy12_image2 / distance12_image2

                    ex13_image2 = vx13_image2 / distance13_image2
                    ey13_image2 = vy13_image2 / distance13_image2

                    ex23_image2 = vx23_image2 / distance23_image2
                    ey23_image2 = vy23_image2 / distance23_image2

                    print('ex12_test=', ex12_test)
                    print('ey12_test=', ey12_test)
                    print('e12_test_abs=', ex12_test ** 2 + ey12_test ** 2)

                    print('ex13_test=', ex13_test)
                    print('ey13_test=', ey13_test)
                    print('e13_test_abs=', ex13_test ** 2 + ey13_test ** 2)

                    print('ex23_test=', ex23_test)
                    print('ey23_test=', ey23_test)
                    print('e23_test_abs=', ex23_test ** 2 + ey23_test ** 2)

                    print('ex12_image2=', ex12_image2)
                    print('ey12_image2=', ey12_image2)
                    print('e12_image2_abs=', ex12_image2 ** 2 + ey12_image2 ** 2)

                    print('ex13_image2=', ex13_image2)
                    print('ey13_image2=', ey13_image2)
                    print('e13_image2_abs=', ex13_image2 ** 2 + ey13_image2 ** 2)

                    print('ex23_image2=', ex23_image2)
                    print('ey23_image2=', ey23_image2)
                    print('e23_image2_abs=', ex23_image2 ** 2 + ey23_image2 ** 2)

                    if abs(distance12_test * (alpha12_linear) ** 1 - distance12_image2) <= 0.3 * distance12_test * (
                    alpha12_linear) ** 1:
                        if ex12_test * ex12_image2 >= 0 and ey12_test * ey12_image2 >= 0:
                            if (ex12_test * ey12_test) / (ex12_image2 * ey12_image2) >= 0:
                                if abs(abs((ex12_test / ey12_test)) - abs((ex12_image2 / ey12_image2))) < 0.3:

                                    if abs(distance13_test * (
                                            alpha12_linear) ** 1 - distance13_image2) <= 0.3 * distance13_test * (
                                            alpha12_linear) ** 1:
                                        if ex13_test * ex13_image2 >= 0 and ey13_test * ey13_image2 >= 0:
                                            if (ex13_test * ey13_test) / (ex13_image2 * ey13_image2) >= 0:
                                                if abs(abs((ex13_test / ey13_test)) - abs((ex13_image2 / ey13_image2))) < 0.3:

                                                    if abs(distance23_test * (
                                                    alpha12_linear) ** 1 - distance23_image2) <= 0.3 * distance23_test * (
                                                    alpha12_linear) ** 1:
                                                        if ex23_test * ex23_image2 >= 0 and ey23_test * ey23_image2 >= 0:
                                                            if (ex23_test * ey23_test) / (ex23_image2 * ey23_image2) >= 0:
                                                                if abs(abs((ex23_test / ey23_test)) - abs(
                                                                    (ex23_image2 / ey23_image2))) < 0.3:

                                                                    cv_img2 = cv2.drawContours(cv_img2, [contour_dict1['cnt']], 0, [255, 255, 255], -1)
                                                                    cv_img2 = cv2.drawContours(cv_img2, [contour_dict2['cnt']], 0, [255, 255, 255], -1)
                                                                    cv_img2 = cv2.drawContours(cv_img2, [contour_dict3['cnt']], 0, [255, 255, 255], -1)
                                                                    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
                                                                    canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

        print("about_method")
    else:
        tkinter.messagebox.showinfo(title="Welcome",
                                    message="Choose 3 contours!\n" + "contours number=" + str(len(test_cnts_list)))


def exit_method():
    print("exit_method")
    exit()


def GT_image1_method():
    global photo1

    global cv_img1

    global image1_number

    global path_to_image1

    global test_cnts_list

    cv_img1 = get_GT_image(image1_number)
    cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)

    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

    test_cnts_list = []

    print("GT_image1_method")


def GT_image2_method():
    global photo2

    global cv_img2

    global image2_number

    global path_to_image2

    global test_cnts_list

    cv_img2 = get_GT_image(image2_number)
    cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2RGB)

    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
    canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

    print("GT_image2_method")


def find_contours_of_image_1():
    global cv_img1
    global cv_img2
    global cv_img3

    global gt_image1
    global gt_image2

    global photo1
    global photo2
    global photo3

    global test_cnts_list

    image = get_GT_image(image1_number)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 20, 255, 0)

    kernel = np.ones((2, 2), np.uint8)

    dilation = cv2.dilate(thresh, kernel, iterations=1)

    contours1, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours1 = []
    for cnt in contours1:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours1.append(cnt)

    cv_img1 = image
    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

    test_cnts_list = []

    return new_contours1


def find_contours_of_image_2():
    global cv_img1
    global cv_img2
    global cv_img3

    global gt_image1
    global gt_image2

    global photo1
    global photo2
    global photo3

    image = get_GT_image(image2_number)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 20, 255, 0)

    kernel = np.ones((2, 2), np.uint8)

    dilation = cv2.dilate(thresh, kernel, iterations=1)

    contours2, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours2 = []
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours2.append(cnt)

    cv_img2 = image
    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
    canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

    return new_contours2


def reset_image1_method():
    global photo1

    global cv_img1

    global image1_number

    global path_to_image1

    global test_cnts_list

    recreate_output_file()

    cv_img1 = cv2.cvtColor(cv2.imread(path_to_image1), cv2.COLOR_BGR2RGB)
    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

    test_cnts_list = []

    print("reset_image1_method")


def reset_image2_method():
    global photo2

    global cv_img2

    global image2_number

    global path_to_image2

    global test_cnts_list

    cv_img2 = cv2.cvtColor(cv2.imread(path_to_image2), cv2.COLOR_BGR2RGB)
    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
    canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

    test_cnts_list = []

    print("reset_image2_method")


def left_mouse_button_pressed_method(event):
    global cv_img1
    global cv_img2

    global gt_image1
    global gt_image2

    global photo1

    global contours1_list
    global contours2_list

    global test_cnts_list

    cv2.circle(cv_img1, (event.x, event.y), 3, (255, 0, 0), -1)

    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

    # get GT of images with numbers=image1_number and image2_number
    gt_image1 = get_GT_image(image1_number)
    gt_image2 = get_GT_image(image2_number)

    # get GT contours
    contours1_list = generate_contours_list_from_GROUND_TRUTH_file(gt_image1)
    print('len(contours1_list)=', len(contours1_list))

    for cnt1 in contours1_list:
        dist = cv2.pointPolygonTest(cnt1, (event.x, event.y), True)
        if dist > 0:  # Positive value if the point is inside the contour
            if len(test_cnts_list) < 3 and is_unique_contour(test_cnts_list, cnt1) == True:
                test_cnts_list.append(cnt1)
                print('len(test_cnts_list)=', len(test_cnts_list))

                cv_img1 = cv2.drawContours(cv_img1, [cnt1], 0, [0, 255, 0], -1)
                photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                if len(test_cnts_list) == 3:
                    tkinter.messagebox.showinfo(title="Triplet generation", message="Well done! 3 contours chosen!")

            elif len(test_cnts_list) == 3:
                tkinter.messagebox.showinfo(title="Triplet generation", message="Contours have been already chosen!")


def create_list_of_contours_dict(contours_list):
    list_of_contours_dict = []
    cnt_id = 0
    for cnt in contours_list:
        contours_dict = {}
        contours_dict['id'] = cnt_id
        contours_dict['cnt'] = cnt
        contours_dict['area'] = cv2.contourArea(cnt)
        contours_dict['perimeter'] = cv2.arcLength(cnt, True)
        contours_dict['arper'] = cv2.contourArea(cnt) / cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            cx = int(M['m10'] / (M['m00'] + 0.000001))
            cy = int(M['m01'] / (M['m00'] + 0.000001))
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        contours_dict['center'] = (cx, cy)
        contours_dict['order'] = -1

        list_of_contours_dict.append(contours_dict)
        cnt_id += 1
    return list_of_contours_dict


def sort_list_of_contours_dict_in_triplet(list_of_contours_dict):
    cx1, cy1 = list_of_contours_dict[0]['center']
    cx2, cy2 = list_of_contours_dict[1]['center']
    cx3, cy3 = list_of_contours_dict[2]['center']

    if np.min(cy1, cy2, cy3) == cy1:
        pass


def get_contour_by_order(list_of_contours_dict, order):
    result_contour = None
    for contour_dict in list_of_contours_dict:
        if contour_dict['order'] == order:
            result_contour = contour_dict
    return result_contour


def create_list_of_triplets_dict(list_of_contours_dict_for_triplet):
    list_of_triplets_dict = []

    for contour_dict in list_of_contours_dict_for_triplet:
        pass


# Create a window
window = tkinter.Tk()
window.title("OpenCV and Tkinter")
window.geometry("1400x700")
recreate_output_file()

test_cnts_list = []

label_title = tkinter.Label(master=window,
                            width=20,
                            text="Crocodiles GUI Util",
                            font=("arial", 19, "bold")
                            ).place(x=90, y=5)

# Load an image using OpenCV
cv_img1 = cv2.cvtColor(cv2.imread(path_to_image1), cv2.COLOR_BGR2RGB)
cv_img2 = cv2.cvtColor(cv2.imread(path_to_image2), cv2.COLOR_BGR2RGB)

# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, channels = cv_img1.shape

# Create a canvas that can fit the above image
canvas1 = tkinter.Canvas(master=window, width=width, height=height)
canvas2 = tkinter.Canvas(master=window, width=width, height=height)
canvas3 = tkinter.Canvas(master=window, width=width * 2, height=height)

canvas1.place(x=10, y=10)
canvas2.place(x=700, y=10)
canvas3.place(x=250, y=400)

# bind mouse events to canvas
canvas1.bind(sequence="<Button-1>", func=left_mouse_button_pressed_method)

# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))

# Add a PhotoImage to the Canvas
canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)
canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

# contours1_list = generate_contours_list_from_GROUND_TRUTH_file(cv_img1)
# contours2_list = generate_contours_list_from_GROUND_TRUTH_file(cv_img2)

# Button Run
btn_run = tkinter.Button(window, text="Run", width=15, command=about_method)
btn_run.place(x=20, y=500)

# Button Exit
btn_exit = tkinter.Button(window, text="Exit", width=15, command=exit_method)
btn_exit.place(x=20, y=550)

# Button GT image
btn_GT1 = tkinter.Button(window, text="GT", width=15, command=GT_image1_method)
btn_GT1.place(x=20, y=330)

btn_GT2 = tkinter.Button(window, text="GT", width=15, command=GT_image2_method)
btn_GT2.place(x=720, y=330)

# Button Reset
btn_reset1 = tkinter.Button(window, text="Reset", width=15, command=reset_image1_method)
btn_reset1.place(x=170, y=330)

btn_reset2 = tkinter.Button(window, text="Reset", width=15, command=reset_image2_method)
btn_reset2.place(x=870, y=330)

# Button Contours
btn_cnts1 = tkinter.Button(window, text="cnts", width=15, command=find_contours_of_image_1)
btn_cnts1.place(x=320, y=330)

btn_cnts2 = tkinter.Button(window, text="cnts", width=15, command=find_contours_of_image_2)
btn_cnts2.place(x=1020, y=330)

# Create menu
menu = tkinter.Menu(master=window)
window.config(menu=menu)

submenu_1 = tkinter.Menu(master=menu)
menu.add_cascade(label="File", menu=submenu_1)
submenu_1.add_command(label="Exit", command=exit_method)

submenu_2 = tkinter.Menu(master=menu)
menu.add_cascade(label="Options", menu=submenu_2)
submenu_2.add_command(label="About", command=about_method)

# Run the window loop
window.mainloop()
