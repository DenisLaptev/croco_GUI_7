import tkinter
import tkinter.messagebox
import cv2
import PIL.Image, PIL.ImageTk
import os
import numpy as np
import matcher

from controller_image_transforms import *
from controller_dnets import *

NUMBER_OF_CONTOURS = 3

R_MIN = 0.1
R_MAX = 5.0

MIN_ANGLE = 0.1
MAX_ANGLE = 1.57
#MAX_ANGLE = 2.1

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
    if os.path.exists("../resources/csv/contours3_from_gui.csv"):
        os.remove("../resources/csv/contours3_from_gui.csv")


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

    if len(test_cnts_list) == NUMBER_OF_CONTOURS:

        tkinter.messagebox.showinfo(title="Welcome",
                                    message="Run App!\n" + "contours number=" + str(len(test_cnts_list)))

        test_cnts_list
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
    for cnt in new_contours1:
        cv_img1 = cv2.drawContours(cv_img1, [cnt], 0, [0, 255, 0], -1)

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

    # path_to_file_csv_from_gui = '../resources/csv/mycsv4_all_50.csv'
    path_to_file_csv_from_gui = '../resources/csv/mycsv4_all.csv'
    contours2 = matcher.read_contours(path_to_file_csv_from_gui)

    new_contours2 = []
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours2.append(cnt)

    cv_img2 = image
    for cnt in new_contours2:
        cv_img2 = cv2.drawContours(cv_img2, [cnt], 0, [0, 255, 0], -1)

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
    global cv_img3

    global gt_image1
    global gt_image2

    global photo1
    global photo2
    global photo3

    global c1

    global contours1_list
    global contours2_list

    global test_cnts_list
    global list_of_available_cnts_ids_all

    cv2.circle(cv_img1, (event.x, event.y), 3, (255, 0, 0), -1)

    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

    # get GT of images with numbers=image1_number and image2_number
    gt_image1 = get_GT_image(image1_number)
    gt_image2 = get_GT_image(image2_number)

    # get GT contours
    contours1_list = generate_contours_list_from_GROUND_TRUTH_file(gt_image1)
    # print('len(contours1_list)=', len(contours1_list))

    for i1 in range(len(contours1_list)):
        dist = cv2.pointPolygonTest(contours1_list[i1], (event.x, event.y), True)
        if dist > 0:  # Positive value if the point is inside the contour
            if len(test_cnts_list) < NUMBER_OF_CONTOURS and is_unique_contour(test_cnts_list,
                                                                              contours1_list[i1]) == True:
                test_cnts_list.append(contours1_list[i1])
                print('len(test_cnts_list)=', len(test_cnts_list))

                if len(test_cnts_list) == 1:
                    cv_img1 = get_GT_image(image1_number)
                    cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)

                    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                    print('1 cnt in test_cnts_list')
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[0]], 0, [0, 255, 0], -1)

                    list_of_available_cnts_ids_case1 = c1[i1].vn
                    list_of_available_cnts_ids_all = list_of_available_cnts_ids_case1

                    f = open('cnts1', 'w')
                    for cnt1_id in list_of_available_cnts_ids_case1:
                        f.write(str(cnt1_id) + ', ')
                    f.close()


                elif len(test_cnts_list) == 2:
                    cv_img1 = get_GT_image(image1_number)
                    cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)

                    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                    print('2 cnt in test_cnts_list')
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[0]], 0, [0, 255, 0], -1)
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[1]], 0, [0, 255, 0], -1)

                    file1 = open('cnts1', 'r')
                    text1 = file1.read()
                    list_of_available_cnts_ids_case1 = text1.split(', ')

                    list_of_available_cnts_ids_case2 = c1[i1].vn



                    list_of_available_cnts_ids_all = []
                    new_list_of_available_cnts_ids_all = []
                    for cnt111 in list_of_available_cnts_ids_case1:
                        for cnt222 in list_of_available_cnts_ids_case2:
                            if len(cnt111) > 0:
                                if int(cnt111) == int(cnt222):
                                    list_of_available_cnts_ids_all.append(int(cnt111))
                    # for id3 in list_of_available_cnts_ids_all:
                    #     flag=True
                    #     for cnt111 in list_of_available_cnts_ids_case1:
                    #         for cnt222 in list_of_available_cnts_ids_case2:
                    #             if len(cnt111) > 0:
                    #                 if int(cnt111) == int(cnt222):
                    #                     temp = matcher.init_triplet(c1, int(cnt111), int(cnt222), int(id3))
                    #                     # validate triplet
                    #                     if matcher.is_good_triplet(temp, MIN_ANGLE, MAX_ANGLE)==False:
                    #                         flag=False
                    #     if flag==True:
                    #         new_list_of_available_cnts_ids_all.append(int(cnt111))
                    # list_of_available_cnts_ids_all=new_list_of_available_cnts_ids_all
                    f = open('cnts2', 'w')
                    for cnt2_id in list_of_available_cnts_ids_all:
                        f.write(str(cnt2_id) + ', ')
                    f.close()
                    print('list_of_available_cnts_ids_case1=', list_of_available_cnts_ids_case1)
                    print('list_of_available_cnts_ids_case2=', list_of_available_cnts_ids_case2)
                    print('list_of_available_cnts_ids_all=', list_of_available_cnts_ids_all)

                elif len(test_cnts_list) == 3:

                    cv_img1 = get_GT_image(image1_number)
                    cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)

                    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                    print('3 cnt in test_cnts_list')
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[0]], 0, [0, 255, 0], -1)
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[1]], 0, [0, 255, 0], -1)
                    cv_img1 = cv2.drawContours(cv_img1, [test_cnts_list[2]], 0, [0, 255, 0], -1)

                    file1 = open('cnts1', 'r')
                    text1 = file1.read()
                    list_of_available_cnts_ids_case1 = text1.split(', ')

                    file2 = open('cnts2', 'r')
                    text2 = file2.read()
                    list_of_available_cnts_ids_case2 = text2.split(', ')

                    list_of_available_cnts_ids_case3 = c1[i1].vn

                    list_of_available_cnts_ids_all = []
                    for cnt111 in list_of_available_cnts_ids_case1:
                        for cnt222 in list_of_available_cnts_ids_case2:
                            for cnt333 in list_of_available_cnts_ids_case3:
                                if len(cnt111) > 0 and len(cnt222) > 0:
                                    if int(cnt111) == int(cnt222) and int(cnt111) == int(cnt333):
                                        list_of_available_cnts_ids_all.append(int(cnt111))
                    print('list_of_available_cnts_ids_case1=', list_of_available_cnts_ids_case1)
                    print('list_of_available_cnts_ids_case2=', list_of_available_cnts_ids_case2)
                    print('list_of_available_cnts_ids_case3=', list_of_available_cnts_ids_case3)
                    print('list_of_available_cnts_ids_all=', list_of_available_cnts_ids_all)

                for num in list_of_available_cnts_ids_all:
                    cv_img1 = cv2.drawContours(cv_img1, [contours1_list[num]], 0, [255, 255, 0], 2)

                photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                if len(test_cnts_list) == NUMBER_OF_CONTOURS:
                    tkinter.messagebox.showinfo(title="Triplet generation",
                                                message="Well done! " + str(NUMBER_OF_CONTOURS) + " contours chosen!")
                    make_csv_from_cnts_list(path_to_save_csv_file='../resources/csv/contours3_from_gui.csv',
                                            cnts_list=test_cnts_list)
                    try:
                        result_image3, img1_with_cnts, img2_with_cnts = d_nets_method(image1_number, image2_number)
                    except Exception as e:
                        print(e)
                        tkinter.messagebox.showinfo(title="Error!",message="Bad contours. Try again")
                        exit_method()
                    cv_img1 = img1_with_cnts
                    cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)

                    cv_img2 = img2_with_cnts
                    cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2RGB)

                    cv_img3 = result_image3
                    cv_img3 = cv2.cvtColor(cv_img3, cv2.COLOR_BGR2RGB)

                    dim = (width * 2, height)
                    # resize image
                    cv_img3 = cv2.resize(cv_img3, dim, interpolation=cv2.INTER_AREA)

                    photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img1))
                    canvas1.create_image(0, 0, image=photo1, anchor=tkinter.NW)

                    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img2))
                    canvas2.create_image(0, 0, image=photo2, anchor=tkinter.NW)

                    photo3 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img3))
                    canvas3.create_image(0, 0, image=photo3, anchor=tkinter.NW)

            elif len(test_cnts_list) == NUMBER_OF_CONTOURS:
                tkinter.messagebox.showinfo(title="Triplet generation", message="Contours have been already chosen!")


# Create a window
window = tkinter.Tk()
window.title("OpenCV and Tkinter")
window.geometry("1400x700")
recreate_output_file()

test_cnts_list = []
list_of_available_cnts_ids_all = []

path_to_file_csv_from_gui = '../resources/csv/mycsv3_all.csv'
contours1_forcolorhelp = matcher.read_contours(path_to_file_csv_from_gui)
c1 = matcher.get_contours_data(contours1_forcolorhelp)
for i in range(len(c1)):
    # check vertice status
    if c1[i].status == False:
        continue
    # find_triplet_neighbours(c, i, 15.0, 20.0)
    matcher.find_triplet_neighbours(c1, i, R_MIN, R_MAX)
    # print('c1[i].vn=',c1[i].vn)

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
