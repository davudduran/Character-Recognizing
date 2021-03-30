import queue
import threading
import subprocess
import PySimpleGUI as sg
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

def main(filename, threshhold, gui_queue):
    image = Image.open(filename)
    print("Taken image:", image.size, "as", image.format)
    format = "." + str(image.format).lower()
    imArray = np.empty([image.size[1], image.size[0]])
    print("Binarizing Image with value", threshhold)
    binarized_image, imArray = threshold(image, imArray, threshhold)
    binarized_image.save(save_dir + "\\0binarized_image" + format)
    binarized_image.thumbnail((800, 800))
    window["-IMAGE-"].update(data=ImageTk.PhotoImage(binarized_image),size=(800,800))

    print("Labeling Image, this may take a while.")
    arr_labeled_img, num_labels, arr_blobs, labels = blob_coloring_8_connected(imArray, 255)  # arr blobs holds colors
    # print the number of objects as the number of different labels
    print("Labeling finished!\n" + str(num_labels) + " objects found in the input image.")
    # write the values in the labeled image to a file
    # convert the numpy array of the colored components (blobs) to a PIL Image
    img_blobs = Image.fromarray(arr_blobs)
    # img_blobs.show()  # display the colored components (blobs)
    img_blobs.save(save_dir + "\\1labeled_image" + format)
    img_with_rectangles = img_blobs.copy()
    img_blobs.thumbnail((800,800))
    window["-IMAGE-"].update(data=ImageTk.PhotoImage(img_blobs),size=(800,800))

    min_x, min_y, max_x, max_y = findObjects(num_labels, arr_labeled_img, labels)
    coloredEdit = ImageDraw.Draw(img_with_rectangles)
    imageEdit = ImageDraw.Draw(image)

    labels = list(labels)
    aCount = 0
    aList = []
    bCount = 0
    bList = []
    cCount = 0
    cList = []
    oneCount = 0
    oneList = []
    print("Starting to recognize objects")
    for i in range(min_y.size):
        difference_x = int(max_x[i] - min_x[i])
        difference_y = int(max_y[i] - min_y[i])
        fontsize = difference_y / 3
        if fontsize < 20: fontsize = 20
        if int(int(fontsize) / 6) < 1:
            rectangle_width = 1
        else:
            rectangle_width = int(int(fontsize) / 8)
        font = ImageFont.truetype(r'C:\Users\Sârius\Desktop\Junk\Pics\arial.ttf', int(fontsize))
        coloredEdit.rectangle([min_x[i], min_y[i], max_x[i], max_y[i]], outline="red", width=rectangle_width)
        imageEdit.rectangle([min_x[i], min_y[i], max_x[i], max_y[i]], outline="red", width=rectangle_width)

        temp = np.zeros((difference_y, difference_x), dtype=int)
        for j in range(difference_y):
            for k in range(difference_x):
                if arr_labeled_img[int(min_y[i]) + j][int(min_x[i]) + k] == labels[i]:
                    temp[j][k] = arr_labeled_img[int(min_y[i]) + j][int(min_x[i]) + k]
        temp = reverseBinarization(temp)

        labeled_arr, label_count, colors_blob, labels_list = blob_coloring_8_connected(temp, 255)
        if label_count == 1:
            coloredEdit.text((min_x[i], min_y[i] - int(fontsize)), "C", fill="red", font=font)
            imageEdit.text((min_x[i], min_y[i] - int(fontsize)), "C", fill="red", font=font)
            cList.append((min_x[i] , min_y[i] ))
            cList.append((max_x[i] , max_y[i] ))
            cCount += 1
        elif label_count == 2:
            coloredEdit.text((min_x[i], min_y[i] - int(fontsize)), "A", fill="red", font=font)
            imageEdit.text((min_x[i], min_y[i] - int(fontsize)), "A", fill="red", font=font)
            aList.append((min_x[i] , min_y[i]))
            aList.append((max_x[i] , max_y[i] ))
            aCount += 1
        elif label_count == 3:
            coloredEdit.text((min_x[i], min_y[i] - int(fontsize)), "B", fill="red", font=font)
            imageEdit.text((min_x[i], min_y[i] - int(fontsize)), "B", fill="red", font=font)
            bList.append((min_x[i], min_y[i]))
            bList.append((max_x[i], max_y[i]))
            bCount += 1
        else:
            coloredEdit.text((min_x[i], min_y[i] - int(fontsize)), "¿", fill="red", font=font)
            imageEdit.text((min_x[i], min_y[i] - int(fontsize)), "¿", fill="red", font=font)
            oneList.append((min_x[i], min_y[i]))
            oneList.append((max_x[i], max_y[i]))
            oneCount += 1
    img_with_rectangles.save(save_dir + "\\2Colored_result" + format)
    img_with_rectangles.thumbnail((800, 800))
    window["-IMAGE-"].update(data=ImageTk.PhotoImage(img_with_rectangles),size=(800,800))

    image.save(save_dir + "\\3RESULT" + format)
    image.thumbnail((800,800))
    window["-IMAGE-"].update(data=ImageTk.PhotoImage(image),size=(800,800))

    counts="A: " + str(aCount) + ", B: " + str(bCount) + ", C:" + str(cCount) + ", Unknown: " + str(oneCount)
    txtFile.write(counts)
    print(counts)
    txtFile.write("\nA: " + str(aCount))
    for i in aList:
        txtFile.write(", " + str(i))

    txtFile.write("\nB: " + str(bCount))
    for i in bList:
        txtFile.write(", " + str(i))

    txtFile.write("\nC: " + str(cCount))
    for i in cList:
        txtFile.write(", " + str(i))

    txtFile.write("\nunknown: " + str(oneCount))
    for i in oneList:
        txtFile.write(", " + str(i))

    txtFile.close()
    gui_queue.put('Image Recognition is finished!\nImages and data`s are saved to the directory file')


def reverseBinarization(array):
    x, y = array.shape
    for i in range(x):
        for j in range(y):
            if array[i][j] != 0:
                array[i][j] = 0
            else:
                array[i][j] = 255
    return array


def findObjects(num_labels, arr_labeled_img, labels):
    min_x = np.full(num_labels, np.inf)
    max_x = np.zeros(num_labels)
    min_y = np.full(num_labels, np.inf)
    max_y = np.zeros(num_labels)
    n_rows, n_cols = arr_labeled_img.shape

    for i in range(n_rows):
        for j in range(n_cols):  # go over image
            if arr_labeled_img[i][j] in labels:
                index = labels.index(arr_labeled_img[i][j])
                if min_x[index] > j - 3:
                    min_x[index] = j - 3
                if max_x[index] < j + 3:
                    max_x[index] = j + 3
                if min_y[index] > i - 3:
                    min_y[index] = i - 3
                if max_y[index] < i + 3:
                    max_y[index] = i + 3

    return min_x, min_y, max_x, max_y


def threshold(image, imArray, th):
    if image.format == "PNG":
        image = image.convert("RGBA")
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                r, g, b, a = image.getpixel((i, j))
                if int((r + g + b) / 3) < th and a > 100:
                    image.putpixel((i, j), (255, 255, 255, 255))
                    imArray[j][i] = 255
                else:
                    image.putpixel((i, j), (0, 0, 0, 255))
                    imArray[j][i] = 0

    else:
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                r, g, b = image.getpixel((i, j))
                temp = int((r + g + b) / 3)
                if temp < th:
                    image.putpixel((i, j), (255, 255, 255))
                    imArray[j][i] = 255
                else:
                    image.putpixel((i, j), (0, 0, 0))
                    imArray[j][i] = 0
    return image, imArray


def blob_coloring_8_connected(arr_bin, ONE):
    # get the numbers of rows and columns in the array of the binary image
    n_rows, n_cols = arr_bin.shape
    # max possible label value is set as 10000
    max_label = 10000
    # initially all the pixels in the image are labeled as max_label
    arr_labeled_img = np.zeros(shape=(n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            arr_labeled_img[i][j] = max_label
    # keep track of equivalent labels in an array
    # initially this array contains values from 0 to max_label - 1
    equivalent_labels = np.arange(max_label, dtype=int)
    # labeling starts with k = 1
    k = 1
    # first pass to assign initial labels and update equivalent labels from conflicts
    # for each pixel in the binary image
    # --------------------------------------------------------------------------------
    for i in range(1, n_rows - 1):
        for j in range(1, n_cols - 1):
            c = arr_bin[i][j]  # value of the current (center) pixel
            l = arr_bin[i][j - 1]  # value of the left pixel
            label_l = arr_labeled_img[i][j - 1]  # label of the left pixel
            u = arr_bin[i - 1][j]  # value of the upper pixel
            label_u = arr_labeled_img[i - 1][j]  # label of the upper pixel

            ul = arr_bin[i - 1][j - 1]
            label_ul = arr_labeled_img[i - 1][j - 1]
            r = arr_bin[i][j + 1]
            label_r = arr_labeled_img[i][j + 1]
            ur = arr_bin[i - 1][j + 1]
            label_ur = arr_labeled_img[i - 1][j + 1]

            # only the non-background pixels are labeled
            if c == ONE:
                # get the minimum of the labels of the upper and left pixels
                min_label = min(label_u, label_l, label_ul, label_r, label_ur)
                # if both upper and left pixels are background pixels
                if min_label == max_label:
                    # label the current (center) pixel with k and increase k by 1
                    arr_labeled_img[i][j] = k
                    k += 1
                # if at least one of upper and left pixels is not a background pixel
                else:
                    # label the current (center) pixel with min_label
                    arr_labeled_img[i][j] = min_label
                    # if upper pixel has a bigger label and it is not a background pixel
                    if min_label != label_u and label_u != max_label:
                        # update the array of equivalent labels for label_u
                        update_array(equivalent_labels, min_label, label_u)
                    # if left pixel has a bigger label and it is not a background pixel
                    if min_label != label_l and label_l != max_label:
                        # update the array of equivalent labels for label_l
                        update_array(equivalent_labels, min_label, label_l)
                    if min_label != label_ul and label_ul != max_label:
                        update_array(equivalent_labels, min_label, label_ul)
                    if min_label != label_r and label_r != max_label:
                        update_array(equivalent_labels, min_label, label_r)
                    if min_label != label_ur and label_ur != max_label:
                        update_array(equivalent_labels, min_label, label_ur)
    # final reduction in the array of equivalent labels to obtain the min. equivalent
    # label for each used label (values from 1 to k - 1) in the first pass of labeling
    # --------------------------------------------------------------------------------
    for i in range(1, k):
        index = i
        while equivalent_labels[index] != index:
            index = equivalent_labels[index]
        equivalent_labels[i] = equivalent_labels[index]
    # create a color map for randomly coloring connected components (blobs)
    # --------------------------------------------------------------------------------
    color_map = np.zeros(shape=(k, 3), dtype=np.uint8)
    np.random.seed(0)
    for i in range(k):
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)
    # create an array for the image to store randomly colored blobs
    arr_color_img = np.zeros(shape=(n_rows, n_cols, 3), dtype=np.uint8)
    # second pass to resolve labels by assigning the minimum equivalent label for each
    # label in arr_labeled_img and color connected components (blobs) randomly
    # --------------------------------------------------------------------------------
    for i in range(1, n_rows - 1):
        for j in range(1, n_cols - 1):
            # only the non-background pixels are taken into account
            if arr_bin[i][j] == ONE:
                test = equivalent_labels[arr_labeled_img[i][j]]
                arr_labeled_img[i][j] = test
                arr_color_img[i][j][0] = color_map[arr_labeled_img[i][j], 0]
                arr_color_img[i][j][1] = color_map[arr_labeled_img[i][j], 1]
                arr_color_img[i][j][2] = color_map[arr_labeled_img[i][j], 2]
            # change the label values of background pixels from max_label to 0
            else:
                arr_labeled_img[i][j] = 0
    # obtain the set of different values of the labels used to label the image
    different_labels = set(equivalent_labels[1:k])
    # compute the number of different values of the labels used to label the image
    num_different_labels = len(different_labels)
    # return the labeled image as a numpy array, number of different labels and the
    # image with colored blobs (components) as a numpy array
    different_labels = list(different_labels)
    return arr_labeled_img, num_different_labels, arr_color_img, sorted(different_labels)


# Function for updating the equivalent labels array by merging label1 and label2
# that are determined to be equivalent
def update_array(equ_labels, label1, label2):
    # determine the small and large labels between label1 and label2
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    # starting index is the large label
    index = lab_large
    # using an infinite while loop
    while True:
        # update the label of the currently indexed array element with lab_small when
        # it is bigger than lab_small
        if equ_labels[index] > lab_small:
            lab_large = equ_labels[index]
            equ_labels[index] = lab_small
            # continue the update operation from the newly encountered lab_large
            index = lab_large
        # update lab_small when a smaller label value is encountered
        elif equ_labels[index] < lab_small:
            lab_large = lab_small  # lab_small becomes the new lab_large
            lab_small = equ_labels[index]  # smaller value becomes the new lab_small
            # continue the update operation from the new value of lab_large
            index = lab_large
        # end the loop when the currently indexed array element is equal to lab_small
        else:  # equ_labels[index] == lab_small
            break


sg.theme('DefaultNoMoreNagging')
isChoosen = False
file_list_column = [
    [
        sg.Text("Image Folder", size=(10, 1), ),
        sg.In(size=(28, 1), enable_events=True, key="-FOLDER-")
    ],
    [
        sg.FileBrowse(button_text="Choose Image", enable_events=True, key="-BROWSE-", target="-FOLDER-", size=(20, 1),
                      pad=((116, 0), (5, 0)))
    ],
    [sg.Slider(range=(1, 255),
               default_value=150,
               size=(27, 15),
               orientation='horizontal',
               font=('Helvetica', 12), key="-THRESHOLD-", pad=(0, (10, 0)))

     ],
    [
        sg.Text(text="Threshold Value", pad=((100, 0), (0, 30)))
    ],
    [
        sg.Button(button_text="RUN!", key="-RUN-", bind_return_key=True, pad=((50, 0), 0)),
        sg.Button(button_text="Open Directory", key="directory", pad=((50, 0), 0))
    ],
    [
        sg.HSeparator(pad=(0, (5, 5)))
    ],
    [
        sg.Output(size=(40, 40), key="CONSOLE")
    ]
]

image_viewer_column = \
    [
    [sg.Image(key="-IMAGE-", size=(800, 800))]
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column)
    ]
]

gui_queue = queue.Queue()  # queue used to communicate between the gui and the threads
window = sg.Window("Image Viewer", layout)

# Create directory
dirName = 'Outputs'
dir_path = os.path.dirname(os.path.realpath(__file__))
save_dir = str(dir_path + "\\" + dirName)
completeName = os.path.join(save_dir, "Results.txt")
txtFile = open(completeName, "w")
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")

while True:
    event, values = window.read(timeout=150)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "-FOLDER-":
        filename = values["-BROWSE-"]
        try:
            pickedImage = Image.open(filename)
            pickedImage.thumbnail((800, 800))
            window["-IMAGE-"].update(data=ImageTk.PhotoImage(pickedImage),size=(800,800))
            filenameList = str(filename).split("/")
            print(filenameList[len(filenameList) - 1],"is choosen")
            isChoosen = True
        except:
            window["CONSOLE"].update("INVALID IMAGE TYPE")
    elif event == "-RUN-":
        if not isChoosen:
            window["CONSOLE"].update("Please choose an image first.")
        else:
            try:
                txtFile.write(filenameList[len(filenameList) - 1] + "\n")
                threading.Thread(target=main, args=(filename, values["-THRESHOLD-"], gui_queue), daemon=True).start()
            except:
                print("Some problem occured while Running\nRestarting program may help!")
    elif event == "directory":
        subprocess.Popen(r'explorer C:\Users\Sârius\PycharmProjects\comp204\{}'.format(dirName))
    try:
        message = gui_queue.get_nowait()
    except queue.Empty:  # get_nowait() will get exception when Queue is empty
        message = None  # break from the loop if no more messages are queued up

    # if message received from queue, display the message in the Window
    if message:
        print(message)

window.close()
