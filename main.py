import subprocess
subprocess.run('pip install -r requirements.txt', shell=True, capture_output=True)
from scripts.dupl_image_finder import *
import os
import string
import os.path
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv
import shutil
from PIL import Image, ExifTags


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception as e:
        pass
        # print(str(e))
    return s


def yolov5_integrity_check(images_folder="images/", labels_folder="labels/",
                           force_remove=False, verbose=True):
    nt = len(os.listdir(images_folder))  # total images
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    for image in tqdm(os.listdir(images_folder), desc=f"Yolov5 integrity check on {nt} images..."):
        base_name = os.path.splitext(os.path.basename(image))[0]
        im_file = f"{images_folder}{image}"
        lb_file = f"{labels_folder}{base_name}.txt"
        try:
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    assert f.read() == b'\xff\xd9', 'corrupted JPEG'

            # verify labels
            if os.path.isfile(lb_file):
                nf += 1  # label found
                with open(lb_file, 'r') as f:
                    line = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x) > 8 for x in line]):  # is segment
                        classes = np.array([x[0] for x in line], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in line]  # (cls, xy1...)
                        line = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    line = np.array(line, dtype=np.float32)
                if len(line):
                    assert line.shape[1] == 5, 'labels require 5 columns each'
                    assert (line >= 0).all(), 'negative labels'
                    assert (line[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(line, axis=0).shape[0] == line.shape[0], 'duplicate labels'
                else:
                    ne += 1  # label empty
                    if verbose:
                        print(f"\n{im_file} has a empty label file. Removing image and label if force_remove is True")
                    if force_remove:
                        os.remove(im_file)
                        os.remove(lb_file)
            else:
                nm += 1  # label missing
                if verbose:
                    print(f"\n{im_file} does not have a label file. Removing image if force_remove is True")
                if force_remove:
                    os.remove(im_file)

        except Exception as e:
            nc += 1
            if verbose:
                msg = f"\n[w pthreadpool-cpp.cc:90] WARNING: Ignoring corrupted image and/or label " \
                      f"{im_file}: {e}. Removing image and label if force_remove is True"
                print(msg)
            if force_remove:
                try:
                    os.remove(im_file)
                    os.remove(lb_file)
                except Exception as e:
                    print(e)
    if verbose:
        print(f"\nTotal images: {nt}, labels found: {nf}, labels missing: {nm}, labels empty: {ne}, corrupt: {nc}")


def empty_folder(fl):
    for file in tqdm(os.listdir(fl), desc=f"Emptying {fl}"):
        path = f"{fl}{file}"
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


def path_check(path, create=False):
    if not path[-1] == '/':
        path = f'{path}/'
    if create:
        if not os.path.isdir(path):
            os.makedirs(path)
    return path


def read_label(label_path):
    with open(label_path, 'r') as f:
        content = f.readlines()
    return content


def seg_line(line):
    new_line = line.split(' ')
    return new_line


def de_seg_line(line):
    new_line = ' '.join(map(str, line))
    return new_line


def write_label(lines, label_path):
    with open(label_path, 'w') as f:
        f.writelines(lines)


def count_instances(folder_path='labels/', save_to='csv/index_to_name.csv'):
    path = path_check(folder_path, create=False)
    index_csv = pd.read_csv('csv/index_to_name.csv')
    instance_csv = pd.DataFrame(columns=['Class', 'Name', 'Code', 'Instances', 'ImageCount'])
    instance_csv.Class = index_csv.Class
    instance_csv.Name = index_csv.Name
    instance_csv.Code = index_csv.Code
    instance_csv.fillna(0, inplace=True)
    for file in tqdm(os.listdir(path), desc='Counting the instances in your dataset.'):
        lines = read_label(f'{path}{file}')
        mute = np.zeros(8, dtype=bool)
        for line in lines:
            line = line.strip()
            segmented_line = seg_line(line)
            current_class = int(segmented_line[0])
            instance_csv.loc[instance_csv.Class == current_class, 'Instances'] += 1
            if not mute[current_class]:
                instance_csv.loc[instance_csv.Class == current_class, 'ImageCount'] += 1
                mute[current_class] = True
    instance_csv.to_csv(save_to, index=False)


def get_classes(image_id):
    lines = read_label(f'labels/{image_id}.txt')
    classes = []
    for line in lines:
        segmented_line = seg_line(line)
        classes.append(int(segmented_line[0]))
    return list(set(classes))


def make_image_csv(images_folder='images/', save_to='csv/img_list.csv'):
    img_folder = path_check(images_folder, create=False)
    img_list_csv = pd.DataFrame(columns=['ImageID', 'Width', 'Height', 'Size(MB)', 'Classes'])
    img_list = []
    for img in tqdm(os.listdir(img_folder), desc='Building image list.'):
        filepath = f'{img_folder}{img}'
        image = cv.imread(filepath)
        filename, file_extension = os.path.splitext(img)
        img_list.append(filename)
        image_size = float('%.3f' % float(os.stat(filepath).st_size*(9.537*10**-7)))
        classes = get_classes(filename)
        data = {'ImageID': filename, 'Width': image.shape[1], 'Height': image.shape[0], 'Size(MB)': image_size,
                'Classes': classes}
        img_list_csv = img_list_csv.append(data, ignore_index=True)
    img_list_csv.to_csv(save_to, index=False)
    print(f'{len(img_list_csv)} images found.')
    return img_list


def make_labels(classes, class_desc_csv_loc="csv/class-descriptions-boxable.csv",
                main_csv_loc="csv/main.csv", image_folder='images/', out_label_folder='labels/', verbose=False):

    class_names_list = classes
    for i in range(len(class_names_list)):
        class_names_list[i] = string.capwords(class_names_list[i])
    if verbose:
        print(f'\nClass names list is {class_names_list}')

    if len(os.listdir(out_label_folder)) > 0:
        empty_folder(out_label_folder)

    pbar = tqdm(total=5, desc='Reading csv. This can take some time')
    class_desc_csv = pd.read_csv(class_desc_csv_loc, header=None)
    class_desc_csv.columns = ['Code', 'Name']
    class_codes_list = class_desc_csv.loc[class_desc_csv['Name'].isin(class_names_list), 'Code'].tolist()
    if verbose:
        print(f'\nClass CODE list is {class_codes_list}')
    pbar.update()
    index_to_name_csv = pd.DataFrame(columns=['Class', 'Name', 'Code'])
    index_list = []
    for i in range(len(class_codes_list)):
        index_list.append(i)
    index_to_name_csv['Class'] = index_list
    index_to_name_csv['Code'] = class_codes_list
    pbar.update()
    for code in class_codes_list:
        name = class_desc_csv.loc[class_desc_csv['Code'] == code, 'Name']
        name = str(name.iloc[0])
        index_to_name_csv.loc[index_to_name_csv['Code'] == code, 'Name'] = name
    index_to_name_csv.to_csv('csv/index_to_name.csv', index=False)
    pbar.update()
    img_list = []
    for img in os.listdir(image_folder):
        filename, file_extension = os.path.splitext(img)
        img_list.append(filename)
    pbar.update()
    main_csv = pd.read_csv(main_csv_loc)
    main_csv = main_csv.loc[main_csv['LabelName'].isin(class_codes_list)]
    main_csv = main_csv.loc[main_csv['ImageID'].isin(img_list)]
    unique_names = main_csv.ImageID.unique()
    pbar.update()
    pbar.close()
    if verbose:
        print(f'\nFound {len(unique_names)} images in the main.csv that has a desired class in.')

    for name in tqdm(unique_names, desc='Creating Labels'):
        proc1 = main_csv.loc[main_csv['ImageID'] == name]
        if not proc1.empty:
            proc2 = proc1.loc[proc1['LabelName'].isin(class_codes_list)]
            if not proc2.empty:

                the_image = cv.imread(f'images/{name}.jpg')

                for i, row in enumerate(proc2.itertuples(), 1):
                    label_name = class_desc_csv.loc[
                        class_desc_csv['Code'] == row.LabelName, 'Name']
                    label_name = str(label_name.iloc[0])
                    label_class = index_to_name_csv.loc[index_to_name_csv['Name'] == label_name, 'Class']
                    label_class = int(label_class.iloc[0])

                    x = (row.XMin + row.XMax) / 2  # convert xmin xmax ymin ymax to xywh yolov5 format
                    y = (row.YMin + row.YMax) / 2
                    w = row.XMax - row.XMin
                    h = row.YMax - row.YMin

                    # min bbox width 10pixels min bbox height 13pixels
                    if not (round(the_image.shape[1] * w) < 10 or round(the_image.shape[0] * h) < 13):

                        wrtline = f"{label_class} {x} {y} {w} {h}\n"
                        if i == len(proc2):
                            wrtline = wrtline.strip()  # remove newline at end of last line

                        with open(f"{out_label_folder}{name}.txt", 'a') as f:
                            f.write(wrtline)


def get_name_for_class(cls, label_path):
    try:
        index_to_name_csv = pd.read_csv('csv/index_to_name.csv')
        name = index_to_name_csv.loc[index_to_name_csv['Class'] == int(cls), 'Name']
        name = str(name.iloc[0])
        return name
    except Exception as e:
        print(f'{cls} at {label_path} and e is {e}')
        exit()


def draw_box(image, label_filepath, normalized=True):
    lines = read_label(label_filepath)
    new = []
    for line in lines:
        new.append(line.split(' '))
    # .shape[0] = Height
    # .shape[1] = Width
    # .shape[2] = Colour Channels
    for line in new:
        if len(line) == 6:
            conf = float(line[5])
            conf = (round(conf * 100)) / 100
        else:
            conf = 0
        if normalized:
            x = float(line[1]) * image.shape[1]
            y = float(line[2]) * image.shape[0]
            w = float(line[3]) * image.shape[1]
            h = float(line[4]) * image.shape[0]
        else:
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])

        x1 = round(x - (w / 2))
        y1 = round(y - (h / 2))
        x2 = round(x + (w / 2))
        y2 = round(y + (h / 2))
        # b   g   r
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        font = cv.FONT_HERSHEY_SIMPLEX
        name = get_name_for_class(line[0], label_filepath)
        if conf == 0:
            image = cv.putText(image, name, (x1, y1), font, 0.75, (0, 0, 255), thickness=2)
        else:  # add conf to label if available
            image = cv.putText(image, f"{name} {conf}", (x1, y1), font, 0.75, (0, 0, 255), thickness=2)
    return image


def label_images(images_folder, labels_folder, labeled_images_folder):
    if not os.path.isdir(labeled_images_folder):
        os.mkdir(labeled_images_folder)

    labeled_images_folder = path_check(labeled_images_folder)
    if len(os.listdir(labeled_images_folder)) > 0:
        empty_folder(labeled_images_folder)
    for filename in tqdm(os.listdir(images_folder), desc='Labeling Images'):
        base_name = os.path.splitext(os.path.basename(filename))[0]

        image_filepath = f"{images_folder}{filename}"
        label_filepath = f"{labels_folder}{base_name}.txt"
        if os.path.isfile(label_filepath):
            img = cv.imread(image_filepath)
            img = draw_box(img, label_filepath)

            basedir = os.getcwd()
            os.chdir(labeled_images_folder)
            cv.imwrite(base_name + '.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            os.chdir(basedir)


def find(file):
    for root, dirs, files in os.walk('.'):
        if file in files:
            return os.path.join(root, file)


def merge_csv(type_data, verbose=False):
    merge_csv_list = get_applicable_csv(type_data=type_data)
    pbar = tqdm(total=len(merge_csv_list)+1, desc='Merging csv. This can take a while')
    csv_list = []
    for csv in merge_csv_list:
        if os.path.isfile(f'csv/{csv}'):
            if verbose:
                print(f'\nFound {csv}')
            csv = pd.read_csv(f'csv/{csv}')
            csv_list.append(csv)
        pbar.update()
    if len(csv_list) > 1:
        main_csv = pd.concat(csv_list, ignore_index=True)
    else:
        main_csv = csv_list[0]
    main_csv.to_csv('csv/main.csv')
    pbar.update()
    pbar.close()


def split(output_folder='split/', images_folder='images/', labels_folder='labels/', sp=[70, 20, 10]):
    labels_folder = path_check(labels_folder, create=False)
    images_folder = path_check(images_folder, create=False)
    output_folder = path_check(output_folder, create=False)

    folder_list = [f'{output_folder}train/images/', f'{output_folder}train/labels/',
                   f'{output_folder}valid/images/', f'{output_folder}valid/labels/',
                   f'{output_folder}test/images/', f'{output_folder}test/labels/']

    if not os.path.isdir(output_folder):
        for folder_path in folder_list:
            os.makedirs(folder_path)

    total_files = len(os.listdir(images_folder))

    for folder_path in folder_list:
        if len(os.listdir(folder_path)) > 0:
            empty_folder(folder_path)

    sec1 = sp[0]
    sec2 = sp[1]
    sec3 = sp[2]
    low = 0
    for file in tqdm(os.listdir(images_folder), desc=f'Splitting into {sp[0]}% train, {sp[1]}% valid,'
                                                     f'{sp[2]}% test,'):
        filename = file
        filepath = f"{images_folder}{filename}"
        base_name = os.path.splitext(os.path.basename(filename))[0]
        high = sec1 + sec2 + sec3
        p = random.uniform(low, high)
        if p <= sec1:
            if len(os.listdir(folder_list[0])) < ((total_files * sp[0]) / 100):
                shutil.copy(filepath, f'{folder_list[0]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[1]}{base_name}.txt')
            else:
                sec1 = 0

        if (p > sec1) & (p <= (sec1 + sec2)):
            if len(os.listdir(folder_list[2])) < ((total_files * sp[1]) * 100):
                shutil.copy(filepath, f'{folder_list[2]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[3]}{base_name}.txt')
            else:
                sec2 = 0

        if p > (sec1 + sec2):
            if len(os.listdir(folder_list[4])) < ((total_files * sp[2]) * 100):
                shutil.copy(filepath, f'{folder_list[4]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[5]}{base_name}.txt')
            else:
                sec3 = 0

    for i in range(0, 6, 2):
        yolov5_integrity_check(images_folder=folder_list[i], labels_folder=folder_list[i + 1], force_remove=True,
                               verbose=False)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scale_up=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im


def resize(images_folder, output_folder, image_size=640):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for im in tqdm(os.listdir(images_folder), desc=f'Resizing {len(os.listdir(images_folder))} images'
                                                   f' in {images_folder} to {image_size}'):
        img_data = cv.imread(f'{images_folder}{im}')
        img = letterbox(img_data, new_shape=image_size)
        cv.imwrite(f'{output_folder}{im}', img)


def setup(classes, verbose=False):

    class_names_list = classes
    if not os.path.isdir('labels'):
        os.mkdir('labels')
    if not os.path.isdir('images'):
        os.mkdir('images')

    if os.path.isdir('OIDv6'):
        if os.path.isdir('OIDv6/test'):
            for cls in os.listdir('OIDv6/test'):
                class_names_list.append(string.capwords(cls))
        if os.path.isdir('OIDv6/validation'):
            for cls in os.listdir('OIDv6/validation'):
                class_names_list.append(string.capwords(cls))
        if os.path.isdir('OIDv6/train'):
            for cls in os.listdir('OIDv6/train'):
                class_names_list.append(string.capwords(cls))
        class_names_list = list(set(class_names_list))

        if verbose:
            print(f'\nClass names list in setup is {class_names_list}')

        for cls in class_names_list:
            if os.path.isdir(f'OIDv6/test/{cls}'):
                for jpg in os.listdir(f'OIDv6/test/{cls}'):
                    name, ext = os.path.splitext(jpg)
                    path = f'OIDv6/test/{cls}/{jpg}'
                    if ext == '.jpg':
                        shutil.copy(path, f'images/{jpg}')

            if os.path.isdir(f'OIDv6/validation/{cls}'):
                for jpg in os.listdir(f'OIDv6/validation/{cls}'):
                    name, ext = os.path.splitext(jpg)
                    path = f'OIDv6/validation/{cls}/{jpg}'
                    if ext == '.jpg':
                        shutil.copy(path, f'images/{jpg}')

            if os.path.isdir(f'OIDv6/train/{cls}'):
                for jpg in os.listdir(f'OIDv6/train/{cls}'):
                    name, ext = os.path.splitext(jpg)
                    path = f'OIDv6/train/{cls}/{jpg}'
                    if ext == '.jpg':
                        shutil.copy(path, f'images/{jpg}')
    else:
        print('Please import images into the images folder.')
        exit()

    if not os.path.isdir('csv'):
        os.mkdir('csv')
    if os.path.isdir('OIDv6'):
        if os.path.isdir('OIDv6/metadata'):
            for csv in os.listdir('OIDv6/metadata'):
                shutil.copy(f'OIDv6/metadata/{csv}', f'csv/{csv}')
        if os.path.isdir('OIDv6/boxes'):
            for csv in os.listdir('OIDv6/boxes'):
                shutil.copy(f'OIDv6/boxes/{csv}', f'csv/{csv}')
    else:
        print('Please import class-descriptions-boxable.csv and train/test/validate csv into csv folder.')
        exit()
    if not os.path.isfile('csv/class-descriptions-boxable.csv'):
        print('Please import class-descriptions-boxable.csv and train/test/validate csv into csv folder.')
        exit()
    return class_names_list


def check_dataset(dataset):
    i = 0
    test_dataset = dataset
    while os.path.isdir(test_dataset):
        i += 1
        test_dataset = f'{dataset}{i}'
    dataset = test_dataset
    dataset = path_check(dataset, create=True)
    return dataset


def get_applicable_csv(type_data):
    main_csv_list = ['oidv6-train-annotations-bbox.csv', 'validation-annotations-bbox.csv', 'test-annotations-bbox.csv']
    test_csv_list = []
    if type_data == 'all':
        test_csv_list = main_csv_list
    elif type_data == 'train':
        test_csv_list.append(main_csv_list[0])
    elif type_data == 'validation':
        test_csv_list.append(main_csv_list[1])
    elif type_data == 'test':
        test_csv_list.append(main_csv_list[2])
    else:
        print(f'--type_data: invalid choice: {type_data}, (choose from train, validation, test, all)')
        exit()
    return test_csv_list


def main(opt):
    print(colorstr('Yolov5 Dataset Builder: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


def run(classes,
        dataset='dataset',
        dataset_exists=False,
        duplicates=False,
        label_img=False,
        split_data=False,
        resize_img=0,
        verbose=False,
        download=False,
        type_data='train',
        limit=0):

    if not dataset_exists:
        dataset = check_dataset(dataset)
    base_dir = os.getcwd()

    csv_search_list = get_applicable_csv(type_data=type_data)
    for csv in csv_search_list:
        found_loc = find(csv)
        if found_loc:
            if not os.path.isdir(os.path.join(dataset, 'OIDv6/boxes')):
                os.makedirs(os.path.join(dataset, 'OIDv6/boxes'))
            shutil.copy(found_loc, os.path.join(dataset, 'OIDv6/boxes'))

    annotations_search_list = ['class-descriptions-boxable.csv']
    for csv in annotations_search_list:
        found_loc = find(csv)
        if found_loc:
            if not os.path.isdir(os.path.join(dataset, 'OIDv6/metadata')):
                os.makedirs(os.path.join(dataset, 'OIDv6/metadata'))
            shutil.copy(found_loc, os.path.join(dataset, 'OIDv6/metadata'))

    os.chdir(dataset)

    classes = ' '.join(classes)
    print(f'Currently working in {os.getcwd()}')

    if download:
        oidv6_string = f'oidv6 downloader --yes --no_clear_shell en' \
                       f' --type_data {type_data} --classes {classes} --limit {limit}'
        subprocess.run(oidv6_string, shell=True)

    classes = classes.split()
    classes = setup(classes=classes, verbose=verbose)

    merge_csv(type_data=type_data, verbose=verbose)
    if duplicates:
        compare_images('images/', show_imgs=False, similarity="High", rotate=False, force_remove=True, verbose=verbose)
    make_labels(classes=classes, verbose=verbose)
    if label_img:
        label_images(images_folder='images/', labels_folder='labels/', labeled_images_folder='labeled_images/')
    count_instances()
    yolov5_integrity_check(force_remove=True, verbose=verbose)
    if resize_img != 0:
        split_data = True
    if split_data:
        split()
        if resize_img != 0:
            split_list = ['split/train/images/', 'split/valid/images/', 'split/test/images/']
            for folder_path in split_list:
                resize(images_folder=folder_path, output_folder=folder_path, image_size=resize_img)
    os.chdir(base_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset', help='Name of the dataset.')
    parser.add_argument('--dataset_exists', action='store_true', help='Save images to an existing dataset.')
    parser.add_argument('--duplicates', action='store_true', help='Check for duplicate images. Can be slow on big data.')
    parser.add_argument('--label_img', action='store_true', help='Create a separate folder and label the images.')
    parser.add_argument('--split_data', action='store_true', help='Split dataset into train, valid, test split.')
    parser.add_argument('--resize_img', type=int, default=0, help='Size in px to resize images to.')
    parser.add_argument('--verbose', action='store_true', help='Print out more info on every step.')
    parser.add_argument('--download', action='store_true', help='Use the OIDv6 downloader to download images.')
    parser.add_argument('--type_data', type=str, default='train', help='Download from subset [train, validation, test, all].')
    parser.add_argument('--limit', default=0, type=int, help='Limit the amount of images to download.')
    parser.add_argument('--classes', nargs='+', type=str, help='Path to classes.txt or list classes like "Cat Dog Ant".', required=True)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    os.chdir('.')
    opt = parse_opt()
    main(opt)
