from tqdm import tqdm
import os
import os.path
import pandas as pd
import numpy as np
import cv2 as cv
import shutil
from PIL import Image, ExifTags
import math
import random

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
    except:
        pass
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
            segments = []  # instance segments
            if os.path.isfile(lb_file):
                nf += 1  # label found
                with open(lb_file, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x) > 8 for x in l]):  # is segment
                        classes = np.array([x[0] for x in l], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                        l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 5, 'labels require 5 columns each'
                    assert (l >= 0).all(), 'negative labels'
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                else:
                    ne += 1  # label empty
                    if verbose:
                        print(f"\n{im_file} has a empty label file. Removing image and label if force_remove is True")
                    if force_remove:
                        os.remove(im_file)
                        os.remove(lb_file)
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm += 1  # label missing
                if verbose:
                    print(f"\n{im_file} does not have a label file. Removing image if force_remove is True")
                if force_remove:
                    os.remove(im_file)

                l = np.zeros((0, 5), dtype=np.float32)
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
    print(f"\nTotal images: {nt}, labels found: {nf}, labels missing: {nm}, labels empty: {ne}, corrupt: {nc}")


def empty_folder(folder):
    for file in tqdm(os.listdir(folder), desc=f"Emptying {folder}..."):
        path = f"{folder}{file}"
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


def get_name_for_class(cls, label_path):
    try:
        # if cls == 15:
        # print(f'{cls} at {label_path}')
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
    labeled_images_folder = path_check(labeled_images_folder)
    empty_folder(labeled_images_folder)
    for filename in tqdm(os.listdir(images_folder), desc='Labeling Images...'):
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

def get_classes(ImageID):
    lines = read_label(f'labels/{ImageID}.txt')
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
        image_size = float('%.3f' % float((os.stat(filepath).st_size)*(9.537*10**-7)))
        classes = get_classes(filename)
        data = {'ImageID': filename, 'Width': image.shape[1], 'Height': image.shape[0], 'Size(MB)': image_size,
                'Classes': classes}
        img_list_csv = img_list_csv.append(data, ignore_index=True)
    img_list_csv.to_csv(save_to, index=False)
    print(f'{len(img_list_csv)} images found.')
    return img_list


def make_labels(class_desc_csv_loc="csv/class-descriptions-boxable.csv",
                main_csv_loc="csv/main.csv"):
    class_list = ['/m/01g317', '/m/012w5l', '/m/0k4j', '/m/0199g', '/m/01bms0', '/m/01d380', '/m/01j4z9', '/m/01j5ks',
                  '/m/01lsmm', '/m/03l9g', '/m/04ctx', '/m/06c54', '/m/07r04', '/m/0c2jj', '/m/0_dqb', '/m/0gxl3',
                  '/m/0h8jyh6']

    out_label_folder = 'labels/'
    image_folder = 'images/'
    empty_folder(out_label_folder)

    img_list = make_image_csv(image_folder)

    print('Reading csv. This can take some time...')
    class_desc_csv = pd.read_csv(class_desc_csv_loc)
    class_desc_with_column_csv = class_desc_csv
    class_desc_with_column_csv.columns = ['Code', 'Name']

    index_to_name_csv = pd.DataFrame(columns=['Class', 'Name', 'Code'])
    index_list = []
    for i in range(len(class_list)):
        index_list.append(i)
    index_to_name_csv['Class'] = index_list
    index_to_name_csv['Code'] = class_list

    for code in class_list:
        Name = class_desc_with_column_csv.loc[class_desc_with_column_csv['Code'] == code, 'Name']
        Name = str(Name.iloc[0])
        index_to_name_csv.loc[index_to_name_csv['Code'] == code, 'Name'] = Name

    index_to_name_csv.to_csv('csv/index_to_name.csv', index=False)

    main_csv = pd.read_csv(main_csv_loc)
    main_csv = main_csv.loc[main_csv['LabelName'].isin(class_list)]
    main_csv = main_csv.loc[main_csv['ImageID'].isin(img_list)]

    unique_names = main_csv.ImageID.unique()

    print(f'Found {len(unique_names)} images in the main.csv that has a desired class in.')

    for name in tqdm(unique_names, desc='Creating Labels...'):
        proc1 = main_csv.loc[main_csv['ImageID'] == name]
        if not proc1.empty:
            proc2 = proc1.loc[proc1['LabelName'].isin(class_list)]
            if not proc2.empty:

                DieFoto = cv.imread(f'images/{name}.jpg')

                for i, row in enumerate(proc2.itertuples(), 1):
                    label_name = class_desc_with_column_csv.loc[
                        class_desc_with_column_csv['Code'] == row.LabelName, 'Name']
                    label_name = str(label_name.iloc[0])
                    label_class = index_to_name_csv.loc[index_to_name_csv['Name'] == label_name, 'Class']
                    label_class = int(label_class.iloc[0])

                    x = (row.XMin + row.XMax) / 2  # convert xmin xmax ymin ymax to xywh yolov5 format
                    y = (row.YMin + row.YMax) / 2
                    w = row.XMax - row.XMin
                    h = row.YMax - row.YMin

                    # min bbox width 10pixels min bbox height 13pixels
                    if not (round(DieFoto.shape[1] * w) < 10 or round(DieFoto.shape[0] * h) < 13):

                        wrtline = f"{label_class} {x} {y} {w} {h}\n"
                        if i == len(proc2):
                            wrtline = wrtline.strip()  # remove newline at end of last line

                        with open(f"{out_label_folder}{name}.txt", 'a') as f:
                            f.write(wrtline)


def merge_csv():
    print('Merging csv. This can take a while...')
    train_csv = pd.read_csv("csv/train-annotations-bbox.csv")
    val_csv = pd.read_csv("csv/validation-annotations-bbox.csv")
    test_csv = pd.read_csv("csv/test-annotations-bbox.csv")

    main_csv = pd.concat([train_csv, val_csv, test_csv], ignore_index=True)
    main_csv.to_csv('csv/main.csv')


def setup():
    os.chdir('.')
    print(f'Currently working in {os.getcwd()}')
    if not os.path.isdir('labels'):
        os.mkdir('labels')
    if not os.path.isdir('labeled_images'):
        os.mkdir('labeled_images')
    if not os.path.isdir('images'):
        os.mkdir('images')
        print('Please import images into the images folder.')
        exit()
    if not os.path.isdir('csv'):
        os.mkdir('csv')
        print('Please import class-descriptions-boxable.csv and train/test/validate csv into csv folder.')
        exit()
    if not os.path.isfile('csv/class-descriptions-boxable.csv'):
        print('Please import class-descriptions-boxable.csv and train/test/validate csv into csv folder.')
        exit()


def count_instances(folder_path='labels/', save_to='csv/index_to_name.csv', csv=None):
    path = path_check(folder_path, create=False)
    index_csv = pd.read_csv('csv/index_to_name.csv')
    instance_csv = pd.DataFrame(columns=['Class', 'Name', 'Code', 'Instances', 'ImageCount'])
    instance_csv.Class = index_csv.Class
    instance_csv.Name = index_csv.Name
    instance_csv.Code = index_csv.Code
    instance_csv.fillna(0, inplace=True)
    if csv.empty:
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
    else:
        search_list = csv.ImageID.tolist()
        for file in tqdm(search_list, desc='Counting the instances in your dataset.'):
            lines = read_label(f'labels/{file}.txt')
            mute = np.zeros(8, dtype=bool)
            for line in lines:
                line = line.strip()
                segmented_line = seg_line(line)
                current_class = int(segmented_line[0])
                instance_csv.loc[instance_csv.Class == current_class, 'Instances'] += 1
                if not mute[current_class]:
                    instance_csv.loc[instance_csv.Class == current_class, 'ImageCount'] += 1
                    mute[current_class] = True
        #instance_csv.to_csv(save_to, index=False)
        return instance_csv

def merge_classes(classes_list, new_class_name, label=False):
    index_to_name_csv = pd.read_csv('csv/index_to_name.csv')

    cut_df = index_to_name_csv[~index_to_name_csv.Name.isin(classes_list)]
    out_df = pd.DataFrame(cut_df)

    work_df = index_to_name_csv[index_to_name_csv.Name.isin(classes_list)]
    old_codes_list = work_df.Code.tolist()
    class_list = work_df.Class.tolist()
    new_code = class_list[0]

    new_entry = pd.Series({'Class': new_code, 'Name': new_class_name, 'Code': old_codes_list})
    new_df = out_df.append(new_entry, ignore_index=True)
    new_df.to_csv('csv/index_to_name.csv', index=False)

    for file in os.listdir('labels/'):
        filepath = f'labels/{file}'
        lines = read_label(filepath)
        new_lines = []
        for line in lines:
            line = line.split(' ')
            cls = int(line[0])
            if cls in class_list:
                line[0] = new_code
            wrtline = ' '.join(map(str, line))
            new_lines.append(wrtline)
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    instances_csv = pd.read_csv('csv/instances.csv')

    cut_df = instances_csv[~instances_csv.LabelName.isin(old_codes_list)]

    out_df = pd.DataFrame(cut_df)

    work_df = instances_csv[instances_csv.LabelName.isin(old_codes_list)]
    instances_sum = work_df.Instances.sum()
    image_count_sum = work_df.ImageCount.sum()

    new_entry = pd.Series({'LabelName': old_codes_list, 'FriendlyName': new_class_name, 'Instances': instances_sum,
                           'ImageCount': image_count_sum})
    new_df = out_df.append(new_entry, ignore_index=True)
    new_df.to_csv('csv/instances.csv', index=False)

    if label:
        label_images(images_folder='images/', labels_folder='labels/', labeled_images_folder='labeled_images/')


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


def remap(prev_cls, new_cls, folder=None, csv_path=None, label=False):
    if csv_path:
        csv = pd.read_csv(csv_path)
        csv.loc[csv.Class == prev_cls, 'Class'] = new_cls
        csv.sort_values(by='Class', inplace=True)
        csv.to_csv(csv_path, index=False)
    if folder:
        folder = path_check(folder)
        for file in os.listdir(folder):
            label_path = f'{folder}{file}'
            lines = read_label(label_path)
            wrtlines = []
            for line in lines:
                segmented_line = seg_line(line)
                if int(segmented_line[0]) == prev_cls:
                    segmented_line[0] = new_cls
                wrtlines.append(de_seg_line(segmented_line))
            write_label(wrtlines, label_path)
    if label:
        label_images(images_folder='images/', labels_folder='labels/', labeled_images_folder='labeled_images/')


def merge_labels(master_folder, slave_folder, out_folder):

    out_folder = path_check(out_folder)
    master_folder = path_check(master_folder)
    slave_folder = path_check(slave_folder)

    if out_folder == master_folder or out_folder == slave_folder:
        inplace = True
    else:
        inplace = False

    if not inplace:
        empty_folder(out_folder)


    index = list(set(os.listdir(master_folder) + os.listdir(slave_folder)))
    for file in index:
        present_index = []
        new_label = []

        if os.path.isfile(f'{master_folder}{file}'):
            master_content = read_label(f'{master_folder}{file}')
            for line in master_content:
                line = line.strip()
                segmented_line = seg_line(line)
                present_class = int(segmented_line[0])
                present_index.append(present_class)  # Create a index of the classes present in the new label
                new_label.append(f'{line}\n')

        if os.path.isfile(f'{slave_folder}{file}'):
            slave_content = read_label(f'{slave_folder}{file}')
            for line in slave_content:
                segmented_line = seg_line(line)
                present_class = int(segmented_line[0])
                if not (present_class in present_index):  # If the class is not in the master folder then add it.
                    new_label.append(line)
                else:  # todo implement IOU less than 0.45 to add
                    # IF there is a object of example a person already in the image from the master folder then we don't want to add
                    # that same bounding box from the slave folder. So we loop through all the entries already in the new label
                    # and if there is a entry with the same CLASS as the object I am working with than calc the distance between the
                    # midpoints of the bboxes. If that distance is more than 0.05 (normalized distance) then it is probably a new instance
                    # of that object and we should add it. Better would be to use IOU less than 0.45
                    already_present = False
                    for entry in new_label:
                        if not already_present:  # If the bbox in question is not already on the image...
                            seg_entry = seg_line(entry)  # segment the line
                            if int(seg_entry[0]) == present_class:  # the first char of each line has the class id
                                # root((x2-x1)^2+(y2-y1)^2)
                                distance = math.sqrt((float(segmented_line[1]) - float(seg_entry[1])) ** 2 + (
                                        float(segmented_line[2]) - float(seg_entry[2])) ** 2)
                                if distance < 0.075:
                                    already_present = True
                    if not already_present:
                        new_label.append(line)

        new_label[-1] = new_label[-1].strip()  # the last line in the label should not have a newline at the end
        write_label(new_label, f'{out_folder}{file}')


def classes_merge_loop():
    merge_classes(['Rifle', 'Handgun'], 'Weapon')
    merge_classes(['Drill', 'Chainsaw', 'Wrench', 'Scissors', 'Hammer', 'Screwdriver', 'Axe', 'Chisel', 'Grinder']
                  , 'Tool')


def remap_loop():
    remap(4, 6, folder='labels/', csv_path='csv/index_to_name.csv')
    remap(1, 7, folder='labels/', csv_path='csv/index_to_name.csv')
    remap(3, 1, folder='labels/', csv_path='csv/index_to_name.csv')
    remap(12, 3, folder='labels/', csv_path='csv/index_to_name.csv')
    remap(11, 4, folder='labels/', csv_path='csv/index_to_name.csv')
    remap(10, 5, folder='labels/', csv_path='csv/index_to_name.csv')


def xywh2xyxy(x):
    new_label = []
    new_label.append(x[0] - (x[2] / 2))
    new_label.append(x[1] - (x[3] / 2))
    new_label.append(x[0] + (x[2] / 2))
    new_label.append(x[1] + (x[3] / 2))
    return new_label


def split(output_folder, images_folder, labels_folder, split = [70, 20, 10]):
    labels_folder = path_check(labels_folder, create=False)
    images_folder = path_check(images_folder, create=False)
    output_folder = path_check(output_folder, create=False)

    folder_list = [f'{output_folder}train/images/', f'{output_folder}train/labels/',
                   f'{output_folder}valid/images/', f'{output_folder}valid/labels/',
                   f'{output_folder}test/images/', f'{output_folder}test/labels/']

    if not os.path.isdir('trim_split'):
        for folder in folder_list:
            os.makedirs(folder)

    total_files = len(os.listdir(images_folder))

    for folder in folder_list:
        empty_folder(folder)

    sec1 = split[0]
    sec2 = split[1]
    sec3 = split[2]
    low = 0
    for file in tqdm(os.listdir(images_folder), desc=f'Splitting into {split[0]}% train, {split[1]}% valid,'
                                                     f'{split[2]}% test,'):
        filename = file
        filepath = f"{images_folder}{filename}"
        base_name = os.path.splitext(os.path.basename(filename))[0]
        high = sec1 + sec2 + sec3
        p = random.uniform(low, high)
        if p <= sec1:
            if len(os.listdir(folder_list[0])) < ((total_files * split[0])/100):
                shutil.copy(filepath, f'{folder_list[0]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[1]}{base_name}.txt')
            else:
                sec1 = 0

        if (p > sec1) & (p <= (sec1 + sec2)):
            if len(os.listdir(folder_list[2])) < ((total_files * split[1])*100):
                shutil.copy(filepath, f'{folder_list[2]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[3]}{base_name}.txt')
            else:
                sec2 = 0

        if p > (sec1 + sec2):
            if len(os.listdir(folder_list[4])) < ((total_files * split[2])*100):
                shutil.copy(filepath, f'{folder_list[4]}{filename}')
                shutil.copy(f'{labels_folder}{base_name}.txt', f'{folder_list[5]}{base_name}.txt')
            else:
                sec3 = 0

    for i in range(0, 6, 2):
        yolov5_integrity_check(images_folder=folder_list[i], labels_folder=folder_list[i + 1], force_remove=False,
                               verbose=True)

def Main():
    if not os.path.isfile('csv/main.csv'):
        merge_csv()
    make_labels()
    classes_merge_loop()
    remap_loop()
    merge_labels('labels/', 'other_labels/', 'labels/')
    # copy in weapon dataset
    yolov5_integrity_check(force_remove=False, verbose=True)
    label_images(images_folder='images/', labels_folder='labels/', labeled_images_folder='labeled_images/')
    count_instances()


if __name__ == '__main__':
    IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    setup()
    answer = input(
        f"\nMain <0> \nMerge CSV <1> \nMake Labels <2> \nConvert cls-xyxy to cls-xywh <3> \n"
        f"Label images (Yolo formatted) <4> \nyolov5 integrity check <5> \n")
    print(f"\nYou picked option {answer}")

    if answer == '0':  # run the full program
        Main()

    if answer == '1':  # Make main.csv
        merge_csv()

    if answer == '2':  # Make labels
        make_labels()

    if answer == '3':  # yolov5 integrity check
        yolov5_integrity_check()

    if answer == '4':  # Label images
        label_images(images_folder='images/', labels_folder='labels/',
                     labeled_images_folder='labeled_images/')

    if answer == '5':  #

        master_csv = pd.read_csv('csv/img_list.csv')
        master_csv = master_csv.loc[(master_csv.Width <= 1280) & (master_csv.Height <= 1280) &
                                    (master_csv.Width >= 480) & (master_csv.Height >= 480) &
                                    (master_csv['Size(MB)'] <= 1.0)]

        #test_instances_csv = count_instances(csv=master_csv)

        only_humans_or_car_csv = master_csv.loc[(master_csv.Classes == '[0]') | (master_csv.Classes == '[2]')]
        other_csv = master_csv.loc[(master_csv.Classes != '[0]') & (master_csv.Classes != '[2]')]


        only_humans_or_car_csv = only_humans_or_car_csv.loc[(only_humans_or_car_csv.Width >= 640)
                                                            & (only_humans_or_car_csv.Height >= 640)]

        only_humans_or_car_csv.sort_values(by='Size(MB)', inplace=True)
        only_humans_or_car_csv = only_humans_or_car_csv.iloc[:-4826]


        #from scipy import stats
        #z_scores = stats.zscore(master_csv[['Width', 'Height', 'Size(MB)']])  # trim 3 std deviations away
        #abs_z_scores = np.abs(z_scores)
        #filtered_entries = (abs_z_scores < 3).all(axis=1)
        #trim_csv = master_csv[filtered_entries]

        master_csv = other_csv.append(only_humans_or_car_csv)

        len = len(master_csv)
        print(len)
        master_instances_csv = count_instances(csv=master_csv)
        w = master_csv.sort_values(by='Width', ascending=False)
        h = master_csv.sort_values(by='Height', ascending=False)
        s = master_csv.sort_values(by='Size(MB)', ascending=False)
        print(master_csv.head)
        master_csv.to_csv('csv/trimmed.csv', index=False)
        master_instances_csv.to_csv('csv/trim_main_instances.csv', index=False)

    if answer == '6':

        trim_csv = pd.read_csv('csv/trimmed.csv')
        ids = trim_csv.ImageID.tolist()

        len = len(trim_csv.ImageID.unique())


        for id in ids:
            filename = f'{id}.jpg'
            filepath = f"images/{filename}"
            base_name = id

            shutil.copy(filepath, f'trimmed_images/{filename}')
            shutil.copy(f'labels/{base_name}.txt', f'trimmed_labels/{base_name}.txt')
        print('')

    if answer == '7':
        yolov5_integrity_check(force_remove=True, verbose=True)
        label_images(images_folder='images/', labels_folder='labels/', labeled_images_folder='labeled_images/')

    if answer == '8':
        yolov5_integrity_check(images_folder='trimmed_images/', labels_folder='trimmed_labels/', force_remove=False, verbose=True)


    if answer == '9':
        pass


    if answer == '10':
        split(output_folder = 'trim_split/', images_folder = 'trimmed_images/', labels_folder = 'trimmed_labels/',
              split = [80, 20, 0])