import cv2 as cv
import shutil
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm
from PIL import Image, ExifTags
import torch
from pathlib import Path

def read_label(label_path):
    with open(label_path) as l:
        content = l.readlines()
    return content


def rescale_frame(frame, x=640, y=640):
    dimensions = (x, y)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


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
        if conf == 0:
            image = cv.putText(image, line[0], (x1, y1), font, 0.75, (0, 0, 255), thickness=2)
        else:  # add conf to label if available
            image = cv.putText(image, f"{line[0]} {conf}", (x1, y1), font, 0.75, (0, 0, 255), thickness=2)
    return image


def empty_folder(folder):
    for file in tqdm(os.listdir(folder), desc=f"Emptying {folder}..."):
        path = f"{folder}{file}"
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


def integrity_check(images_folder, labels_folder):
    for filename in tqdm(os.listdir(images_folder), desc='Converting all images to .jpg'):
        filepath = images_folder + filename
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        extension = os.path.splitext(os.path.basename(filepath))[1]
        if extension != '.jpg':
            img = cv.imread(filepath)
            os.remove(filepath)
            cv.imwrite(f"{images_folder}{base_name}.jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    print('\nChecking if there is a label for every image')  # Check if there is a label for every image
    for filename in tqdm(os.listdir(images_folder)):
        base_name = os.path.splitext(os.path.basename(filename))[0]

        image_filepath = f"{images_folder}{filename}"
        label_filepath = f"{labels_folder}{base_name}.txt"

        try:
            with open(label_filepath) as f:
                lines = f.readlines()

                if len(lines) < 1:  # empty file
                    print(
                        f"\nFile -{labels_folder}{base_name}.txt- is empty."
                        f" Deleting this file and image at {image_filepath}")
                    os.remove(image_filepath)
                    os.remove(label_filepath)

                if len(np.unique(lines)) < len(lines):  # duplicate rows todo NEED SOME FIXING LOOK AT 010000096.txt
                    print(
                        f"\nFile -{label_filepath}- has duplicate lines."
                        f" Deleting this file and image at {image_filepath}")
                    os.remove(image_filepath)
                    os.remove(label_filepath)
        except Exception as e:
            print(f"\nFile -{label_filepath}- not found. Deleting image at {image_filepath}. Exception is {e}")
            os.remove(image_filepath)

    print('\nChecking if there is a image for every label')  # Check if there is a image for every label
    for filename in tqdm(os.listdir(labels_folder)):
        base_name = os.path.splitext(os.path.basename(filename))[0]

        image_filepath = f"{images_folder}{base_name}.jpg"
        label_filepath = f"{labels_folder}{filename}"

        if not os.path.isfile(image_filepath):
            print(f"\nImage at -{image_filepath}- not found. Deleting label at -{label_filepath}")
            os.remove(label_filepath)

    print('\nChecking label format')
    for filename in tqdm(os.listdir(labels_folder)):  # remove space at the end of a line
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        filepath = f"{labels_folder}{filename}"
        
        with open(filepath, 'r+') as f:
            lines = f.readlines()
            new_lines = []
            i = 1
            error_found = 0
            for line in lines:
                if line.endswith(" \n"):
                    error_found = 1
                    if i == len(lines):
                        line = line.replace(' \n', '')
                    else:
                        line = line.replace(' \n', '\n')
                    print(f"\nFound a double space in {filename} and fixed it")
                    new_lines.append(line)
                else:
                    new_lines.append(line)

            if error_found == 1:
                os.remove(filepath)
                with open(filepath, 'a+') as g:  # write the correct file
                    for line in new_lines:
                        g.write(line)

            for test_line in new_lines:  # check if the label is correct and if not then delete image and label
                test_line = test_line.split(' ')
                for item in test_line:
                    try:
                        if float(item) > 1:
                            print(f"\nError in {filepath}")
                            print(f"\nLine is {test_line}")
                            print(f"\nDeleted both the label file and image file")
                            os.remove(filepath)
                            os.remove(f"{images_folder}{base_name}.jpg")
                    except Exception as e:
                        print(f"\nCan't fix {filepath} the item is {item} and test_line is {test_line}."
                              f" Exception is {e}")
                        os.remove(filepath)
                        os.remove(f"{images_folder}{base_name}.jpeg")


def label_images(images_folder, labels_folder, labeled_images_folder):

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


def rename(images_folder, labels_folder, name):
    i = 0
    for filename in os.listdir(images_folder):
        filepath = images_folder + filename
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        extension = os.path.splitext(os.path.basename(filepath))[1]

        basedir = os.getcwd()  # rename both the image and label
        os.chdir(images_folder)
        os.rename(filename, f"{name}_{i}{extension}")
        os.chdir(basedir)
        os.chdir(labels_folder)
        os.rename(f"{base_name}.txt", f"{name}_{i}.txt")
        os.chdir(basedir)
        i += 1


def check_resolution(folder, w_threshold=320, h_threshold=180, less_more='less',
                     show=False, rescale_save=False, delete=False):
    # or_and = 'or'  todo implement or_and
    t0 = time.time()
    count = 0
    found = False
    print(f"\nChecking image resolution...")
    for image in tqdm(os.listdir(folder)):
        # base_name = os.path.splitext(os.path.basename(image))[0]
        image_filepath = f"{folder}{image}"
        # label_filepath = f"More weapons/Labels/{base_name}.txt"

        img = cv.imread(image_filepath)
        w_scale = img.shape[1]
        h_scale = img.shape[0]
        if less_more == 'less':
            if img.shape[1] < w_threshold or img.shape[0] < h_threshold:
                found = True

                if img.shape[1] < w_threshold:
                    w_scale = w_threshold
                else:
                    w_scale = img.shape[1]

                if img.shape[0] < h_threshold:
                    h_scale = h_threshold
                else:
                    h_scale = img.shape[0]

        else:
            if img.shape[1] > w_threshold or img.shape[0] > h_threshold:
                found = True

                if img.shape[1] > w_threshold:
                    w_scale = w_threshold
                else:
                    w_scale = img.shape[1]

                if img.shape[0] > h_threshold:
                    h_scale = h_threshold
                else:
                    h_scale = img.shape[0]

        if found:
            count += 1
            found = False
            if show:
                cv.imshow(f"{image} has Width: {img.shape[1]} Height: {img.shape[0]}", img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                # if count == 10:
                #     break

            if rescale_save:
                img = rescale_frame(img, w_scale, h_scale)
                cv.imwrite(image_filepath, img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

            if delete:
                os.remove(image_filepath)

    t1 = time.time()
    print(f"Found {count} images in {t1 - t0} sec.")


def plot_distribution(target_folder):
    dist = []
    for label in tqdm(os.listdir(target_folder)):
        label_filepath = f"{target_folder}{label}"
        lines = read_label(label_filepath)
        split = []
        for line in lines:
            split.append(line.split(' '))
        for part in split:
            dist.append(part[5])
    sns.set_style('white')
    dist = np.array(dist)
    dist = dist.astype(np.float)
    print(len(dist))

    count, bins_count = np.histogram(dist, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    plt.plot(bins_count[1:] * 100 - 8, pdf / 10, color="red", label="PDF")
    plt.plot(bins_count[1:] * 100 - 8, cdf / 10, label="CDF")
    plt.legend()

    dist = dist * 100
    sns.distplot(dist, kde_kws={"color": "lime"}, fit=skewnorm, bins=90)
    xt = []
    for i in range(0, 102, 2):
        xt.append(i)
    plt.xticks(xt)
    plt.ylim(0, 0.02)
    plt.xlim(30, 70)
    for xc in range(101):
        plt.axvline(x=xc, color='black', alpha=0.1)
    plt.xlabel('% Certainty')
    plt.show()


def video_to_frames(video_path = f"data/videos/Hunting Trespasser gets Painted.mp4",
                    video_labels = f"data/videos/video_labels/", video_frames = f"data/videos/video_frames/",
                    labeled_video_frames = f"More weapons/videos/labeled_video_frames/", show = False,
                    check = True, label = True):
    empty_folder(video_frames)
    frame_n = 1
    cap = cv.VideoCapture(video_path)
    max_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        print(f"Processing frame {frame_n}/{max_frames}")
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        basedir = os.getcwd()
        os.chdir(video_frames)
        cv.imwrite(f"{str(frame_n)}.jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        os.chdir(basedir)

        if show:
            font = cv.FONT_HERSHEY_SIMPLEX
            frame = cv.putText(frame, f"{str(frame_n)}/{max_frames}", (5, 25), font, 1, (0, 0, 255), thickness=2)
            cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        frame_n += 1
    cap.release()
    cv.destroyAllWindows()

    if check:
        integrity_check(video_frames, video_labels)
    if label:
        label_images(video_frames, video_labels, labeled_video_frames)


def check_bbox_size(images_folder=f"data/Images/", labels_folder=f"data/Labels/", show=False, less_more='less',
                        w_ratio=0.0, h_ratio=0.0):
    print('\nScanning')
    i = 0
    for filename in tqdm(os.listdir(images_folder)):
        base_name = os.path.splitext(os.path.basename(filename))[0]

        image_filepath = f"{images_folder}{filename}"
        label_filepath = f"{labels_folder}{base_name}.txt"
        found = 0

        lines = read_label(label_filepath)
        new = []
        for line in lines:
            new.append(line.split(' '))
        # line[3] = Width
        # line[4] = Height
        for line in new:
            if w_ratio != 0:
                if less_more == 'less':
                    if float(line[3]) < w_ratio:  # and float(line[4]) < size:
                        found = True
                else:
                    if float(line[3]) > w_ratio:  # and float(line[4]) < size:
                        found = True

            if h_ratio != 0:
                if less_more == 'less':
                    if float(line[4]) < h_ratio:  # and float(line[4]) < size:
                        found = True
                else:
                    if float(line[4]) > h_ratio:  # and float(line[4]) < size:
                        found = True
        if found:
            i += 1
            if show:
                img = cv.imread(image_filepath)
                img = draw_box(img, label_filepath)
                cv.imshow('1', img)
                cv.waitKey(0)
    if less_more == 'less':
        print(f"found a total of {i} files with x < {w_ratio} and y < {h_ratio}")
    else:
        print(f"found a total of {i} files with x > {w_ratio} and y > {h_ratio}")

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


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
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


def split_det_img(imgs_folder=f"data/Images/", lbl_filepath=f"data/Labels/", name=f"sep",
                  conf_thresh=0.5, lbl_images=False, split=True):
    """Images in yolov5 detection runs does not split detection images from rest.
     When saving labels, paste labels in Labels and the images in Images. Choose option 9"""
    Path(f"data/separate/{name}").mkdir(parents=True, exist_ok=True)
    Path(f"data/separate/{name}/images").mkdir(parents=True, exist_ok=True)
    Path(f"data/separate/{name}/labeled_images").mkdir(parents=True, exist_ok=True)
    Path(f"data/separate/{name}/labels").mkdir(parents=True, exist_ok=True)
    for label in os.listdir(lbl_filepath):
        shutil.copy(f"{lbl_filepath}{label}", f"data/separate/{name}/labels/{label}")
    for label in tqdm(os.listdir(f"data/separate/{name}/labels/"), desc='Splitting...'):
        lbl_path = f"data/separate/{name}/labels/{label}"
        lines = read_label(lbl_path)
        new = []
        for line in lines:
            new.append(line.split(' '))
        for line in new:
            if len(line) == 6:
                conf = float(line[5])
                if conf >= conf_thresh:
                    base_name = os.path.splitext(os.path.basename(label))[0]  # bn is basename
                    img_filepath = f"data/separate/images_10000_coco/{base_name}.jpg"
                    shutil.copy(img_filepath, f"data/separate/{name}/images/{base_name}.jpg")
                    break
            else:
                base_name = os.path.splitext(os.path.basename(label))[0]  # bn is basename
                img_filepath = f"data/separate/{name}/images/{base_name}.jpg"
                shutil.copy(img_filepath, f"data/separate/{name}/labeled_images/{base_name}.jpg")
                break
    if lbl_images:
        label_images(images_folder=f"data/separate/{name}/images/",
                     labels_folder=f"data/separate/{name}/labels/",
                     labeled_images_folder=f"data/separate/{name}/labeled_images/")
    if split:
        train_folder = f"data/separate/{name}/into_train/"
        valid_folder = f"data/separate/{name}/into_valid/"

        Path(train_folder).mkdir(parents=True, exist_ok=True)
        Path(valid_folder).mkdir(parents=True, exist_ok=True)

        empty_folder(train_folder)
        empty_folder(valid_folder)

        target_folder = f"data/separate/{name}/images/"
        total_files = len(os.listdir(target_folder))

        '''This segment ensures that the highest conf photos get into the into_train folder'''
        for label in os.listdir(f"data/separate/{name}/labels/"):
            lbl_path = f"data/separate/{name}/labels/{label}"
            lines = read_label(lbl_path)
            new = []
            for line in lines:
                new.append(line.split(' '))
            for line in new:
                if len(line) == 6:
                    conf = float(line[5])
                    if conf >= 0.67:
                        base_name = os.path.splitext(os.path.basename(label))[0]  # bn is basename
                        img_filepath = f"data/separate/images_10000_coco/{base_name}.jpg"
                        shutil.copy(img_filepath, f"data/separate/{name}/into_train/{base_name}.jpg")
                        break
        '''###############################################################################################'''

        print(f"\nRandomly spitting the total number of files, {total_files} into 90% train 20% valid")
        i = len(os.listdir(f"data/separate/{name}/into_train/"))
        while i < int(0.9 * total_files):
            p = random.randint(0, len(os.listdir(target_folder)))
            filename = os.listdir(target_folder)[p - 1]
            filepath = f"{target_folder}{filename}"
            if not os.path.isfile(f"data/separate/{name}/into_train/{filename}"):
                shutil.copy(filepath, f"{train_folder}{filename}")
                i += 1
        for filename in os.listdir(target_folder):
            filepath = f"{target_folder}{filename}"
            if not os.path.isfile(f"data/separate/{name}/into_train/{filename}"):
                if not os.path.isfile(f"data/separate/{name}/into_valid/{filename}"):
                    shutil.copy(filepath, f"{valid_folder}{filename}")


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
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm += 1  # label missing
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


def cls_xyxy2xywh(source_folder):  # Convert cls-xyxy to cls-xywh inplace
    for file in tqdm(os.listdir(source_folder), desc='Converting label format...'):
        with open(f'{source_folder}{file}', 'r') as f:
            lines = f.readlines()
            count = 0
            out_lines = []
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                x = line[1:]
                y = np.copy(x)
                y[0] = (float(x[0]) + float(x[2])) / 2
                y[1] = (float(x[1]) + float(x[3])) / 2
                y[2] = float(x[2]) - float(x[0])  # width
                y[3] = float(x[3]) - float(x[1])  # height
                new_line = [line[0]]
                for seg in y:
                    new_line.append(seg)
                new_line = ' '.join(map(str, new_line))
                if count < len(lines):
                    new_line = f'{new_line}\n'
                    count += 1
                out_lines.append(new_line)

            with open(f'{source_folder}{file}', 'w') as g:
                g.writelines(out_lines)


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

if __name__ == '__main__':
    IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    answer = input(
        f"\nIntegrity check <1> \nLabel images (Yolo formatted) <2> \nSplit into 80% train 15% valid 5% test <3> "
        f"\nChange labels from x1y1x2y2 to Yolo labels <4> \nTrim data with width and height on labels <5> \nScan <6> "
        f"\nBreak video into images <7> \nRename <8> \nSplit images with detections from rest <9> "
        f"\nPlot distribution <10> \nCheck resolution <11> \n")
    print(f"\nYou picked option {answer}")

    if answer == '0':  # yolov5 integrity check
        yolov5_integrity_check('data/Images', 'data/Labels')

    if answer == '1':  # Integrity check
        integrity_check(images_folder='data/Images/', labels_folder='data/Labels/')

    if answer == '2':  # Label images with Yolo formatted labels
        label_images(images_folder='data/Images/', labels_folder='data/new Labels/',
                     labeled_images_folder='data/labeled images/')

    if answer == '3':  # Split into 80% train 15% valid 5% test
        target_folder = 'data/images/'
        total_files = len(os.listdir(target_folder))

        train_images = f"data/split/train/images/"  # Set destination folders
        train_labels = f"data/split/train/labels/"

        valid_images = f"data/split/valid/images/"
        valid_labels = f"data/split/valid/labels/"

        test_images = f"data/split/test/images/"
        test_labels = f"data/split/test/labels/"

        empty_folder(train_images)  # Empty all folders first
        empty_folder(train_labels)

        empty_folder(valid_images)
        empty_folder(valid_labels)

        empty_folder(test_images)
        empty_folder(test_labels)

        print(f"\nRandomly spitting the total number of files, {total_files} into 80% train 15% valid 5% test")
        for i in tqdm(range(int(0.8*total_files)), desc='Creating Training set'):
            p = random.randint(0, len(os.listdir(target_folder)))
            filename = os.listdir(target_folder)[p-1]

            filepath = f"data/images/{filename}"
            base_name = os.path.splitext(os.path.basename(filename))[0]

            shutil.move(filepath, f"{train_images}{filename}")
            shutil.move(f"data/Labels/{base_name}.txt", f"{train_labels}{base_name}.txt")

        for i in tqdm(range(int(0.15*total_files)), desc='Creating Validation set'):
            p = random.randint(0, len(os.listdir(target_folder)))
            filename = os.listdir(target_folder)[p-1]

            filepath = f"data/images/{filename}"
            base_name = os.path.splitext(os.path.basename(filename))[0]

            shutil.move(filepath, f"{valid_images}{filename}")
            shutil.move(f"data/Labels/{base_name}.txt", f"{valid_labels}{base_name}.txt")

        for filename in tqdm(os.listdir(f"data/images/"), desc='Creating Testing set'):
            filepath = f"data/images/{filename}"
            base_name = os.path.splitext(os.path.basename(filename))[0]

            shutil.move(filepath, f"{test_images}{filename}")
            shutil.move(f"data/Labels/{base_name}.txt", f"{test_labels}{base_name}.txt")

    if answer == '4':  # Change labels from x1y1x2y2 to Yolo labels
        print('Fixing...')
        for filename in os.listdir('data/Labels/'):

            filepath = 'data/Labels/' + filename
            base_name = os.path.splitext(os.path.basename(filepath))[0]

            img = cv.imread(f"data/Images/{base_name}.jpg")

            basedir = os.getcwd()
            os.chdir('data/New Labels/')
            with open(filename, "w+") as g:
                os.chdir(basedir)
                lines = read_label(f"data/Labels/{filename}")
                new = []
                writefile = []
                i = 0
                for line in lines:
                    if i > 0:
                        new.append(line.split(' '))
                    i += 1
                print(f"Dimensions of image {base_name} is {img.shape}")
                # .shape[0] = Height
                # .shape[1] = Width
                # .shape[2] = Colour Channels
                i = 1
                for line in new:
                    x1 = int(line[0])
                    y1 = int(line[1])
                    x2 = int(line[2])
                    y2 = int(line[3])
                    x = ((x1 + ((x2 - x1) / 2)) / img.shape[1])
                    y = ((y1 + ((y2 - y1) / 2)) / img.shape[0])
                    w = ((x2 - x1)/img.shape[1])
                    h = ((y2 - y1)/img.shape[0])

                    if i == len(new):
                        g.write(f"0 {x} {y} {w} {h}")
                    else:
                        g.write(f"0 {x} {y} {w} {h}\n")
                    i += 1

    if answer == '5':   # Trim data with width and height on labels
        force_remove = False
        images_folder = 'data/open_imagesv2/images/'  # 'More weapons/Images/'
        labels_folder = 'data/open_imagesv2/labels/'  # 'More weapons/Labels/'

        sp = 0  # small pixels
        bb = 0  # big on both axis
        sb = 0  # small on both axis
        sx = 0  # small on x axis
        sy = 0  # small on y axis
        by = 0  # big on x axis
        bx = 0  # big on y axis
        for filename in tqdm(os.listdir(images_folder), desc='Trimming bases on bbox size'):
            base_name = os.path.splitext(os.path.basename(filename))[0]

            image_filepath = f"{images_folder}{filename}"
            label_filepath = f"{labels_folder}{base_name}.txt"

            flag_to_remove = False
            lines = read_label(label_filepath)
            new = []
            for line in lines:
                new.append(line.split(' '))
            # img.shape[1] = Width
            # img.shape[0] = Height
            # line[3] = Width
            # line[4] = Height
            for seg in new:

                DieFoto = cv.imread(image_filepath)
                if DieFoto.shape[1] * float(seg[3]) < 10 or DieFoto.shape[0] * float(seg[4]) < 13:
                    sp += 1
                    print(f'\nName is {filename}\nImage X px size is {DieFoto.shape[1] * float(seg[3])}\n'
                          f'Box Y px size is {DieFoto.shape[0] * float(seg[4])}')

                if float(seg[3]) > 1 and float(seg[4]) > 1:
                    bb += 1
                    flag_to_remove = True

                elif float(seg[3]) < 0.05 and float(seg[4]) < 0.05:
                    sb += 1
                    flag_to_remove = True

                elif float(seg[3]) < 0.03:
                    sx += 1
                    flag_to_remove = True

                elif float(seg[4]) < 0.03:
                    sy += 1
                    flag_to_remove = True

                elif float(seg[3]) > 1:
                    bx += 1
                    flag_to_remove = True

                elif float(seg[4]) > 1:
                    by += 1
                    flag_to_remove = True

            if flag_to_remove:
                if force_remove:
                    os.remove(image_filepath)
                    os.remove(label_filepath)
        print(sp)
        print(f"found a total of {bb} files with w and h > 1 . {sb} files with w and h < 0.05. {sx} files with w < 0.03"
              f" and {sy} files with h < 0.03. {bx} files with w > 1 and {by} files with h > 1")

    if answer == '6':   # Scan for x size boxes
        check_bbox_size(images_folder=f"data/Images/", labels_folder=f"data/Labels/", show=False, less_more='less',
                        w_ratio=0.03, h_ratio=0.03)

    if answer == '7':  # break video into images
        video_to_frames(video_path=f"data/videos/Hunting Trespasser gets Painted.mp4",
                        video_labels=f"data/videos/video_labels/", video_frames=f"data/videos/video_frames/",
                        labeled_video_frames=f"More weapons/videos/labeled_video_frames/", show=False, check=True,
                        label=True)

    if answer == '8':  # rename
        rename('More weapons/Images/', 'More weapons/Labels/', 'hunter')

    if answer == '9':  # Split img with detection from rest
        split_det_img(imgs_folder=f"data/separate/images_10000_coco/", lbl_filepath=f"data/separate/input_labels/",
                      name=f"sep_ref_ft4_real", conf_thresh=0.5, lbl_images=True)

    if answer == '10':  # Plot distribution
        plot_distribution(f"data/labels_for_plot/")

    if answer == '11':  # Check resolution
        check_resolution(folder=f"data/holder/", w_threshold=320, h_threshold=180, less_more='less', show=False,
                         rescale_save=False, delete=False)

    if answer == '12':
        strip_optimizer(f='best.pt', s='Security_L.pt')

    if answer == '13':  # Convert cls-xyxy to cls-xywh
        pass
