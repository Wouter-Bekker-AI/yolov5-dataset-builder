import numpy as np
import cv2 as cv
import os
import imghdr
from tqdm import tqdm


""" 
Duplicate Image Finder (DIF): function that searches a given directory for images and finds duplicate/similar images among them.
Outputs the number of found duplicate/similar image pairs with a list of the filenames having lower resolution.
"""

image_files = []
lower_res = []


def compare_images(directory,  show_imgs=True,  similarity="high", force_remove=False,  compression=50,  rotate=False,
                   verbose=False):
    """
    directory (str).........Folder to search for duplicate/similar images
    show_imgs (bool)........True = shows the duplicate/similar images found in output
                            False = doesn't show found images
    similarity (str)........"high" = Searches for duplicate images, more precise
                            "low" = Finds similar images
    compression (int).......Recommended not to change default value
                            compression in px (height x width) of the images before being compared
                            the higher the compression i.e. the higher the pixel size, the more computational resources
                            and time required
    verbose (bool)..........Print results to console if True
    force_remove (bool).....Removes duplicates if True
    """
    # list where the found duplicate/similar images are stored
    global lower_res

    imgs_matrix = create_imgs_matrix(directory, compression)

    if imgs_matrix.any():
        if similarity == "low":  # search for similar images
            ref = 1000
        else:                    # search for 1:1 duplicate images
            ref = 200

        main_img = 0
        compared_img = 1
        nrows, ncols = compression, compression
        srow_A = 0
        erow_A = nrows
        srow_B = erow_A
        erow_B = srow_B + nrows

        pbar = tqdm(total=len(folder_files), desc='Checking for duplicates')
        while erow_B <= imgs_matrix.shape[0]:
            halt = 0
            while compared_img < (len(image_files)) and not halt:
                # select two images from imgs_matrix
                imgA = imgs_matrix[srow_A: erow_A, 0: ncols]  # rows # columns
                imgB = imgs_matrix[srow_B: erow_B, 0: ncols]  # rows # columns
                # compare the images
                if not rotate:
                    if image_files[main_img] not in lower_res:
                        err = mse(imgA, imgB)
                        if err < ref:
                            if show_imgs == True:
                                show_img_figs(image_files[main_img], image_files[compared_img], err)
                                show_file_info(compared_img, main_img)
                            halt = check_img_quality(directory, image_files[main_img], image_files[compared_img], lower_res)
                elif rotate:
                    rotations = 0  # check all rotations of images
                    while image_files[main_img] not in lower_res and rotations <= 3:
                        if rotations != 0:
                            imgB = rotate_img(imgB)
                        err = mse(imgA, imgB)
                        if err < ref:
                            if show_imgs == True:
                                show_img_figs(image_files[main_img], image_files[compared_img], err)
                                show_file_info(compared_img, main_img)
                            halt = check_img_quality(directory, image_files[main_img], image_files[compared_img], lower_res)
                        rotations += 1
                srow_B += nrows
                erow_B += nrows
                compared_img += 1
            srow_A += nrows
            erow_A += nrows
            srow_B = erow_A
            erow_B = srow_B + nrows
            main_img += 1
            compared_img = main_img + 1
            pbar.update(1)
        pbar.close()

        if verbose:
            print(f'\nDONE. Found {len(lower_res)} duplicate image pairs in {len(folder_files)} total images.')
            if len(lower_res) > 0:
                print(f'\nThe following files had lower resolution:\n{lower_res}')

        if force_remove:
            for filename in lower_res:
                try:
                    os.remove(f"{directory}{filename}")
                except Exception:
                    print(f"Could not remove {filename}")


# Function that searches the folder for image files, converts them to a matrix
def create_imgs_matrix(directory, compression):
    imgs_matrix = None
    global image_files
    # create list of all files in directory
    global folder_files
    folder_files = [filename for filename in os.listdir(directory)]
    
    # create images matrix   
    counter = 0
    for filename in tqdm(folder_files, desc='Creating Image Matrix'):
        if not os.path.isdir(directory + filename) and imghdr.what(directory + filename):
            img = cv.imdecode(np.fromfile(directory + filename, dtype=np.uint8), cv.IMREAD_UNCHANGED)
            if type(img) == np.ndarray:
                if len(img.shape) == 2:  # conver greyscale img to 3 channel img
                    img = np.stack((img,) * 3, axis=-1)
                img = img[...,0:3]
                img = cv.resize(img, dsize=(compression, compression), interpolation=cv.INTER_CUBIC)
                if counter == 0:
                    imgs_matrix = img
                    image_files.append(filename)
                    counter += 1
                else:
                    try:
                        imgs_matrix = np.concatenate((imgs_matrix, img))
                        image_files.append(filename)
                    except:
                        print(f"Can't add {filename} to Matrix. It might be a black and white image.")
    return imgs_matrix


# Function that calulates the mean squared error (mse) between two image matrices
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Function that plots two compared image files and their mse
def show_img_figs(imageA, imageB, err):
    img_a = cv.imread(f"{folder}{imageA}")
    img_b = cv.imread(f"{folder}{imageB}")
    img_a = cv.resize(img_a, dsize=(640, 640), interpolation=cv.INTER_CUBIC)
    img_b = cv.resize(img_b, dsize=(640, 640), interpolation=cv.INTER_CUBIC)
    display = np.concatenate((img_a, img_b), axis=1)
    cv.imshow(f"{imageA} and {imageB} has a mse of {err}", display)
    cv.waitKey(0)
    cv.destroyAllWindows()


#Function for rotating an image matrix by a 90 degree angle
def rotate_img(image):
    image = np.rot90(image, k=1, axes=(0, 1))
    return image


# Function for printing filename info of plotted image files
def show_file_info(compared_img, main_img):
    print("Duplicate file: " + image_files[main_img] + " and " + image_files[compared_img])


# Function for appending items to a list
def add_to_list(filename, list):
    list.append(filename)


# Function for checking the quality of compared images, appends the lower quality image to the list
def check_img_quality(directory, imageA, imageB, list):
    size_imgA = os.stat(directory + imageA).st_size
    size_imgB = os.stat(directory + imageB).st_size
    stop = False
    if size_imgA > size_imgB:
        if imageB not in lower_res:
            add_to_list(imageB, list)
    else:
        if imageA not in lower_res:
            add_to_list(imageA, list)
            stop = True
    return stop


if __name__ == '__main__':
    #global lower_res, folder, image_files

    folder = 'test_images/'
    compare_images(folder, show_imgs=False, similarity="High ", compression=50, rotate=False, force_remove=False)




