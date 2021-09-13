# yolov5-dataset-builder
A wrapper around the OIDv6 downloader to create yolov5 datasets with one command

WORK IN PROGRESS

To test it out, just git clone and run the following command to get a mini dataset of Cats and Dog:                                                   

python main.py --classes Cat Dog --limit 20 --type_data test --dataset MiniCatsDogs --duplicates --label_img --split_data --resize_img 640


Arguments:

--dataset, default='dataset', help='Name of the dataset.'

--dataset_exists, action='store_true', help='Save images to an existing dataset.'

--duplicates, action='store_true', help='Check for duplicate images. Can be slow on big data.'

--label_img, action='store_true', help='Create a separate folder and label the images.'

--split_data, action='store_true', help='Split dataset into train, valid, test split.'

--resize_img, type=int, default=0, help='Size in px to resize images to.'

--verbose, action='store_true', help='Print out more info on every step.'

--no_download, action='store_true', help='Do not use the OIDv6 downloader to download images.'

--type_data, type=str, default='train', help='Download from subset [train, validation, test, all].'

--limit, default=0, type=int, help='Limit the amount of images to download.'

--classes, nargs='+', type=str, help='Path to classes.txt or list classes like "Cat Dog Ant".', required=True


More documentation to come soon
