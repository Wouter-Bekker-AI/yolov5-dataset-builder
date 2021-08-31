# yolov5-dataset-builder
A wrapper around the OIDv6 downloader to create yolov5 datasets with one command

WORK IN PROGRESS

To test it out, just git clone and run the following command to get a mini dataset of Cats and Dog:                                                   

python main.py --classes Cat Dog --limit 20 --type_data test --download --dataset MiniCatsDogs --duplicates --label_img --split_data --resize_img 640

More documentation to come soon
