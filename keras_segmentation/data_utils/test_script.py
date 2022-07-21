from codecs import ignore_errors
import data_loader  
import cv2 as cv2  

images_path = "/Volumes/WD_Drive/MSc_Project/CATARACTS_SEGMENTATION/CaDISv2/Video01/Images"
segs_path = "/Volumes/WD_Drive/MSc_Project/CATARACTS_SEGMENTATION/CaDISv2/Video01/Labels"
other_inputs_path = "/Users/vanessaigodifo/MSc_Project/Optical_Flows/Video01"
batch_size = 64
n_classes = 3
input_height = 540
input_width = 960
output_height = input_height
output_width = input_width
do_augment = False
augmentation_name = "aug_all"
custom_augmentation = None
preprocessing = None  
read_image_type = cv2.IMREAD_COLOR
ignore_segs = False


data_loader.image_segmentation_generator(images_path, segs_path, batch_size,
                                        n_classes, input_height, input_width,
                                        output_height, output_width,
                                        do_augment=False,
                                        augmentation_name="aug_all",
                                        custom_augmentation=None,
                                        other_inputs_paths=None, preprocessing=None,
                                        read_image_type=cv2.IMREAD_COLOR , ignore_segs=False)
