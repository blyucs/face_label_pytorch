


class Config(object):

    SUB_CLASS = 1 + 8 + 1 #(face + conp + bg)
    IMAGE_SHAPE = [512,512,3]
    IMAGE_CUT_SHAPE = [28,28,64] # per feature level
    LEARNING_RATE = 0.0001   #0.0001
    IMAGE_MIN_DIM = 512
    WEIGHT_DECAY= 5e-4

    TOPK = 100

    BATCH_SIZE = 2

    TRAIN_SUB_ROIS_PER_IMAGE = 20

    img_height = 512  # Height of the input images
    img_width = 512  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    subtract_mean = [84.4, 72.0, 64.6]  # The per-channel mean of the images in the dataset
    swap_channels = True  # The color channel order in the original SSD is BGR
    n_classes = 5  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    # scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    # scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    #scales =None# [0.08, 0.16, 0.32, 0.64, 0.96]
    scales = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]   # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 128, 256,
             512]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    limit_boxes = True  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
    coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
    normalize_coords = True


    def __init__(self):

        self.Non = 0




