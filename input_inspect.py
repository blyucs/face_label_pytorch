import sseg_data_input2 as sseg_data_input
from sseg_config import Config
from utils.ssd_box_encode_decode_utils import SSDBoxEncoder

config = Config()
predictor_sizes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
ssd_box_encoder = SSDBoxEncoder(img_height=config.img_height,
                                img_width=config.img_width,
                                n_classes=config.n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=config.scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=config.aspect_ratios,
                                two_boxes_for_ar1=config.two_boxes_for_ar1,
                                steps=config.steps,
                                offsets=config.offsets,
                                limit_boxes=config.limit_boxes,
                                variances=config.variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=config.coords,
                                normalize_coords=config.normalize_coords)


train_generator = sseg_data_input.data_generator_helen(batch_size=config.BATCH_SIZE, shuffle=True,
                                                       ssd_box_encoder=ssd_box_encoder,
                                                       image_shape=[config.img_height, config.img_width], config=config,
                                                       augment=True)


for i in range(100):
    _ = next(train_generator)