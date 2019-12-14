import tensorflow as tf
from keras.engine.topology import Layer
from sseg_config import Config
config = Config

class ImageROICut(Layer):

    def __init__(self,config,**kwargs):
        super(ImageROICut,self).__init__(**kwargs)
        self.config = config

    def call(self,inputs,mask=None):
        """
        Implements ROI cut on img

        :param input_img: [batch_size,img_height,img_width,channel_]
        :param input_gt_mask: [batch_size,img_height,img_width,sub_class]
        :param rois: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,(xmin,ymin,xmax,ymax)]
        :param config: config

        :return: img_cut [batch_size,cut_size,cut_size,channel]
        :return: mask_cut [batch_size,cut_size,cut_size,sub_class]

        """

        self.inputs = inputs
        share_features = self.inputs[0]
        input_gt_mask = self.inputs[1]
        rois = self.inputs[2]

        boxes_input = rois  # [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,(xmin,ymin,xmax,ymax)]
        x1, y1, x2, y2 = tf.split(boxes_input, 4, axis=2)

        # normalize and get (y1,x1,y2,x2) for tensorflow img_crop
        x1 /= tf.cast(self.config.IMAGE_SHAPE[1], tf.float32)
        x2 /= tf.cast(self.config.IMAGE_SHAPE[1], tf.float32)
        y1 /= tf.cast(self.config.IMAGE_SHAPE[0], tf.float32)
        y2 /= tf.cast(self.config.IMAGE_SHAPE[0], tf.float32)
        boxes = tf.concat([y1, x1, y2, x2], axis=2)

        boxes_cut = tf.reshape(boxes, [-1, 4])
        box_indicest = []
        for i in range(self.config.BATCH_SIZE):
            box_indicest.append(tf.ones([tf.shape(rois)[1]]) * i)
        box_indices = tf.stack(box_indicest)
        box_indices = tf.reshape(box_indices, [-1])
        box_indices = tf.cast(box_indices, tf.int32)

        boxes_cut = tf.stop_gradient(boxes_cut)
        box_indices = tf.stop_gradient(box_indices)

        feature_cuts=[]
        for i in range(len(share_features)):
            feature_cut = tf.image.crop_and_resize(share_features[0], boxes_cut, box_indices, self.config.IMAGE_CUT_SHAPE[:2])

            feature_cut = tf.reshape(feature_cut,[tf.shape(rois)[0], tf.shape(rois)[1], self.config.IMAGE_CUT_SHAPE[0],
                                           self.config.IMAGE_CUT_SHAPE[1], tf.shape(share_features[0])[-1]])
            feature_cuts.append(feature_cut)
        feature_cuts = tf.concat(feature_cuts,axis=-1)


        target_mask = tf.image.crop_and_resize(input_gt_mask, boxes_cut, box_indices, self.config.IMAGE_CUT_SHAPE[:2])
        target_mask = tf.reshape(target_mask,[self.config.BATCH_SIZE, self.config.TRAIN_SUB_ROIS_PER_IMAGE, self.config.IMAGE_CUT_SHAPE[0],
                                       self.config.IMAGE_CUT_SHAPE[1], self.config.SUB_CLASS])

        return [feature_cuts, target_mask]

    def compute_output_shape(self, input_shape):

        return [
            (self.config.BATCH_SIZE, self.config.TRAIN_SUB_ROIS_PER_IMAGE, self.config.IMAGE_CUT_SHAPE[0],
                                       self.config.IMAGE_CUT_SHAPE[1], self.config.IMAGE_CUT_SHAPE[2]*4),
            (self.config.BATCH_SIZE, self.config.TRAIN_SUB_ROIS_PER_IMAGE, self.config.IMAGE_CUT_SHAPE[0],
                                               self.config.IMAGE_CUT_SHAPE[1],
                                               9)
        ]

