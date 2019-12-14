import tensorflow as tf
from keras.engine.topology import Layer
from sseg_config import Config
config = Config

class ROIAlign(Layer):

    def __init__(self,config,**kwargs):
        super(ROIAlign,self).__init__(**kwargs)
        self.config = config
        self.cut_shape=(self.config.IMAGE_CUT_SHAPE[0],self.config.IMAGE_CUT_SHAPE[1],)

    def call(self,inputs,mask=None):
        """
        Implements ROI cut on img

        :param input_img: [batch_size,img_height,img_width,channel_]
        :param rois: [batch_size,top_k,(xmin,ymin,xmax,ymax)]
        :param config: config

        :return: img_cut [batch_size,cut_size,cut_size,channel]

        """


        input_feaures = inputs[0]
        rois = inputs[1]

        boxes_input = rois  # [batch_size,top_k,(xmin,ymin,xmax,ymax)]
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
            box_indicest.append(tf.ones([tf.shape(inputs[1])[1]]) * i)
        box_indices = tf.stack(box_indicest)
        box_indices = tf.reshape(box_indices, [-1])
        box_indices = tf.cast(box_indices, tf.int32)

        boxes_cut = tf.stop_gradient(boxes_cut)
        box_indices = tf.stop_gradient(box_indices)
        #input_img = tf.stop_gradient(input_img)

        feature_cut = tf.image.crop_and_resize(tf.cast(input_feaures,tf.float32), boxes_cut, box_indices, self.config.IMAGE_CUT_SHAPE[:2],method='bilinear')

        feature_cut = tf.reshape(feature_cut,[tf.shape(rois)[0], tf.shape(rois)[1], self.config.IMAGE_CUT_SHAPE[0],
                                       self.config.IMAGE_CUT_SHAPE[1], tf.shape(input_feaures)[-1]])



        return feature_cut

    def compute_output_shape(self, input_shape):

        #return input_shape[1][:2]+ self.cut_shape+(input_shape[0][-1],)
        return (self.config.BATCH_SIZE,
                self.config.TRAIN_SUB_ROIS_PER_IMAGE,
                self.config.IMAGE_CUT_SHAPE[0],
                self.config.IMAGE_CUT_SHAPE[1],
                input_shape[0][-1])

class ROIAlign_Target(Layer):

    def __init__(self,config,**kwargs):
        super(ROIAlign_Target,self).__init__(**kwargs)
        self.config = config


    def call(self,inputs,mask=None):
        """
        Implements ROI cut on img

        :param input_img: [batch_size,img_height,img_width,channel_]
        :param rois: [batch_size,top_k,(class_id,score,xmin,ymin,xmax,ymax)]
        :param config: config

        :return: img_cut [batch_size,cut_size,cut_size,channel]

        """


        input_feaures = inputs[0]
        rois = inputs[1]

        boxes_input = rois  # [batch_size,top_k,(xmin,ymin,xmax,ymax)]
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
            box_indicest.append(tf.ones([tf.shape(inputs[1])[1]]) * i)
        box_indices = tf.stack(box_indicest)
        box_indices = tf.reshape(box_indices, [-1])
        box_indices = tf.cast(box_indices, tf.int32)

        boxes_cut = tf.stop_gradient(boxes_cut)
        box_indices = tf.stop_gradient(box_indices)
        #input_img = tf.stop_gradient(input_img)

        feature_cut = tf.image.crop_and_resize(tf.cast(input_feaures,tf.float32), boxes_cut, box_indices, [self.config.IMAGE_CUT_SHAPE[0]*4,self.config.IMAGE_CUT_SHAPE[1]*4],method='bilinear')

        feature_cut = tf.reshape(feature_cut,[tf.shape(rois)[0], tf.shape(rois)[1], self.config.IMAGE_CUT_SHAPE[0]*4,
                                       self.config.IMAGE_CUT_SHAPE[1]*4, tf.shape(input_feaures)[-1]])



        return feature_cut

    def compute_output_shape(self, input_shape):

        #return input_shape[1][:2]+ self.cut_shape+(input_shape[0][-1],)
        return (self.config.BATCH_SIZE,
                self.config.TRAIN_SUB_ROIS_PER_IMAGE,
                self.config.IMAGE_CUT_SHAPE[0]*4,
                self.config.IMAGE_CUT_SHAPE[1]*4,
                input_shape[0][-1])


