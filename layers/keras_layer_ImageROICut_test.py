import tensorflow as tf
from keras.engine.topology import Layer
from sseg_config import Config
config = Config

class ImageROICut_test(Layer):

    def __init__(self,config,**kwargs):
        super(ImageROICut_test,self).__init__(**kwargs)
        self.config = config

    def call(self,inputs,mask=None):
        """
        Implements ROI cut on img

        :param input_img: [batch_size,img_height,img_width,channel_]
        :param rois: [batch_size,top_k,(class_id,score,xmin,ymin,xmax,ymax)]
        :param config: config

        :return: img_cut [batch_size,cut_size,cut_size,channel]

        """


        input_img = inputs[0]
        rois = inputs[1]

        boxes_input = rois  # [batch_size,top_k,(xmin,ymin,xmax,ymax)]
        x1, y1, x2, y2 = tf.split(boxes_input, 4, axis=2)

        # normalize and get (y1,x1,y2,x2) for tensorflow img_crop
        x1 /= tf.cast(tf.shape(input_img)[2], tf.float32)
        # x1 = tf.cast(x1/512.0, tf.float32)
        # x2 = tf.cast(x2 / 512.0, tf.float32)
        # y1 = tf.cast(y1 / 512.0, tf.float32)
        # y2 = tf.cast(y2 / 512.0, tf.float32)
        x2 /= tf.cast(tf.shape(input_img)[2], tf.float32)
        y1 /= tf.cast(tf.shape(input_img)[1], tf.float32)
        y2 /= tf.cast(tf.shape(input_img)[1], tf.float32)
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
        input_img = tf.stop_gradient(input_img)

        img_cut = tf.image.crop_and_resize(tf.cast(input_img,tf.float32), boxes_cut, box_indices, self.config.IMAGE_CUT_SHAPE[:2],method='bilinear')

        img_cut = tf.reshape(img_cut,[self.config.BATCH_SIZE, self.config.TRAIN_SUB_ROIS_PER_IMAGE, self.config.IMAGE_CUT_SHAPE[0],
                                       self.config.IMAGE_CUT_SHAPE[1], self.config.IMAGE_CUT_SHAPE[2]])
        #img_cut = tf.expand_dims(img_cut,0)


        return img_cut

    def compute_output_shape(self, input_shape):

        return   (self.config.BATCH_SIZE, self.config.TRAIN_SUB_ROIS_PER_IMAGE, self.config.IMAGE_CUT_SHAPE[0],
                                       self.config.IMAGE_CUT_SHAPE[1], self.config.IMAGE_CUT_SHAPE[2])


