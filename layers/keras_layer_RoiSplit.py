from utils import common_utils as utils
from keras.engine.topology import Layer
import tensorflow as tf
from sseg_config import Config

config = Config()

def rois_split(rois_all):
    """

    :param rois_all: [top_k, (class_id,score,xmin,ymin,xmax,ymax)]
    :return:
    """

    class_ids = rois_all[:, 0]
    scores = rois_all[:, 1]
    rois = rois_all[:, 2:]

    # eye_l
    one_bool = (class_ids >= 1) & (class_ids <= 1)
    one_index = tf.where(one_bool)[:, 0]

    select_length = tf.minimum(tf.shape(one_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    one_index = one_index[:select_length]

    rois1 = tf.gather(rois, one_index)
    PD1 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois1)[0], 0)
    rois1 = tf.pad(rois1, [(0, PD1), (0, 0)])


    # eye_r
    two_bool = (class_ids >= 2) & (class_ids <= 2)
    two_index = tf.where(two_bool)[:, 0]

    select_length = tf.minimum(tf.shape(two_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    two_index = two_index[:select_length]

    rois2 = tf.gather(rois, two_index)
    PD2 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois2)[0], 0)
    rois2 = tf.pad(rois2, [(0, PD2), (0, 0)])

    # nose
    three_bool = (class_ids >= 3) & (class_ids <= 3)
    three_index = tf.where(three_bool)[:, 0]

    select_length = tf.minimum(tf.shape(three_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    three_index = three_index[:select_length]

    rois3 = tf.gather(rois, three_index)
    PD3 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois3)[0], 0)
    rois3 = tf.pad(rois3, [(0, PD3), (0, 0)])

    # mouth
    four_bool = (class_ids >= 4) & (class_ids <= 4)
    four_index = tf.where(four_bool)[:, 0]

    select_length = tf.minimum(tf.shape(four_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    four_index = four_index[:select_length]

    rois4 = tf.gather(rois, four_index)
    PD4 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois4)[0], 0)
    rois4 = tf.pad(rois4, [(0, PD4), (0, 0)])


    # face
    five_bool = (class_ids >= 5) & (class_ids <= 5)
    five_index = tf.where(five_bool)[:, 0]

    select_length = tf.minimum(tf.shape(five_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    five_index = five_index[:select_length]

    rois5 = tf.gather(rois, five_index)
    PD5 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois5)[0], 0)
    rois5 = tf.pad(rois5, [(0, PD5), (0, 0)])



    return rois1,rois2,rois3,rois4,rois5


class RoiSplit(Layer):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.


    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
          coordinates
    rois2: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
      coordinates
    rois3: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
      coordinates
    rois4: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
          coordinates

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(RoiSplit, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs,mask=None):
        proposals = inputs[0]


        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois1","rois2","rois3","rois4","rois5"]


        outputs = utils.batch_slice(
            [proposals],
            lambda x: rois_split(x),
            self.config.BATCH_SIZE, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),# rois
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),  # rois

        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]



def rois_split_score(rois_all):
    """

    :param rois_all: [top_k, (class_id,score,xmin,ymin,xmax,ymax)]
    :return:
    """

    class_ids = rois_all[:, 0]
    scores = rois_all[:, 1]
    rois = rois_all[:, 2:]

    # eye_l
    one_bool = (class_ids >= 1) & (class_ids <= 1)
    one_index = tf.where(one_bool)[:, 0]

    select_length = tf.minimum(tf.shape(one_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    one_index = one_index[:select_length]

    rois1 = tf.gather(rois, one_index)
    PD1 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois1)[0], 0)
    rois1 = tf.pad(rois1, [(0, PD1), (0, 0)])

    score1 = tf.gather(scores, one_index)
    SPD1 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(score1)[0], 0)
    score1 = tf.pad(score1, [(0, SPD1)])


    # eye_r
    two_bool = (class_ids >= 2) & (class_ids <= 2)
    two_index = tf.where(two_bool)[:, 0]

    select_length = tf.minimum(tf.shape(two_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    two_index = two_index[:select_length]

    rois2 = tf.gather(rois, two_index)
    PD2 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois2)[0], 0)
    rois2 = tf.pad(rois2, [(0, PD2), (0, 0)])

    score2 = tf.gather(scores, two_index)
    SPD2 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(score2)[0], 0)
    score2 = tf.pad(score2, [(0, SPD2)])

    # nose
    three_bool = (class_ids >= 3) & (class_ids <= 3)
    three_index = tf.where(three_bool)[:, 0]

    select_length = tf.minimum(tf.shape(three_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    three_index = three_index[:select_length]

    rois3 = tf.gather(rois, three_index)
    PD3 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois3)[0], 0)
    rois3 = tf.pad(rois3, [(0, PD3), (0, 0)])

    score3 = tf.gather(scores, three_index)
    SPD3 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(score3)[0], 0)
    score3 = tf.pad(score3, [(0, SPD3)])

    # mouth
    four_bool = (class_ids >= 4) & (class_ids <= 4)
    four_index = tf.where(four_bool)[:, 0]

    select_length = tf.minimum(tf.shape(four_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    four_index = four_index[:select_length]

    rois4 = tf.gather(rois, four_index)
    PD4 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois4)[0], 0)
    rois4 = tf.pad(rois4, [(0, PD4), (0, 0)])


    score4 = tf.gather(scores, four_index)
    SPD4 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(score4)[0], 0)
    score4 = tf.pad(score4, [(0, SPD4)])


    # face
    five_bool = (class_ids >= 5) & (class_ids <= 5)
    five_index = tf.where(five_bool)[:, 0]

    select_length = tf.minimum(tf.shape(five_index)[0],config.TRAIN_SUB_ROIS_PER_IMAGE)
    five_index = five_index[:select_length]

    rois5 = tf.gather(rois, five_index)
    PD5 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(rois5)[0], 0)
    rois5 = tf.pad(rois5, [(0, PD5), (0, 0)])

    score5 = tf.gather(scores, five_index)
    SPD5 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(score5)[0], 0)
    score5 = tf.pad(score5, [(0, SPD5)])



    return rois1,rois2,rois3,rois4,rois5,score1,score2,score3,score4,score5


class RoiSplit_score(Layer):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.


    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
          coordinates
    rois2: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
      coordinates
    rois3: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
      coordinates
    rois4: [batch, TRAIN_SUBROIS_PER_IMAGE, (xmin,ymin,xmax,ymax)] in normalized
          coordinates

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(RoiSplit_score, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs,mask=None):
        proposals = inputs[0]


        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois1","rois2","rois3","rois4","rois5",
                 "socre1","score2","score3","score4","score5"]


        outputs = utils.batch_slice(
            [proposals],
            lambda x: rois_split_score(x),
            self.config.BATCH_SIZE, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),# rois
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE, 4),  # rois

            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE),  # score
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE),  # score
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE),  # score
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE),  # score
            (None, self.config.TRAIN_SUB_ROIS_PER_IMAGE),  # score

        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None,None, None, None, None, None]
