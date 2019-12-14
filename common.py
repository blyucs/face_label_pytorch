import tensorflow as tf
from sseg_config import Config


config = Config()


def rois_split(rois_all):
    """

    :param rois_all: [batch_size, top_k, (class_id,score,xmin,ymin,xmax,ymax)]
    :return:
    """

    class_ids = rois_all[:,:,0]
    scores = rois_all[:,:,1]
    boxs = rois_all[:,:,2:]

    # eye_l
    one_bool = (class_ids >= 1) & (class_ids <= 1)
    one_index = tf.where(one_bool)[:,0]

    boxs1 = tf.gather(boxs, one_index)

    PD1 = tf.maximum(config.TRAIN_SUB_ROIS_PER_IMAGE - tf.shape(boxs1)[0], 0)
    boxs1 = tf.pad(boxs1,[(0,PD1),(0,0)])

    return boxs1

if __name__ == '__main__':
    sess = tf.InteractiveSession()



