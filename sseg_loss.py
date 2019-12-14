import tensorflow as tf
import keras.backend as K





def boxc_loss(y_true, y_pred):
    '''
    Compute the loss of the SSD model prediction against the ground truth.

    Arguments:
        y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
            where `#boxes` is the total number of boxes that the model predicts
            per image. Be careful to make sure that the index of each given
            box in `y_true` is the same as the index for the corresponding
            box in `y_pred`. The last axis must have length `#classes + 12` and contain
            `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
            in this order, including the background class. The last eight entries of the
            last axis are not used by this function and therefore their contents are
            irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
            where the last four entries of the last axis contain the anchor box
            coordinates, which are needed during inference. Important: Boxes that
            you want the cost function to ignore need to have a one-hot
            class vector of all zeros.
        y_pred (Keras tensor): The model prediction. The shape is identical
            to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
            The last axis must contain entries in the format
            `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

    Returns:
        A scalar, the total multitask loss for classification and localization.
    '''

    neg_pos_ratio=3
    n_neg_min=0
    alpha=1.0

    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    n_boxes = tf.shape(y_pred)[1]  # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

    # 1: Compute the losses for class and box predictions for every box

    classification_loss = tf.to_float(
        _log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
    localization_loss = tf.to_float(
        _smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

    # 2: Compute the classification losses for the positive and negative targets

    # Create masks for the positive and negative ground truth classes
    negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
    positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # Tensor of shape (batch_size, n_boxes)

    # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
    n_positive = tf.reduce_sum(positives)

    # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
    # (Keras loss functions must output one scalar loss value PER batch item, rather than just
    # one scalar for the entire batch, that's why we're not summing across all axes)
    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    # Compute the classification loss for the negative default boxes (if there are any)

    # First, compute the classification loss for all negative boxes
    neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
    n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                    dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
    # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
    # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
    # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
    # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
    # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
    # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
    # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
    # is at most the number of negative boxes for which there is a positive classification loss.

    # Compute the number of negative examples we want to account for in the loss
    # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.to_int32(n_positive), n_neg_min), n_neg_losses)

    # In the unlikely case when either (1) there are no negative ground truth boxes at all
    # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
    def f1():
        return tf.zeros([batch_size])

    # Otherwise compute the negative loss
    def f2():
        # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
        # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
        # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

        # To do this, we reshape `neg_class_loss_all` to 1D...
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
        # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
        values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)  # We don't need sorting
        # ...and with these indices we'll create a mask...
        negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
        negatives_keep = tf.to_float(
            tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)
        # ...and use it to keep only those boxes and mask all other classification losses
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)  # Tensor of shape (batch_size,)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

    class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

    # 3: Compute the localization loss for the positive targets
    #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    # 4: Compute the total loss

    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`  #[batch_size]
    # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
    # because the relevant criterion to average our loss over is the number of positive boxes in the batch
    # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
    # over the batch size, we'll have to multiply by it.
    #total_loss *= tf.to_float(batch_size)
    #total_loss = K.mean(total_loss)
    #total_loss = K.reshape(total_loss, [1, 1])
    return total_loss




def boxc_loss_focal(y_true, y_pred):

    '''
    Compute the loss of the SSD model prediction against the ground truth.

    Arguments:
        y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
            where `#boxes` is the total number of boxes that the model predicts
            per image. Be careful to make sure that the index of each given
            box in `y_true` is the same as the index for the corresponding
            box in `y_pred`. The last axis must have length `#classes + 12` and contain
            `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
            in this order, including the background class. The last eight entries of the
            last axis are not used by this function and therefore their contents are
            irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
            where the last four entries of the last axis contain the anchor box
            coordinates, which are needed during inference. Important: Boxes that
            you want the cost function to ignore need to have a one-hot
            class vector of all zeros.
        y_pred (Keras tensor): The model prediction. The shape is identical
            to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
            The last axis must contain entries in the format
            `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

    Returns:
        A scalar, the total multitask loss for classification and localization.
    '''

    loc_true = y_true[:,:,-12:-8]
    loc_pred = y_pred[:,:,-12:-8]

    class_true = y_true[:,:,:-12]
    class_pred = y_pred[:,:,:-12]


    # compute location loss
    localization_loss = tf.to_float(_smooth_L1_loss(loc_true, loc_pred))  # Output shape: (batch_size, n_boxes)

    positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # Tensor of shape (batch_size, n_boxes)
    n_positive = tf.reduce_sum(positives)
    loc_loss = tf.reduce_sum(localization_loss * positives)  # one scalar                Tensor of shape (batch_size,)


    # compute class los
    class_loss = softmax_focal(class_true,class_pred)


    # compute all loss
    total_loss = (class_loss + loc_loss) / tf.maximum(1.0, n_positive)

    return total_loss


def softmax_focal(y_true, y_pred,gamma=2.0):
    """ Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

    Returns
        The focal loss of y_pred w.r.t. y_true.
    """

    epsilon = K.constant(1e-4)
    y_true = K.reshape(y_true,(-1,tf.shape(y_true)[-1]))
    y_pred = K.reshape(y_pred, (-1, tf.shape(y_true)[-1]))+epsilon

    focal_weight = (tf.ones(tf.shape(y_pred))-y_pred)**gamma

    cls_loss = -tf.reduce_sum(focal_weight*y_true*K.log(y_pred))



    return cls_loss


def _focal(y_true, y_pred,gamma=2.0):
    """ Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

    Returns
        The focal loss of y_pred w.r.t. y_true.
    """
    labels         = y_true
    classification = y_pred

    focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
    focal_weight = focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(labels, classification) # (batch_size, N, num_classes)


    return tf.reduce_sum(cls_loss, axis=[1,2]) #(batch_size)




def _smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.

    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def _log_loss(y_true, y_pred):
    '''
    Compute the softmax log loss.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.

    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
    return log_loss



def mask_loss1(target_masks_all, pred_masks):
    '''

    :param target_masks: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,all_sub_class]
    :param pred_masks:  [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,sub_class]
    :return:
    '''

    target_masks = []
    sub_class = 3

    target_masks.append(target_masks_all[:, :, :, :, 0])
    target_masks.append(target_masks_all[:, :, :, :, 2])
    inttensor = tf.cast(((target_masks_all[:, :, :, :, 0]>0)|(target_masks_all[:, :, :, :, 2]>0)),dtype=tf.float32)
    target_masks.append(tf.ones(tf.shape(pred_masks)[2:4])-inttensor)

    target_masks = tf.stack(target_masks, axis=-1)

    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3],mask_shape[4]))

    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    ytsum = tf.reduce_sum(target_masks[:,:,:,:-1],axis=[1,2,3])
    p_bool = ytsum>0
    p_index = tf.where(p_bool)[:,0]

    y_true = tf.gather(target_masks,p_index)
    y_pred = tf.gather(pred_masks,p_index)

    # y_true = target_masks
    # y_pred = pred_masks




    loss = K.switch(tf.size(y_true) > 0,
                    _cross_loss(y_true=y_true, y_pred=y_pred,sub_class=sub_class),
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def mask_loss2(target_masks_all, pred_masks):
    '''

    :param target_masks: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,all_sub_class]
    :param pred_masks:  [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,sub_class]
    :return:
    '''

    target_masks = []
    sub_class = 3

    target_masks.append(target_masks_all[:, :, :, :, 1])
    target_masks.append(target_masks_all[:, :, :, :, 3])
    inttensor = tf.cast(((target_masks_all[:, :, :, :, 1]>0)|(target_masks_all[:, :, :, :, 3]>0)),dtype=tf.float32)
    target_masks.append(tf.ones(tf.shape(pred_masks)[2:4])-inttensor)

    target_masks = tf.stack(target_masks, axis=-1)


    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3],mask_shape[4]))

    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    ytsum = tf.reduce_sum(target_masks[:, :, :, :-1], axis=[1, 2, 3])
    p_bool = ytsum > 0
    p_index = tf.where(p_bool)[:, 0]

    y_true = tf.gather(target_masks, p_index)
    y_pred = tf.gather(pred_masks, p_index)




    loss = K.switch(tf.size(y_true) > 0,
                    _cross_loss(y_true=y_true, y_pred=y_pred,sub_class=sub_class),
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def mask_loss3(target_masks_all, pred_masks):
    '''

    :param target_masks: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,all_sub_class]
    :param pred_masks:  [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,sub_class]
    :return:
    '''

    target_masks = []
    sub_class = 2

    target_masks.append(target_masks_all[:, :, :, :, 4])

    target_masks.append(tf.ones(tf.shape(pred_masks)[2:4])-target_masks_all[:, :, :, :, 4])

    target_masks = tf.stack(target_masks, axis=-1)

    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3],mask_shape[4]))

    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    ytsum = tf.reduce_sum(target_masks[:, :, :, :-1], axis=[1, 2, 3])
    p_bool = ytsum > 0
    p_index = tf.where(p_bool)[:, 0]

    y_true = tf.gather(target_masks, p_index)
    y_pred = tf.gather(pred_masks, p_index)



    loss = K.switch(tf.size(y_true) > 0,
                    _cross_loss(y_true=y_true, y_pred=y_pred,sub_class=sub_class),
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def mask_loss4(target_masks_all, pred_masks):
    '''

    :param target_masks: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,all_sub_class]
    :param pred_masks:  [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,sub_class]
    :return:
    '''

    target_masks = []
    sub_class = 4

    target_masks.append(target_masks_all[:, :, :, :, 5])
    target_masks.append(target_masks_all[:, :, :, :, 6])
    target_masks.append(target_masks_all[:, :, :, :, 7])
    inttensor = tf.cast(((target_masks_all[:, :, :, :, 5]>0)|(target_masks_all[:, :, :, :, 6]>0)|(target_masks_all[:, :, :, :, 7]>0)),dtype=tf.float32)
    target_masks.append(tf.ones(tf.shape(pred_masks)[2:4])-inttensor)

    target_masks = tf.stack(target_masks,axis=-1)



    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3],mask_shape[4]))

    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    ytsum = tf.reduce_sum(target_masks[:, :, :, :-1], axis=[1, 2, 3])
    p_bool = ytsum > 0
    p_index = tf.where(p_bool)[:, 0]

    y_true = tf.gather(target_masks, p_index)
    y_pred = tf.gather(pred_masks, p_index)




    loss = K.switch(tf.size(y_true) > 0,
                    _cross_loss(y_true=y_true, y_pred=y_pred,sub_class=sub_class),
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss


def mask_loss_face(target_masks_all, pred_masks):
    '''

    :param target_masks: [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,all_sub_class]
    :param pred_masks:  [batch_size,TRAIN_SUB_ROIS_PER_IMAGE,height,width,sub_class]
    :return:
    '''


    sub_class = 3


    y_true = target_masks_all
    y_pred = pred_masks




    loss = K.switch(tf.size(y_true) > 0,
                    _cross_loss(y_true=y_true, y_pred=y_pred,sub_class=sub_class),
                    tf.constant(0.0))

    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss






def _cross_loss(y_true,y_pred,sub_class):
    '''

    :param y_true:
    :param y_pred:
    :param sub_class:
    :return:
    '''
    epsilon = K.constant(1e-4)
    y_true = K.reshape(y_true,(-1,sub_class)) #[N, sub_class]
    y_pred = K.reshape(y_pred,(-1,sub_class))+epsilon

    loss = -tf.reduce_sum(y_true*K.log(y_pred),reduction_indices=[1])

    return loss


