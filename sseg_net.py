from __future__ import division
import numpy as np


from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2
import keras.layers as KL
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections

def snet(input_image,
            image_size,
            n_classes,
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=None,
            two_boxes_for_ar1=True,
            steps=None,
            offsets=None,
            limit_boxes=False,
            variances=None,
            coords='centroids',
            normalize_coords=False,
            subtract_mean=None,
            divide_by_stddev=None,
            swap_channels=True,
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400):

    n_predictor_layers = 7 # The number of predictor conv layers in the network is 7 for the original SSD512
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Build the network.
    ############################################################################


    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(lambda z: z, output_shape=(img_height, img_width, img_channels), name='identity_layer')(input_image)
    if not (subtract_mean is None):
        x1 = Lambda(lambda z: z - np.array(subtract_mean), output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(lambda z: z / np.array(divide_by_stddev), output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels and (img_channels == 3):
        x1 = Lambda(lambda z: z[...,::-1], output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)



    C1,C2,C3,C4,C5 = resnet_graph_18(x1,stage5=True)

    PC1 = KL.Conv2D(64, (3, 3), activation='relu', padding="SAME", name="pc1")(C3) #64*64*64
    PC2 = KL.Conv2D(64, (3, 3), activation='relu', padding="SAME", name="pc2")(C4) #32*32*64
    PC3 = KL.Conv2D(64, (3, 3), activation='relu', padding="SAME", name="pc3")(C5) #16*16*64
    PC4 = KL.Conv2D(64, (3, 3), activation='relu',strides=(2,2), padding="SAME", name="pc4")(PC3) #8*8
    PC5 = KL.Conv2D(64, (3, 3), activation='relu',strides=(2,2), padding="SAME", name="pc5")(PC4)  # 4*4
    PC6 = KL.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding="SAME", name="pc6")(PC5)  # 2*2
    PC7 = KL.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding="SAME", name="pc7")(PC6)  # 1*1





    # P5 = KL.Conv2D(64, (1, 1), name='fpn_c5p5')(C5)
    # P4 = KL.Add(name="fpn_p4add")([
    #     KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
    #     KL.Conv2D(64, (1, 1), name='fpn_c4p4')(C4)])
    # P3 = KL.Add(name="fpn_p3add")([
    #     KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
    #     KL.Conv2D(64, (1, 1), name='fpn_c3p3')(C3)])
    # P2 = KL.Add(name="fpn_p2add")([
    #     KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
    #     KL.Conv2D(64, (1, 1), name='fpn_c2p2')(C2)])
    # PF =P1 = KL.Add(name="fpn_p1add")([
    #     KL.UpSampling2D(size=(2, 2), name="fpn_p2upsampled")(P2),
    #     KL.Conv2D(64, (1, 1), name='fpn_c1p1')(C1)])
    #


    P5 = KL.Conv2D(64, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p5transpose")(P5),
        KL.Conv2D(64, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p4transpose")(P4),
        KL.Conv2D(64, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p3transpose")(P3),
        KL.Conv2D(64, (1, 1), name='fpn_c2p2')(C2)])
    PF =P1 = KL.Add(name="fpn_p1add")([
        KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p2transpose")(P2),
        KL.Conv2D(64, (1, 1), name='fpn_c1p1')(C1)])


    # Attach 3x3 conv to all P layers to get the final feature maps.
    P1 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p1")(P1) #256*256
    P2 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p2")(P2) #128*128
    P3 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p3")(P3) #64*64
    #P4 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p4")(P4) #32*32

    PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF1", activation='relu')(PF)
    PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF2", activation='relu')(PF)
    PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF3", activation='relu')(PF)
    PF = KL.Conv2DTranspose(64, (2, 2), strides=2, name='DPF1', activation="relu")(PF)
    PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF4", activation='relu')(PF)
    PF = KL.Conv2D(3, (1, 1), padding="SAME", name="PF_out", activation='softmax')(PF)




    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes1 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes1')(PC1) #64*64
    classes2 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes2')(PC2) #32*32
    classes3 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes3')(PC3) #16*16
    classes4 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(PC4) #8*8
    classes5 = Conv2D(n_boxes[4] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(PC5) #4*4
    classes6 = Conv2D(n_boxes[5] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes6')(PC6)  # 2*2
    classes7 = Conv2D(n_boxes[6] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes7')(PC7)  # 1*1
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes1 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes1')(PC1)
    boxes2 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes2')(PC2)
    boxes3 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes3')(PC3)
    boxes4 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(PC4)
    boxes5 = Conv2D(n_boxes[4] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(PC5)
    boxes6 = Conv2D(n_boxes[5] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes6')(PC6)
    boxes7 = Conv2D(n_boxes[6] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes7')(PC7)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors1')(boxes1)
    anchors2 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors2')(boxes2)
    anchors3 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors3')(boxes3)
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[6], next_scale=scales[7], aspect_ratios=aspect_ratios[6],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[6], this_offsets=offsets[6],
                           limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes1_reshaped = Reshape((-1, n_classes), name='classes1_reshape')(classes1)
    classes2_reshaped = Reshape((-1, n_classes), name='classes2_reshape')(classes2)
    classes3_reshaped = Reshape((-1, n_classes), name='classes3_reshape')(classes3)
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes1_reshaped = Reshape((-1, 4), name='boxes1_reshape')(boxes1)
    boxes2_reshaped = Reshape((-1, 4), name='boxes2_reshape')(boxes2)
    boxes3_reshaped = Reshape((-1, 4), name='boxes3_reshape')(boxes3)
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors1_reshaped = Reshape((-1, 8), name='anchors1_reshape')(anchors1)
    anchors2_reshaped = Reshape((-1, 8), name='anchors2_reshape')(anchors2)
    anchors3_reshaped = Reshape((-1, 8), name='anchors3_reshape')(anchors3)
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes1_reshaped,
                                                                 classes2_reshaped,
                                                                 classes3_reshaped,
                                                                 classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes1_reshaped,
                                                             boxes2_reshaped,
                                                             boxes3_reshaped,
                                                             boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors1_reshaped,
                                                                 anchors2_reshaped,
                                                                 anchors3_reshaped,
                                                                 anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           nms_max_output_size=nms_max_output_size,
                                           coords=coords,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           name='decoded_predictions')(predictions)

    #
    # share_feature1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                         kernel_regularizer=l2(l2_reg), name='share_conv1')(pool1)
    # share_feature2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                         kernel_regularizer=l2(l2_reg), name='share_conv2')(pool2)
    # share_feature3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                         kernel_regularizer=l2(l2_reg), name='share_conv3')(pool3)
    # share_feature4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
    #                         kernel_regularizer=l2(l2_reg), name='share_conv4')(pool4)
    #
    # share_features = [share_feature1,share_feature2,share_feature3,share_feature4]
    share_features = [P1,P2,P3]
    return decoded_predictions,predictions,share_features,PF



def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):

    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'


    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)

    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)



    shortcut = KL.Conv2D(nb_filter2, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x



def identity_block(input_tensor, kernel_size, filters,stage, block,
                   use_bias=True):

    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'


    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)

    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)



    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def resnet_graph_18(input_image, stage5=False):

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(32, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    C1 = x = KL.Activation('relu')(x)


    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [32, 32], stage=2, block='a',strides=(1,1))
    C2 = x = identity_block(x, 3, [32, 32], stage=2, block='b')

    # Stage 3
    x = conv_block(x, 3, [64, 64], stage=3, block='a')
    C3 = x = identity_block(x, 3, [64, 64], stage=3, block='b')
    # Stage 4
    x = conv_block(x, 3, [128, 128], stage=4, block='a')
    C4 = x = identity_block(x, 3, [128, 128], stage=4, block='b')

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [256, 256], stage=5, block='a')
        C5 = x = identity_block(x, 3, [256, 256], stage=5, block='b')
    else:
        C5 = None
    return C1, C2, C3, C4, C5