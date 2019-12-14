import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import keras.backend as K
from keras_layers.keras_layer_ROIAlign import ROIAlign,ROIAlign_Target



def SubMask_eb(config,numclass):
    input_feature_map1 = KL.Input(shape=[None, None, 64],name='input_feature_eb1')
    input_feature_map2 = KL.Input(shape=[None, None, 64], name='input_feature_eb2')
    input_feature_map3 = KL.Input(shape=[None, None, 64], name='input_feature_eb3')


    rois = KL.Input(shape=[None,4])
    input_feature_map = [input_feature_map1,input_feature_map2,input_feature_map3]
    outputs = submask_ebnet(input_feature_map=input_feature_map,rois=rois,numclass=numclass,config=config)

    return KM.Model([input_feature_map1,input_feature_map2,input_feature_map3,rois],outputs,name='submask_ebn2')



def submask_ebnet(input_feature_map,rois,numclass,config):
    subFeature1 = ROIAlign(config=config)([input_feature_map[0], rois])
    subFeature2 = ROIAlign(config=config)([input_feature_map[1], rois])
    subFeature3 = ROIAlign(config=config)([input_feature_map[2], rois])


    x = KL.Concatenate(axis=-1)([subFeature1,subFeature2,subFeature3])
    x = KL.TimeDistributed(KL.Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'), name='1rdim_convn')(x)

    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='1conv3')(x)
    x = KL.TimeDistributed(KL.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
                           name='1conv4')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='1dconv4')(x)
    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='1conv5')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='1dconv6')(x)

    out = KL.TimeDistributed(KL.Conv2D(numclass,(1,1),strides=1,activation='softmax'),name='1submask_out')(x)

    return out


def submask_nosenet(input_feature_map,rois,numclass,config):

    subFeature1 = ROIAlign(config=config)([input_feature_map[0], rois])
    subFeature2 = ROIAlign(config=config)([input_feature_map[1], rois])
    subFeature3 = ROIAlign(config=config)([input_feature_map[2], rois])


    x = KL.Concatenate(axis=-1)([subFeature1,subFeature2,subFeature3])


    x = KL.TimeDistributed(KL.Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'),
                           name='2rdim_convn')(x)


    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='2conv3')(x)
    x = KL.TimeDistributed(KL.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
                           name='2conv4')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='2dconv4')(x)
    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='2conv5')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='2dconv6')(x)

    out = KL.TimeDistributed(KL.Conv2D(numclass,(1,1),strides=1,activation='softmax'),name='2submask_out')(x)

    return out

def submask_mousenet(input_feature_map,rois,numclass,config):

    subFeature1 = ROIAlign(config=config)([input_feature_map[0], rois])
    subFeature2 = ROIAlign(config=config)([input_feature_map[1], rois])
    subFeature3 = ROIAlign(config=config)([input_feature_map[2], rois])


    x = KL.Concatenate(axis=-1)([subFeature1,subFeature2,subFeature3])


    x = KL.TimeDistributed(KL.Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal'),
                           name='3rdim_convn')(x)


    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='3conv3')(x)
    x = KL.TimeDistributed(KL.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
                           name='3conv4')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='3dconv4')(x)
    x = KL.TimeDistributed(KL.Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer='he_normal'),name='3conv5')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(64,(2,2),strides=2,activation='relu', kernel_initializer='he_normal'),name='3dconv6')(x)

    out = KL.TimeDistributed(KL.Conv2D(numclass,(1,1),strides=1,activation='softmax'),name='3submask_out')(x)

    return out,subFeature1,subFeature2,subFeature3



def Target_cut(input_gt_mask, rois,config):
    subGT = ROIAlign_Target(config=config)([input_gt_mask, rois])

    return subGT











