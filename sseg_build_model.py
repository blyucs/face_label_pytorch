import keras.layers as KL
import keras.models as KM
import keras
import tensorflow as tf
from models import sseg_net
import sseg_build_graph
import sseg_loss
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,TensorBoard
import sseg_data_input2 as sseg_data_input
from math import ceil
from utils.ssd_box_encode_decode_utils import SSDBoxEncoder
from keras_layers.keras_layer_ImageROICut import ImageROICut
from keras_layers.keras_layer_ImageROICut_test import ImageROICut_test
from keras_layers.keras_layer_RoiSplit import RoiSplit,RoiSplit_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import common_utils as util
from sseg_config import Config
class SSDSEG():

    def __init__(self, mode, config):

        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config

        # class InferenceConfig(Config):
        #     BATCH_SIZE = 4
        self.test_config = Config()


        if mode == 'training':
            self.keras_model = self.build(config=config)
        else:
            self.keras_model = self.build_inference(config=config)


    def build(self,config):

        input_image = KL.Input(shape=[512,512,3], name="input_image")
        input_gt_boxc = KL.Input(shape=[None, config.n_classes+1 +4+4+4], name="input_gt_boxc", dtype=tf.float32)
        input_gt_mask = KL.Input(shape=[512,512,config.SUB_CLASS],name='input_gt_mask',dtype=tf.float32)
        input_face3_gt = KL.Input(shape=[512,512,3], name="input_face3_gt")

        # build ssd_det
        # return det_rois: [batch_size, top_k, (class_id,score,xmin,ymin,xmax,ymax)]
        det_rois,boxc_pred,share_feature,face_mask = sseg_net.snet(input_image=input_image,
                                         image_size=(config.img_height, config.img_width, config.img_channels),
                                         n_classes=config.n_classes,
                                         l2_regularization=0.0005,
                                         scales=config.scales,
                                         aspect_ratios_per_layer=config.aspect_ratios,
                                         two_boxes_for_ar1=config.two_boxes_for_ar1,
                                         steps=config.steps,
                                         offsets=config.offsets,
                                         limit_boxes=config.limit_boxes,
                                         variances=config.variances,
                                         coords=config.coords,
                                         normalize_coords=config.normalize_coords,
                                         subtract_mean=config.subtract_mean,
                                         divide_by_stddev=None,
                                         swap_channels=config.swap_channels,
                                         confidence_thresh=0.7,
                                         iou_threshold=0.5,
                                         top_k=config.TOPK,
                                         nms_max_output_size=100,
                                         min_scale = None,
                                         max_scale = None,
                                         )

        rois1,rois2,rois3,rois4,rois5 = RoiSplit(config=config)([det_rois])


        target_mask1 = sseg_build_graph.Target_cut(input_gt_mask,rois1,config)
        target_mask2 = sseg_build_graph.Target_cut(input_gt_mask,rois2,config)
        target_mask3 = sseg_build_graph.Target_cut(input_gt_mask,rois3,config)
        target_mask4 = sseg_build_graph.Target_cut(input_gt_mask,rois4,config)



        Submask_ebs = sseg_build_graph.SubMask_eb(config=config,numclass=3)
        pred_mask1 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois1])
        pred_mask2 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois2])
        pred_mask3 = sseg_build_graph.submask_nosenet(input_feature_map=share_feature,numclass=2,config=config,rois=rois3)
        pred_mask4 = sseg_build_graph.submask_mousenet(input_feature_map=share_feature,numclass=4,config=config,rois=rois4)


        mask_loss1 = KL.Lambda(lambda x:sseg_loss.mask_loss1(*x),
                              name='mask_loss1')([target_mask1,pred_mask1])
        mask_loss2 = KL.Lambda(lambda x:sseg_loss.mask_loss2(*x),
                              name='mask_loss2')([target_mask2,pred_mask2])
        mask_loss3 = KL.Lambda(lambda x:sseg_loss.mask_loss3(*x),
                              name='mask_loss3')([target_mask3,pred_mask3])
        mask_loss4 = KL.Lambda(lambda x:sseg_loss.mask_loss4(*x),
                              name='mask_loss4')([target_mask4,pred_mask4])

        mask_loss_face = KL.Lambda(lambda x:sseg_loss.mask_loss_face(*x),
                              name='mask_loss_face')([input_face3_gt,face_mask])


        boxc_loss = KL.Lambda(lambda x:sseg_loss.boxc_loss(*x),
                              name='boxc_loss')([input_gt_boxc,boxc_pred])



        inputs = [input_image,input_gt_boxc,input_gt_mask,input_face3_gt]
        outputs = [  boxc_loss ,mask_loss1,mask_loss2,mask_loss3,mask_loss4,mask_loss_face,
                     target_mask3,pred_mask3]

        model  = KM.Model(inputs=inputs,outputs=outputs,name='ssdseg_model')


        return model

    def compile(self,config):

        optimizer = keras.optimizers.Adam(lr=config.LEARNING_RATE)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ['boxc_loss','mask_loss1','mask_loss2','mask_loss3','mask_loss4','mask_loss_face']

        for name in loss_names:
            layer = self.keras_model.get_layer(name)

            if layer.output in self.keras_model.losses:
                continue

            self.keras_model.add_loss(layer.output)
                #tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.

        # reg_losses = [keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        #               for w in self.keras_model.trainable_weights
        #               if 'gamma' not in w.name and 'beta' not in w.name]
        # self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[
                                 None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))


    def train(self,config,epochs,weights_path,start_epoch,debug):


        if weights_path is not None:
            self.keras_model.load_weights(weights_path, by_name=True)
        self.compile(config=config)

        predictor_sizes = [self.keras_model.get_layer('classes1').output_shape[1:3],
                           self.keras_model.get_layer('classes2').output_shape[1:3],
                           self.keras_model.get_layer('classes3').output_shape[1:3],
                           self.keras_model.get_layer('classes4').output_shape[1:3],
                           self.keras_model.get_layer('classes5').output_shape[1:3],
                           self.keras_model.get_layer('classes6').output_shape[1:3],
                           self.keras_model.get_layer('classes7').output_shape[1:3]]

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
                                        normalize_coords=config.normalize_coords,
                                        )

        n_train_samples = 2000
        n_val_samples = 100
        train_generator = sseg_data_input.data_generator_helen(batch_size=config.BATCH_SIZE, shuffle=True,
                                                                ssd_box_encoder=ssd_box_encoder,
                                                                image_shape=[config.img_height, config.img_width],config=config,augment=True)
        test_generator = sseg_data_input.data_generator_helen(batch_size=self.test_config.BATCH_SIZE, shuffle=True,
                                                                    ssd_box_encoder=ssd_box_encoder,
                                                                    image_shape=[self.test_config.img_height, self.test_config.img_width],config=self.test_config,augment=True)

        if debug:
            for z in range(100):
                input_ = next(train_generator)
                l1,l2,l3,l4,l5,tmask,pred,x_toview = self.keras_model.predict(input_[0])

                image = np.uint8(input_[0][0][0,...])

                #print (rois1_n)
                # for t in range(len(rois1_n[0])):
                #     loc = rois1_n[0][t]
                #     loc = np.int32(loc)
                #     cv2.rectangle(image,(loc[0],loc[1]),(loc[2],loc[3]),color=[255,0,0],thickness=2)
                #     plt.imshow(image)
                #     plt.show()
                #
                # for t in range(len(rois2_n[0])):
                #     loc = rois2_n[0][t]
                #     loc = np.int32(loc)
                #     cv2.rectangle(image, (loc[0], loc[1]), (loc[2], loc[3]), color=[0, 255, 0], thickness=2)
                #     plt.imshow(image)
                #     plt.show()
                # for t in range(len(rois3_n[0])):
                #     loc = rois3_n[0][t]
                #     loc = np.int32(loc)
                #     cv2.rectangle(image, (loc[0], loc[1]), (loc[2], loc[3]), color=[0, 0, 255], thickness=2)
                #     plt.imshow(image)
                #     plt.show()
                # for t in range(len(rois4_n[0])):
                #     loc = rois4_n[0][t]
                #     loc = np.int32(loc)
                #     cv2.rectangle(image, (loc[0], loc[1]), (loc[2], loc[3]), color=[0, 0, 0], thickness=2)
                #     plt.imshow(image)
                #     plt.show()


                for i in range(1):

                    ms = np.argmax(tmask[0,i,:,:,:],axis=-1)
                    plt.imshow(ms)
                    plt.show()

                    for j in range(5):
                        plt.imshow(x_toview[0, i, :, :, j])
                        plt.show()


                    # fs = np.argmax(x_toview[0, i, :, :, :], axis=-1)
                    # plt.imshow(fs)
                    # plt.show()


                    ps = np.argmax(pred[0, i, :, :, :], axis=-1)
                    plt.imshow(ps)
                    plt.show()


        # this is the debug of the training process, to check the target mask is or not right , since the output the subnetwork is not right
        callbacks = [ModelCheckpoint('weights/ssdseg_weights_epoch-{epoch:02d}.h5',verbose=1,save_weights_only=True),
                     #TensorBoard('weights/',histogram_freq=0, write_graph=True, write_images=True)]#,
                     EarlyStopping(monitor='val_boxc_loss',min_delta=0.0001,patience=20)]

        self.keras_model.fit_generator(generator=train_generator,
                                      steps_per_epoch=ceil(n_train_samples / config.BATCH_SIZE),
                                      initial_epoch=start_epoch,
                                      epochs=epochs,
                                      callbacks=callbacks,
                                      validation_data=next(test_generator),
                                      validation_steps=ceil(n_val_samples / self.test_config.BATCH_SIZE)
                                      )





    def build_inference(self, config):

        input_image = KL.Input(shape=[512,512,3], name="input_image")


        # build ssd_det
        # return det_rois: [batch_size, top_k, (class_id,score,xmin,ymin,xmax,ymax)]
        det_rois,boxc_pred,share_feature,mask_face3 = sseg_net.snet(input_image=input_image,
                                         image_size=(config.img_height, config.img_width, config.img_channels),
                                         n_classes=config.n_classes,
                                         l2_regularization=0.0005,
                                         scales=config.scales,
                                         aspect_ratios_per_layer=config.aspect_ratios,
                                         two_boxes_for_ar1=config.two_boxes_for_ar1,
                                         steps=config.steps,
                                         offsets=config.offsets,
                                         limit_boxes=config.limit_boxes,
                                         variances=config.variances,
                                         coords=config.coords,
                                         normalize_coords=config.normalize_coords,
                                         subtract_mean=config.subtract_mean,
                                         divide_by_stddev=None,
                                         swap_channels=config.swap_channels,
                                         confidence_thresh=0.5,
                                         iou_threshold=0.1,
                                         top_k=config.TOPK,
                                         nms_max_output_size=15,
                                         min_scale=None,
                                         max_scale=None,
                                         )




        rois1,rois2,rois3,rois4,rois5,score1,score2,score3,score4,score5 = RoiSplit_score(config=config)([det_rois])


        pred_mask3 = sseg_build_graph.submask_nosenet(input_feature_map=share_feature,numclass=2,config=config,rois=rois3)
        pred_mask4,ss1,ss2,ss3 = sseg_build_graph.submask_mousenet(input_feature_map=share_feature,numclass=4,config=config,rois=rois4)
        Submask_ebs = sseg_build_graph.SubMask_eb(config=config,numclass=3)
        pred_mask1 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois1])
        pred_mask2 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois2])



        inputs = [input_image]
        outputs = [ pred_mask1,pred_mask2,pred_mask3,pred_mask4,rois1,rois2,rois3,rois4,rois5,
                    score1, score2, score3, score4, score5,mask_face3,share_feature[2],ss1,ss2,ss3 ]

        model  = KM.Model(inputs=inputs,outputs=outputs,name='ssdseg_model')


        return model

    def inference(self,input_images):
        model = self.keras_model


        p1,p2,p3,p4,r1,r2,r3,r4,r5,s1,s2,s3,s4,s5,mface,sf,st1,st2,st3 = model.predict(input_images)


        full_masks1, sub_roi1, class_id1,ss1 = util.restore_submask(sub_rois=r1[0], sub_score=s1[0],sub_masks=p1[0],
                                                                image_shape=self.config.IMAGE_SHAPE, sub_class=1)

        full_masks2, sub_roi2, class_id2,ss2 = util.restore_submask(sub_rois=r2[0], sub_score=s2[0],sub_masks=p2[0],
                                                                image_shape=self.config.IMAGE_SHAPE, sub_class=2)

        full_masks3, sub_roi3, class_id3,ss3 = util.restore_submask(sub_rois=r3[0], sub_score=s3[0],sub_masks=p3[0],
                                                                image_shape=self.config.IMAGE_SHAPE, sub_class=3)

        full_masks4, sub_roi4, class_id4,ss4 = util.restore_submask(sub_rois=r4[0], sub_score=s4[0],sub_masks=p4[0],
                                                                image_shape=self.config.IMAGE_SHAPE, sub_class=4)

        full_masks5, sub_roi5, class_id5,ss5 = util.restore_submask_face(sub_rois=r5[0],sub_score=s5[0],image_shape=self.config.IMAGE_SHAPE, sub_class=5)


        final_masks = []
        final_maskst = [full_masks1,full_masks2,full_masks3,full_masks4,full_masks5]
        for t in final_maskst:
            if t.shape[0]>0:
                final_masks.append(t)

        if len(final_masks)==0:
            return None
        final_masks = np.concatenate(final_masks,axis=-1)
        final_rois = np.concatenate([sub_roi1,sub_roi2,sub_roi3,sub_roi4,sub_roi5],axis=0)
        final_class_ids = np.concatenate([class_id1,class_id2,class_id3,class_id4,class_id5],axis=0)
        final_score = np.concatenate([ss1, ss2, ss3, ss4, ss5], axis=0)

        import scipy.misc as misc
        p4t = st1
        for j in range(5):
            plt.imshow(p4t[0,0,:,:,j])
            plt.axis('off')
            plt.savefig('ex_temp_img2/1_'+str(j)+'.png')


        p4t = st2
        for j in range(5):
            plt.imshow(p4t[0,0,:,:,j])
            plt.axis('off')
            plt.savefig('ex_temp_img2/2_'+str(j)+'.png')


        p4t = st3
        for j in range(5):
            plt.imshow(p4t[0,0,:,:,j])
            plt.axis('off')
            plt.savefig('ex_temp_img2/3_'+str(j)+'.png')


        # plt.imshow(p4t)
        # plt.axis('off')
        # plt.savefig('p4.png')
        #
        # plt.show()

        resutls = {
            'rois':final_rois,
            'class_ids':final_class_ids,
            'masks':final_masks,
            'face_mask':mface,
            "score":final_score
        }
        return resutls,sf







