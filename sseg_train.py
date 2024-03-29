from sseg_config import Config
import sseg_build_model2
import os
import sseg_build_model
import sseg_data_input2
import  sseg_build_graph
import torchvision.transforms.Lambda as L
import sseg_loss
from keras_layer_RoiSplit import RoiSplit
from ssd_box_encode_decode_utils import  SSDBoxEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':

    # Configurations
    config = Config()

    model = sseg_build_model2.SSDSEG(mode="training", config=config)

    # weights_path = '/mnt/sda1/don/documents/ssd_face/ss_face2_l3_s2_focal/ssdseg_weights_epoch-10.h5'
    # model.train(config=config,epochs=50,weights_path=weights_path,start_epoch=0,debug=False)


#    weights_path = '/mnt/sda1/don/documents/ssd_face/ss_face2_l3_s2_focal/weights/ssdseg_weights_epoch-70.h5'
 #   model.train(config=config2,epochs=120,weights_path=weights_path,start_epoch=70,debug=False)

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

    train_generator = sseg_data_input2.data_generator_helen(batch_size=config.BATCH_SIZE, shuffle=True,
                                                           ssd_box_encoder=ssd_box_encoder,
                                                           image_shape=[config.img_height, config.img_width],
                                                           config=config, augment=True)
    test_generator = sseg_data_input2.data_generator_helen(batch_size=self.test_config.BATCH_SIZE, shuffle=True,
                                                          ssd_box_encoder=ssd_box_encoder,
                                                          image_shape=[self.test_config.img_height,
                                                                       self.test_config.img_width],
                                                          config=self.test_config, augment=True)

    for i in 10:
        model.train()
        input_ = next(train_generator) #input =[batch_gt_images, batch_boxc_true, batch_gt_mask,batch_gt_mask_face3]
        input_gt_boxc = input_[1]
        input_gt_mask = input_[2]
        input_face3_gt = input_[3]

        pred_mask1,pred_mask2,pred_mask3,pred_mask4,face_mask,boxc_pred,det_rois = model(input_[0])

        rois1,rois2,rois3,rois4,rois5 = RoiSplit(config=config)([det_rois])

        target_mask1 = sseg_build_graph.Target_cut(input_gt_mask,rois1,config)
        target_mask2 = sseg_build_graph.Target_cut(input_gt_mask,rois2,config)
        target_mask3 = sseg_build_graph.Target_cut(input_gt_mask,rois3,config)
        target_mask4 = sseg_build_graph.Target_cut(input_gt_mask,rois4,config)

        mask_loss1 = L.Lambda(lambda x:sseg_loss.mask_loss1(*x),
                              name='mask_loss1')([target_mask1,pred_mask1])
        mask_loss2 = L.Lambda(lambda x:sseg_loss.mask_loss2(*x),
                              name='mask_loss2')([target_mask2,pred_mask2])
        mask_loss3 = L.Lambda(lambda x:sseg_loss.mask_loss3(*x),
                              name='mask_loss3')([target_mask3,pred_mask3])
        mask_loss4 = L.Lambda(lambda x:sseg_loss.mask_loss4(*x),
                              name='mask_loss4')([target_mask4,pred_mask4])

        mask_loss_face = L.Lambda(lambda x:sseg_loss.mask_loss_face(*x),
                                  name='mask_loss_face')([input_face3_gt,face_mask])

        boxc_loss = L.Lambda(lambda x:sseg_loss.boxc_loss_focal(*x),
                             name='boxc_loss')([input_gt_boxc,boxc_pred])

#  loss function
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    loss = [mask_loss1,mask_loss2,mask_loss3,mask_loss4,mask_loss_face,boxc_loss]

    loss.backward()

    optimizer.step()







