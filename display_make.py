from sseg_config import Config
import sseg_build_model
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
from utils import visualize
from utils import common_utils as util
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class_names = ['00', 'EB_left', 'EB_right', 'Nose', 'Mouse','Face']

if __name__ == '__main__':

    # Configurations
    class InferenceConfig(Config):
        BATCH_SIZE = 1


    config = InferenceConfig()

    model_infer = sseg_build_model.SSDSEG(mode="inference", config=config)
    weight_path = '/mnt/sda1/don/documents/ssd_face/ex_final/weights/ssdseg_weights_epoch-70.h5'
    model = model_infer.keras_model
    model.load_weights(weight_path, by_name=True)

    image = misc.imread('dis_img/1030333538_1.jpg')


    image = np.expand_dims(image,axis=0)
    t1 = time.time()
    results = model_infer.inference(image)

    results = util.filter_results(results)


    r =results
    face_box = r['rois'][-1,:]
    face_box = np.int32(face_box)
    fx1,fy1,fx2,fy2 = face_box
    face_mask = np.zeros([512,512],dtype=np.float32)
    face_mask[fy1:fy2,fx1:fx2] = 1.0


    mask_face = results['face_mask'][0]
    mask_face[:, :, 0] = mask_face[:, :, 0] * face_mask

    face_pred = np.argmax(mask_face,axis=-1)
    face3_view = util.view_label3(face_pred)
    misc.imsave('ex_temp_img/1030333538_1.jpg',face3_view)
    # plt.imshow(face3_view)
    # plt.show()
    print(time.time()-t1)


    mask_face3_out = np.zeros([512, 512, 3], dtype=np.uint8)
    amax = np.amax(mask_face, 2)
    for i in range(3):
        maskt = mask_face[:, :, i] - amax
        mask_face3_out[:, :, i] = np.where(maskt >= 0.0, 1, 0).astype(np.uint8)

    visualize.display_instances_class(image[0], results['rois'], results['masks'], results['class_ids'],class_names,None)
    # result_pred,full_view = visualize.display_full_face_show(image=image[0], boxes=r['rois'], masks=r['masks'],
    #                                                 class_ids=r['class_ids'],
    #                                                 scores=r['score'],
    #                                                 mask_face3=mask_face3_out)
    #
    # misc.imsave('ex_temp_img/1030333538_1_full2.jpg', full_view)
    # plt.imshow(full_view)
    # plt.show()





