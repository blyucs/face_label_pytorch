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
class_names = ['00', 'eb1', 'eb2', 'nose', 'mouse','face']

if __name__ == '__main__':

    # Configurations
    class InferenceConfig(Config):
        BATCH_SIZE = 1


    config = InferenceConfig()

    model_infer = sseg_build_model.SSDSEG(mode="inference", config=config)
    weight_path = '/mnt/sda1/don/documents/ssd_face/ex_final/weights/ssdseg_weights_epoch-70.h5'
    model = model_infer.keras_model
    model.load_weights(weight_path, by_name=True)

    test_list = open('../data/helen/test.txt').readlines()
    for i in range(5,len(test_list)):
        line = test_list[i]
        name = line.split()[2]
        image = misc.imread('../data/helen/images_512/' + name + '.jpg')


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

        plt.imshow(image[0])
        plt.imshow(np.argmax(mask_face,axis=-1),alpha=0.5)
        plt.show()
        print(time.time()-t1)
        if results==None:
            continue
        visualize.display_instances_class(image[0], results['rois'], results['masks'], results['class_ids'],class_names,results['score'])






