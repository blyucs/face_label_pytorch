from sseg_config import Config
import sseg_build_model
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
from utils import visualize
from utils import common_utils as util
import time
import os
import acc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class_names = ['00', 'eb1', 'eb2', 'nose', 'mouse','face']

if __name__ == '__main__':

    # Configurations
    class InferenceConfig(Config):
        BATCH_SIZE = 1


    config = InferenceConfig()

    model_infer = sseg_build_model.SSDSEG(mode="inference", config=config)
    weight_path = '/mnt/sda1/don/documents/ssd_face/ss_face2_l3_f/weights/ssdseg_weights_epoch-69.h5'
    model = model_infer.keras_model
    model.load_weights(weight_path, by_name=True)

    test_list = open('../data/helen/test.txt').readlines()

    acca = 0
    acc_bg = 0
    acc_bl = 0
    acc_br = 0
    acc_el = 0
    acc_er = 0
    acc_nose = 0
    acc_lup = 0
    acc_mi = 0
    acc_ll = 0

    acc_mouth = 0
    acc_brow = 0
    acc_eye = 0
    acc_face = 0

    num_test = 0
    data_path = '../data/helen/'
    for i in range(len(test_list)):
        line = test_list[i]
        name = line.split()[2]
        image = misc.imread('../data/helen/images_512/' + name + '.jpg')

        mask = np.zeros([512, 512, 11], dtype=np.uint8)
        mask[:, :, 0] = misc.imread(data_path + 'labels_512/' + name + '/' + name + '_lbl00.jpg') / 128
        for i in range(1, 9):
            mask[:, :, i] = misc.imread(
                data_path + 'labels_512/' + name + '/' + name + '_lbl0' + str(i + 1) + '.jpg') / 128
        mask[:, :, 9] = misc.imread(data_path + 'labels_512/' + name + '/' + name + '_lbl01.jpg') / 128
        mask[:, :, 10] = misc.imread(data_path + 'labels_512/' + name + '/' + name + '_lbl10.jpg') / 128

        image = np.expand_dims(image,axis=0)
        t1 = time.time()
        results = model_infer.inference(image)

        results = util.filter_results(results)
        r = results

        face_box = r['rois'][-1,:]
        face_box = np.int32(face_box)
        fx1,fy1,fx2,fy2 = face_box
        face_mask = np.zeros([512,512],dtype=np.float32)
        face_mask[fy1:fy2,fx1:fx2] = 1.0

        print(time.time()-t1)
        if results==None:
            continue
        if len(r['class_ids']) == 5:
            num_test = num_test + 1

            mask_face3_out = np.zeros([512, 512, 3], dtype=np.uint8)
            mask_face3 = r['face_mask'][0]
            mask_face3[:, :, 0] = mask_face3[:, :, 0] * face_mask
            amax = np.amax(mask_face3, 2)
            for i in range(3):
                maskt = mask_face3[:, :, i] - amax
                mask_face3_out[:, :, i] = np.where(maskt >= 0.0, 1, 0).astype(np.uint8)


            # [bg,8,face,hair]
            result_pred = visualize.display_full_face_toacc(image=image[0], boxes=r['rois'], masks=r['masks'],
                                                            class_ids=r['class_ids'],
                                                            scores=r['score'],
                                                            mask_face3=mask_face3_out)

            accta, acct_bg, acct_bl, acct_br, acct_el, acct_er, acct_nose, acct_lup, acct_mi, acct_ll, acct_mouth, acct_brow, acct_eye, acct_face \
                = acc.acc_eval11(eval_images=result_pred, labels=mask)
            #
            #
            #
            #
            #
            acca += accta
            acc_bg += acct_bg
            acc_bl += acct_bl
            acc_br += acct_br
            acc_el += acct_el
            acc_er += acct_er
            acc_nose += acct_nose
            acc_lup += acct_lup
            acc_mi += acct_mi
            acc_ll += acct_ll

            acc_mouth += acct_mouth
            acc_brow += acct_brow
            acc_eye += acct_eye
            acc_face += acct_face

            #
            #
            #
            #     # print (accta)
            #     # print(acct1)
            #     # print(acct2)
            #     # print(acct3)
            #     # print(acct4)
            #     # print(acct5)
            #     # print(acct6)
            #     # print(acct7)
            #     # print(acct8)
            #     # print(acct9)
            #     # print(acct_mouth)
            #     # print(acct_brow)
            #     # print(acct_eye)
            #
            #
            #
            #print('bg: ', acc_bg / num_test)

            #print('b_l: ', acc_bl / num_test)
            #print('b_r: ', acc_br / num_test)
            #print('e_l: ', acc_el / num_test)
            #print('e_r: ', acc_er / num_test)
            print("eye: ", acc_eye / num_test)
            print('brow: ', acc_brow / num_test)
            print('nose: ', acc_nose / num_test)
            print('mouth_in: ', acc_mi / num_test)
            print('lip_u: ', acc_lup / num_test)
            print('lip_l: ', acc_ll / num_test)
            print('mouth_all: ', acc_mouth / num_test)

            print('face: ', acc_face / num_test)
            print('overall: ', acca / num_test)

            print('num_test: ', num_test)






