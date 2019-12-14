import numpy as np
import scipy.misc as misc
import cv2
import matplotlib.pyplot as plt
def data_generator_helen(batch_size, ssd_box_encoder, image_shape, shuffle, augment, config):

    f1 = open('../data/helen/helen_point_train.txt')
    train_list = [line.split()[0] for line in f1.readlines()]
    f1.close()

    b = 0  # batch item index
    batch_gt_images = np.zeros((batch_size, image_shape[0], image_shape[1], 3))
    batch_gt_mask = np.zeros((batch_size, image_shape[0], image_shape[1], config.SUB_CLASS))
    batch_gt_mask_face3 = np.zeros(
        (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
    batch_gt_boxc = []
    image_index = -1
    image_ids = np.arange(0, len(train_list), 1, np.int32)

    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image_name = train_list[image_id]

            image, gt_box, gt_mask,mask_face3 = load_gt(image_name, augment, config)

            if b == 0:
                batch_gt_images = np.zeros((batch_size, image_shape[0], image_shape[1], 3))
                batch_gt_mask = np.zeros((batch_size, image_shape[0], image_shape[1], config.SUB_CLASS))
                batch_gt_mask_face3 = np.zeros(
                    (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
                batch_gt_boxc = []

            batch_gt_boxc.append(gt_box)
            batch_gt_images[b] = image
            batch_gt_mask[b] = gt_mask
            batch_gt_mask_face3[b] = mask_face3
            b += 1

            if b >= batch_size:
                batch_boxc_true = ssd_box_encoder.encode_y(batch_gt_boxc, diagnostics=False)

                inputs = [batch_gt_images, batch_boxc_true, batch_gt_mask,batch_gt_mask_face3]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise


def data_generator_helen_test(batch_size, ssd_box_encoder, image_shape, shuffle, augment, config):

    f1 = open('../data/helen/helen_point_test.txt')
    train_list = [line.split()[0] for line in f1.readlines()]
    f1.close()

    b = 0  # batch item index
    batch_gt_images = np.zeros((batch_size, image_shape[0], image_shape[1], 3))
    batch_gt_mask = np.zeros((batch_size, image_shape[0], image_shape[1], config.SUB_CLASS))
    batch_gt_mask_face3 = np.zeros(
        (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
    batch_gt_boxc = []
    image_index = -1
    image_ids = np.arange(0, len(train_list), 1, np.int32)

    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image_name = train_list[image_id]

            image, gt_box, gt_mask,mask_face3 = load_gt(image_name, augment, config)

            if b == 0:
                batch_gt_images = np.zeros((batch_size, image_shape[0], image_shape[1], 3))
                batch_gt_mask = np.zeros((batch_size, image_shape[0], image_shape[1], config.SUB_CLASS))
                batch_gt_mask_face3 = np.zeros(
                    (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
                batch_gt_boxc = []

            batch_gt_boxc.append(gt_box)
            batch_gt_images[b] = image
            batch_gt_mask[b] = gt_mask
            batch_gt_mask_face3[b] = mask_face3
            b += 1

            if b >= batch_size:
                batch_boxc_true = ssd_box_encoder.encode_y(batch_gt_boxc, diagnostics=False)

                inputs = [batch_gt_images, batch_boxc_true, batch_gt_mask,batch_gt_mask_face3]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise


def load_gt(image_name,augment,config):




    img_dim = str(config.IMAGE_MIN_DIM)
    data_path = '../data/helen/'
    image = misc.imread(data_path+'images_'+img_dim+'/'+image_name+'.jpg')


    mask  = np.zeros([image.shape[0],image.shape[1],config.SUB_CLASS],dtype=np.uint8)

    for i in range(config.SUB_CLASS-2):
        mask[:, :, i] = misc.imread(
            data_path + 'labels_'+img_dim+'/' + image_name + '/' + image_name + '_lbl0' + str(i + 2) + '.jpg')/128


    mask_face_ori = np.uint8(misc.imread(data_path+'labels_'+img_dim+'/'+image_name+'/'+image_name+'_lbl01.jpg')/128)

    mask[:,:,-2] = 1*(mask[:, :, 0] | mask[:, :, 1] | mask[:, :, 2] | mask[:, :, 3] | mask[:, :, 4] | mask[:, :, 5] | mask[:, :, 6] | mask[:,:, 7]|mask_face_ori)
    mask[:, :, -1] = 1 - (mask[:,:,8])

    mask_face3 = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
    mask_face3[:,:,0] = np.copy(mask[:,:,-2])
    mask_face3[:,:,1] = misc.imread(
            data_path + 'labels_'+img_dim+'/' + image_name + '/' + image_name + '_lbl10.jpg')/128
    mask_face3[:,:,2] = misc.imread(
            data_path + 'labels_'+img_dim+'/' + image_name + '/' + image_name + '_lbl00.jpg')/128

    if augment:
        image, mask, mask_face3 = data_augment(image, mask, mask_face3)

    mask_to_box = np.ones([mask.shape[0], mask.shape[1], 5], dtype=np.uint8)
    mask_to_box[:, :, 0] = mask[:, :, 0] | mask[:, :, 2]
    mask_to_box[:, :, 1] = mask[:, :, 1] | mask[:, :, 3]
    mask_to_box[:, :, 2] = mask[:, :, 4]
    mask_to_box[:, :, 3] = mask[:, :, 5] | mask[:, :, 6] | mask[:, :, 7]
    mask_to_box[:, :, 4] = mask[:, :, 8]

    bbox = extract_bboxes_expand(mask_to_box, ex_factor=0.3)

    # for i in range(5):
    #     cv2.rectangle(image, (bbox[i,1],bbox[i,2]), (bbox[i,3],bbox[i,4]), color=[255, 0, 0], thickness=2)
    #     plt.imshow(np.uint8(image))
    #     plt.imshow(mask_to_box[:,:,i],alpha=0.5)
    #     plt.show()

    return image, bbox, mask,mask_face3


def data_augment(img,mask,mask_face3):

    img_ori = img.copy()
    mask_ori = mask.copy()
    mask_face3_ori = mask_face3.copy()
    h,w,c = img_ori.shape

    if np.random.uniform(0., 1.) > 0.5:
        img_ori = cv2.flip(img_ori,1)

        mask_ori = cv2.flip(mask_ori,1)
        lbt = np.copy(mask_ori[:,:,0])
        let = np.copy(mask_ori[:,:,2])
        mask_ori[:,:,0] = mask_ori[:,:,1]
        mask_ori[:,:,2] = mask_ori[:,:,3]
        mask_ori[:,:,1] = lbt
        mask_ori[:,:,3] = let
        mask_face3_ori = cv2.flip(mask_face3_ori,1)


    # argue param
    angle = np.random.randint(-30,30)
    scale = np.random.uniform(0.8, 1.4)
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,scale)

    x_shift = np.random.randint(-50,50)
    y_shift = np.random.randint(-50,50)
    M2 = np.float32([[1,0,x_shift],[0,1,y_shift]])

    img_a = cv2.warpAffine(img_ori,M,(w,h))
    mask_a = cv2.warpAffine(mask_ori,M,(w,h))
    img_a = cv2.warpAffine(img_a,M2,(w,h))
    mask_a = cv2.warpAffine(mask_a,M2,(w,h))

    mask_a[:, :, -1] = 1 - (mask_a[:, :, 8])

    mask_face3_a = cv2.warpAffine(mask_face3_ori,M,(w,h))
    mask_face3_a = cv2.warpAffine(mask_face3_a,M2,(w,h))
    mask_face3_a[:,:,-1] = 1-(mask_face3_a[:,:,0]|mask_face3_a[:,:,1])

    # bbox_ori = bbox.copy()
    # mask_b = np.zeros([h,w,5],dtype=np.uint8)
    # for i in range(5):
    #     boxi = bbox_ori[i]
    #     mask_b[boxi[0]:boxi[2],boxi[1]:boxi[3],i]=1
    # mask_b_a = cv2.warpAffine(mask_b,M,(w,h))
    # bbox_a = utils.extract_bboxes(mask_b_a)

    return img_a,mask_a,mask_face3_a

def extract_bboxes_expand(mask,ex_factor):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 5], dtype=np.int32)
    for i in range(mask.shape[-1]):

        if i==0 or i==1:
            ex_factorx=0.3
        elif i==2:
            ex_factorx=0.2
        elif i==3:
            ex_factorx=0.25
        else:
            ex_factorx=0.1

        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            # x2 += 1
            # y2 += 1

            x1 -= np.int32((x2-x1)*ex_factorx)
            x1 = max(0,x1)

            x2 += np.int32((x2-x1)*ex_factorx)
            x2 = min(m.shape[1],x2)

            y1 -= np.int32((y2 - y1) * ex_factorx)
            y1 = max(0, y1)

            y2 += np.int32((y2 - y1) * ex_factorx)
            y2 = min(m.shape[0], y2)


        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([i+1,x1, y1, x2, y2])
    return boxes.astype(np.int32)
