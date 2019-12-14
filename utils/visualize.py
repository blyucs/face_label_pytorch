
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display
import cv2
from utils import common_utils as util









def display_instances_class(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    #height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        class_id = class_ids[i]
        # if (class_id!=4):
        #     continue
        color = (0,1.,0)
        if class_id!=4:
            continue
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        p = patches.Rectangle((x1-1, y1-1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label

        score = scores[i] if scores is not None else None
        label = class_names[class_id]

        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 - 8, caption,
                color='w', size=15, backgroundcolor="none")


        # Mask
        mask = masks[:, :, :,i]
        #sub_colors = random_colors(9)
        sub_colors = [[0.0,0.66,1.0],
                      [0.0,0.0,1.0],
                      [1.0,0.0,0.0],
                      [0.0,1.0,0.0],
                      [1.0,0.66,0.0],
                      [0.0,1.0,0.66],
                      [0.66,0.0,1.0],
                      [1.0,0.0,0.66],
                      [0.0,0.0,0.0]]
        for x in range(8):
            sub_color = sub_colors[x]
            maskt = mask[:,:,x]
            masked_image = apply_mask(masked_image, maskt, sub_color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros(
        #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=color)
        #     ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig('ex_temp_img/tt1.png')
    plt.show()




def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def display_full_face_toacc(image, boxes, masks, class_ids,mask_face3,scores=None):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box

    """
    # Number of instances
    height, width = image.shape[:2]
    full_result= np.zeros([height,width,11],dtype=np.uint8)

    N = boxes.shape[0]

    # Show area outside image boundaries.
    masked_image = image.astype(np.uint32).copy()
    ms1=0
    ms2=0
    ms3=0
    ms4=0

    for i in range(N):

        # Label
        class_id = class_ids[i]
        score = scores[i]
        #label = class_names[class_id]


        if class_id==1:  # eb_l
            if score>ms1:
                ms1=score
                full_result[:,:,1] = masks[:,:,0,i]
                full_result[:,:,3] = masks[:,:,1,i]

        if class_id ==2:
            if score>ms2:
                ms1=score
                full_result[:,:,2] = masks[:,:,0,i]
                full_result[:,:,4] = masks[:,:,1,i]
        if class_id ==3:
            if score>ms3:
                ms1=score
                full_result[:,:,5] = masks[:,:,0,i]
        if class_id ==4:
            if score>ms4:
                ms1=score
                full_result[:,:,6] = masks[:,:,0,i]
                full_result[:,:,7] = masks[:,:,1,i]
                full_result[:,:,8] = masks[:,:,2,i]
        if class_id ==5:
            full_result[:,:,9] = mask_face3[:,:,0]
            full_result[:, :, 10] = mask_face3[:, :, 1]



    full_result_max = np.argmax(full_result, -1)
    # result_show = util.view_label_det(full_result_max)
    # result_show = np.uint8(result_show)
    # mix = cv2.addWeighted(image,0.7,result_show,0.3,0)
    # plt.imshow(mix)
    # #plt.imshow(np.uint8(result_show),alpha=0.5)
    # plt.show()
    return full_result_max#,result_show

def display_full_face_show(image, boxes, masks, class_ids,mask_face3,scores=None):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box

    """
    # Number of instances
    height, width = image.shape[:2]
    full_result= np.zeros([height,width,11],dtype=np.uint8)

    N = boxes.shape[0]

    # Show area outside image boundaries.
    masked_image = image.astype(np.uint32).copy()
    ms1=0
    ms2=0
    ms3=0
    ms4=0

    for i in range(N):

        # Label
        class_id = class_ids[i]
        score = scores[i]
        #label = class_names[class_id]


        if class_id==1:  # eb_l
            if score>ms1:
                ms1=score
                full_result[:,:,1] = masks[:,:,0,i]
                full_result[:,:,3] = masks[:,:,1,i]

        if class_id ==2:
            if score>ms2:
                ms1=score
                full_result[:,:,2] = masks[:,:,0,i]
                full_result[:,:,4] = masks[:,:,1,i]
        if class_id ==3:
            if score>ms3:
                ms1=score
                full_result[:,:,5] = masks[:,:,0,i]
        if class_id ==4:
            if score>ms4:
                ms1=score
                full_result[:,:,6] = masks[:,:,0,i]
                full_result[:,:,7] = masks[:,:,1,i]
                full_result[:,:,8] = masks[:,:,2,i]
        if class_id ==5:
            full_result[:,:,9] = mask_face3[:,:,0]
            full_result[:, :, 10] = mask_face3[:, :, 1]



    full_result_max = np.argmax(full_result, -1)
    result_show = util.view_label_det2(full_result_max)
    result_show = np.uint8(result_show)
    mix = cv2.addWeighted(image,0.5,result_show,0.5,0)
    # plt.imshow(mix)
    # #plt.imshow(np.uint8(result_show),alpha=0.5)
    # plt.show()
    return full_result_max,mix


def display_full_face_toacc_250(image, boxes, masks, class_ids,mask_face3,scores=None):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box

    """
    # Number of instances
    res = 250
    height, width = image.shape[:2]
    full_result= np.zeros([res,res,11],dtype=np.uint8)

    N = boxes.shape[0]



    for i in range(N):

        # Label
        class_id = class_ids[i]
        score = scores[i]
        #label = class_names[class_id]

        # ori = masks[:,:,0,0]
        # plt.imshow(ori)
        # plt.show()
        #
        # ori_250 = cv2.resize(ori,(250,250),interpolation=cv2.INTER_CUBIC)
        # plt.imshow(ori_250)
        # plt.show()


        if class_id==1:  # eb_l

            full_result[:,:,1] = cv2.resize(masks[:,:,0,i],(res,res),interpolation=cv2.INTER_CUBIC)
            full_result[:,:,3] = cv2.resize(masks[:,:,1,i],(res,res),interpolation=cv2.INTER_CUBIC)

        if class_id ==2:

            full_result[:,:,2] = cv2.resize(masks[:,:,0,i],(res,res),interpolation=cv2.INTER_CUBIC)
            full_result[:,:,4] = cv2.resize(masks[:,:,1,i],(res,res),interpolation=cv2.INTER_CUBIC)
        if class_id ==3:

            full_result[:,:,5] = cv2.resize(masks[:,:,0,i],(res,res),interpolation=cv2.INTER_CUBIC)
        if class_id ==4:

            full_result[:,:,6] = cv2.resize(masks[:,:,0,i],(res,res),interpolation=cv2.INTER_CUBIC)
            full_result[:,:,7] = cv2.resize(masks[:,:,1,i],(res,res),interpolation=cv2.INTER_CUBIC)
            full_result[:,:,8] = cv2.resize(masks[:,:,2,i],(res,res),interpolation=cv2.INTER_CUBIC)
        if class_id ==5:
            full_result[:,:,9] = cv2.resize(mask_face3[:,:,0],(res,res),interpolation=cv2.INTER_CUBIC)
            full_result[:, :, 10] = cv2.resize(mask_face3[:,:,1],(res,res),interpolation=cv2.INTER_CUBIC)



    full_result_max = np.argmax(full_result, -1)
    # image_250 = cv2.resize(image,(250,250))
    # result_show = util.view_label_det(full_result_max)
    # result_show = np.uint8(result_show)
    # mix = cv2.addWeighted(image_250,0.7,result_show,0.3,0)
    # plt.imshow(mix)
    # #plt.imshow(np.uint8(result_show),alpha=0.5)
    # plt.show()
    return full_result_max#,result_show