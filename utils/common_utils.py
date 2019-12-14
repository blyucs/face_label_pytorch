import numpy as np
import tensorflow as tf
import scipy.misc as misc
import cv2
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result



def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def restore_mask_class(mask,bbox,image_shape):
    bbox = np.int32(bbox)
    threshold = 0.5
    x1, y1, x2, y2 = bbox
    full_mask = np.zeros([image_shape[0],image_shape[1],9], dtype=np.uint8)
    amax = np.amax(mask,2)
    for i in range(mask.shape[-1]):
        maskt = mask[:,:,i]-amax
        maskt = np.where(maskt>=0.0,1,0).astype(np.uint8)
        maskt = misc.imresize(
            maskt, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) #/ 255.0
        maskt = np.where(maskt >= threshold, 1, 0).astype(np.uint8)

        # Put the mask in the right location.

        full_mask[y1:y2, x1:x2,i] = maskt
    return full_mask



def restore_submask(sub_rois,sub_score,sub_masks,image_shape,sub_class):
    sumt = np.sum(sub_rois,axis=-1)
    zero_ix = np.where(sumt==0)[0][0]

    sub_roi = sub_rois[:zero_ix,:]
    sub_mask = sub_masks[:zero_ix,:,:,:]

    full_masks=[]
    for i in range(zero_ix):
        full_mask = restore_mask_class(sub_mask[i],sub_roi[i],image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks,axis=-1) if full_masks else np.empty((0,)+sub_mask.shape[1:4])
    class_id = np.ones([zero_ix])*sub_class
    class_id = np.int32(class_id)
    score = sub_score[:zero_ix]
    return full_masks,sub_roi,class_id,score

def restore_submask_withW(sub_rois,sub_score,sub_masks,image_shape,sub_class,window):
    sumt = np.sum(sub_rois,axis=-1)
    zero_ix = np.where(sumt==0)[0][0]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[1], shift[0], shift[1], shift[0]])

    sub_roi = sub_rois[:zero_ix,:]
    sub_mask = sub_masks[:zero_ix,:,:,:]
    # Translate bounding boxes to image domain
    sub_roi = np.multiply(sub_roi - shifts, scales).astype(np.int32)
    #sub_roi = np.abs(sub_roi)

    sub_roi[:, 0] = np.clip(sub_roi[:, 0], 0, image_shape[1])
    sub_roi[:, 1] = np.clip(sub_roi[:, 1], 0, image_shape[0])
    sub_roi[:, 2] = np.clip(sub_roi[:, 2], 0, image_shape[1])
    sub_roi[:, 3] = np.clip(sub_roi[:, 3], 0, image_shape[0])
    full_masks=[]
    for i in range(zero_ix):
        full_mask = restore_mask_class(sub_mask[i],sub_roi[i],image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks,axis=-1) if full_masks else np.empty((0,)+sub_mask.shape[1:4])
    class_id = np.ones([zero_ix])*sub_class
    class_id = np.int32(class_id)
    score = sub_score[:zero_ix]
    return full_masks,sub_roi,class_id,score

def restore_submask_face_withW(sub_rois,sub_score,image_shape,sub_class,window):
    sumt = np.sum(sub_rois,axis=-1)
    zero_ix = np.where(sumt==0)[0][0]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[1], shift[0], shift[1], shift[0]])

    sub_roi = sub_rois[:zero_ix, :]

    # Translate bounding boxes to image domain
    sub_roi = np.multiply(sub_roi - shifts, scales).astype(np.int32)
    sub_roi[:, 0] = np.clip(sub_roi[:, 0], 0, image_shape[1])
    sub_roi[:, 1] = np.clip(sub_roi[:, 1], 0, image_shape[0])
    sub_roi[:, 2] = np.clip(sub_roi[:, 2], 0, image_shape[1])
    sub_roi[:, 3] = np.clip(sub_roi[:, 3], 0, image_shape[0])
    full_masks = np.zeros([image_shape[0], image_shape[1], 9,zero_ix], dtype=np.uint8)
    class_id = np.ones([zero_ix])*sub_class
    class_id = np.int32(class_id)
    score = sub_score[:zero_ix]
    return full_masks,sub_roi,class_id,score



def restore_submask_face(sub_rois,sub_score,image_shape,sub_class):
    sumt = np.sum(sub_rois,axis=-1)
    zero_ix = np.where(sumt==0)[0][0]

    sub_roi = sub_rois[:zero_ix,:]

    full_masks = np.zeros([image_shape[0], image_shape[1], 9,zero_ix], dtype=np.uint8)
    class_id = np.ones([zero_ix])*sub_class
    class_id = np.int32(class_id)
    score = sub_score[:zero_ix]
    return full_masks,sub_roi,class_id,score



def filter_results(results):
    final_rois = results['rois']
    final_class_ids = results['class_ids']
    final_masks = results['masks']
    final_score = results['score']
    mface = results['face_mask']

    face_index = np.where(final_class_ids==5)[0]
    face_score = final_score[face_index]
    face_max_index = face_index[np.argmax(face_score)]
    face_srois = final_rois[face_max_index]
    fx1,fy1,fx2,fy2 = face_srois
    face_sscore = final_score[face_max_index]
    face_cclass = final_class_ids[face_max_index]
    face_cmask = final_masks[:,:,:,face_max_index]



    eb1_index = np.where(final_class_ids==1)[0]
    eb1_good_index = []
    for i in range(len(eb1_index)):
        t_box = final_rois[eb1_index[i]]
        tx1,ty1,tx2,ty2 = t_box
        tcx = (tx1+tx2)/2.
        tcy = (ty1+ty2)/2.
        if (tcx>fx1) & (tcx<fx2) & (tcy>fy1) & (tcy<fy2):
            eb1_good_index.append(eb1_index[i])
    if len(eb1_good_index)==0:
        return None
    eb1_good_index = np.array(eb1_good_index)
    eb1_score = final_score[eb1_good_index]
    eb1_max_index = eb1_good_index[np.argmax(eb1_score)]

    eb1_srois = final_rois[eb1_max_index]
    eb1_sscore = final_score[eb1_max_index]
    eb1_cclass = final_class_ids[eb1_max_index]
    eb1_cmask = final_masks[:,:,:,eb1_max_index]

    eb2_index = np.where(final_class_ids==2)[0]
    eb2_good_index = []
    for i in range(len(eb2_index)):
        t_box = final_rois[eb2_index[i]]
        tx1,ty1,tx2,ty2 = t_box
        tcx = (tx1+tx2)/2.
        tcy = (ty1+ty2)/2.
        if (tcx>fx1) & (tcx<fx2) & (tcy>fy1) & (tcy<fy2):
            eb2_good_index.append(eb2_index[i])
    if len(eb2_good_index)==0:
        return None
    eb2_good_index = np.array(eb2_good_index)
    eb2_score = final_score[eb2_good_index]
    eb2_max_index = eb2_good_index[np.argmax(eb2_score)]

    eb2_srois = final_rois[eb2_max_index]
    eb2_sscore = final_score[eb2_max_index]
    eb2_cclass = final_class_ids[eb2_max_index]
    eb2_cmask = final_masks[:,:,:,eb2_max_index]


    nose_index = np.where(final_class_ids==3)[0]
    nose_good_index = []
    for i in range(len(nose_index)):
        t_box = final_rois[nose_index[i]]
        tx1,ty1,tx2,ty2 = t_box
        tcx = (tx1+tx2)/2.
        tcy = (ty1+ty2)/2.
        if (tcx>fx1) & (tcx<fx2) & (tcy>fy1) & (tcy<fy2):
            nose_good_index.append(nose_index[i])
    if len(nose_good_index)==0:
        return None
    nose_good_index = np.array(nose_good_index)
    nose_score = final_score[nose_good_index]
    nose_max_index = nose_good_index[np.argmax(nose_score)]

    nose_srois = final_rois[nose_max_index]
    nose_sscore = final_score[nose_max_index]
    nose_cclass = final_class_ids[nose_max_index]
    nose_cmask = final_masks[:,:,:,nose_max_index]


    mouth_index = np.where(final_class_ids==4)[0]
    mouth_good_index = []
    for i in range(len(mouth_index)):
        t_box = final_rois[mouth_index[i]]
        tx1,ty1,tx2,ty2 = t_box
        tcx = (tx1+tx2)/2.
        tcy = (ty1+ty2)/2.
        if (tcx>fx1) & (tcx<fx2) & (tcy>fy1) & (tcy<fy2):
            mouth_good_index.append(mouth_index[i])
    if len(mouth_good_index)==0:
        return None
    mouth_good_index = np.array(mouth_good_index)

    mouth_score = final_score[mouth_good_index]
    mouth_max_index = mouth_good_index[np.argmax(mouth_score)]

    mouth_srois = final_rois[mouth_max_index]
    mouth_sscore = final_score[mouth_max_index]
    mouth_cclass = final_class_ids[mouth_max_index]
    mouth_cmask = final_masks[:,:,:,mouth_max_index]


    final_masks2 = np.stack([eb1_cmask,eb2_cmask,nose_cmask,mouth_cmask,face_cmask], axis=-1)
    final_rois2 = np.stack([eb1_srois, eb2_srois, nose_srois, mouth_srois, face_srois], axis=0)
    final_class_ids2 = np.stack([eb1_cclass, eb2_cclass, nose_cclass, mouth_cclass, face_cclass], axis=0)
    final_score2 = np.stack([eb1_sscore, eb2_sscore, nose_sscore, mouth_sscore, face_sscore], axis=0)




    #print(mface)

    results2 = {
        'rois': final_rois2,
        'class_ids': final_class_ids2,
        'masks': final_masks2,
        'face_mask': mface,
        "score": final_score2
    }
    return results2



def view_label_det(im):
    cshape = im.shape
    im_temp = np.zeros([cshape[0], cshape[1], 3])
    for i in range(cshape[0]):
        for j in range(cshape[1]):
            if im[i][j] == 0:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255
            if im[i][j] == 9:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 255
                im_temp[i][j][2] = 0
            if im[i][j] == 2:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 120
            if im[i][j] == 3:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 120
                im_temp[i][j][2] = 240
            if im[i][j] == 4:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 69
                im_temp[i][j][2] = 0
            if im[i][j] == 5:
                im_temp[i][j][0] = 120
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 6:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 215
                im_temp[i][j][2] = 0
            if im[i][j] == 7:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 8:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255
            if im[i][j] == 1: #bg
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 10:
                im_temp[i][j][0] = 120
                im_temp[i][j][1] = 120
                im_temp[i][j][2] = 60

    return im_temp

def view_label_det2(im):
    cshape = im.shape
    im_temp = np.zeros([cshape[0], cshape[1], 3])
    for i in range(cshape[0]):
        for j in range(cshape[1]):
            if im[i][j] == 0:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255
            if im[i][j] == 9:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 255
                im_temp[i][j][2] = 0
            if im[i][j] == 2:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 120
            if im[i][j] == 3:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 120
                im_temp[i][j][2] = 240
            if im[i][j] == 4:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 69
                im_temp[i][j][2] = 0
            if im[i][j] == 5:
                im_temp[i][j][0] = 120
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 6:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 215
                im_temp[i][j][2] = 0
            if im[i][j] == 7:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 8:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255
            if im[i][j] == 1: #bg
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 10:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255

    return im_temp

def view_label3(im):
    cshape = im.shape
    im_temp = np.zeros([cshape[0], cshape[1], 3],dtype=np.uint8)
    for i in range(cshape[0]):
        for j in range(cshape[1]):
            if im[i][j] == 0:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 255
                im_temp[i][j][2] = 0
            if im[i][j] == 1:
                im_temp[i][j][0] = 255
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 0
            if im[i][j] == 2:
                im_temp[i][j][0] = 0
                im_temp[i][j][1] = 0
                im_temp[i][j][2] = 255

    return im_temp

def mold_inputs(images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image to fit the model expected size
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding = resize_image(
            image,
            min_dim=512,
            max_dim=512,
            padding=True)

        # Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, window,
            np.zeros([6], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, scale, windows

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta
def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        # image = scipy.misc.imresize(
        #     image, (round(h * scale), round(w * scale)),interp='cubic')

        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding