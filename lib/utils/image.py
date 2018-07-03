import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
import phillyzip


num_empty_gt = 0
# TODO: This two functions should be merged with individual data loader
def imread(image_path, arg):
    if '.zip@' in image_path:
        im = phillyzip.imread(image_path, arg)
    else:
        im = cv2.imread(image_path, arg)
    return im
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        image_path = roi_rec['image']
        assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
        # if '.zip@' in image_path:
        #     im = phillyzip.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     im = cv2.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        im = imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        image_path = roi_rec['image']
        assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
        # if '.zip@' in image_path:
        #     im = phillyzip.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     im = cv2.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        im = imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '{} does not exist'.format(ref_image)
            ref_im = imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb

def get_double_image(roidb, config, is_train=True):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_roidb = []
    processed_ref_roidb = []
    min_offset = config.TRAIN.MIN_OFFSET if is_train else config.TEST.MIN_OFFSET
    max_offset = config.TRAIN.MAX_OFFSET if is_train else config.TEST.MAX_OFFSET
    def load_roi_rec(filename, old_roi_rec=None):

        num_classes = 31

        classes_map = ['__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049']
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        if old_roi_rec != None:
            roi_rec['pattern'] = old_roi_rec['pattern']
            roi_rec['frame_seg_id'] = old_roi_rec['frame_seg_id']
            roi_rec['frame_seg_len'] = old_roi_rec['frame_seg_len']

        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)

        class_to_index = dict(zip(classes_map, range(num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text), 0)
            y1 = np.maximum(float(bbox.find('ymin').text), 0)
            x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)
            y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)
            if not class_to_index.has_key(obj.find('name').text):
                continue
            valid_objs[ix] = True
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        overlaps = overlaps[valid_objs, :]

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    for i in range(num_images):
        roi_rec = roidb[i]

        image_path = roi_rec['image']
        assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
        # if '.zip@' in image_path:
        #     im = phillyzip.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     # print('cur', image_path)
        #     im = cv2.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        im = imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if not is_train:
            print('cur', image_path)
        # update annotation
        annotation = image_path.replace("Data", "Annotations").replace("JPEG","xml")
        assert os.path.exists(annotation), '{} does not exist'.format(annotation)
        if not is_train:
            roi_rec = load_roi_rec(annotation, roi_rec)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            # print('ref', ref_image)
            assert os.path.exists(ref_image), '{} does not exist'.format(ref_image)

            ref_annotation = ref_image.replace("Data", "Annotations").replace("JPEG","xml")
            assert os.path.exists(ref_annotation), '{} does not exist'.format(ref_annotation)
            ref_roi_rec = load_roi_rec(ref_annotation)
            ref_roi_rec['image'] = ref_image
            # in case of empty gt
            while len(ref_roi_rec['gt_classes']) == 0 and ref_id < roi_rec['frame_seg_id']+max(0, config.TRAIN.MAX_OFFSET):
                ref_id += 1
                ref_image = roi_rec['pattern'] % ref_id
                assert os.path.exists(ref_image), '{} does not exist'.format(ref_image)
                ref_annotation = ref_image.replace("Data", "Annotations").replace("JPEG","xml")
                assert os.path.exists(ref_annotation), '{} does not exist'.format(ref_annotation)
                ref_roi_rec = load_roi_rec(ref_annotation)
                ref_roi_rec['image'] = ref_image
                global num_empty_gt
                num_empty_gt += 1
                print "empty gt {}".format(num_empty_gt)

            ref_im = imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if not is_train:
                print('ref', ref_image)
        else:
            ref_im = im.copy()
            ref_roi_rec = roi_rec.copy()

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)

        new_ref_rec = ref_roi_rec.copy()
        ref_im, ref_im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ref_ims.append(ref_im_tensor)
        ref_im_info = [ref_im_tensor.shape[2], ref_im_tensor.shape[3], ref_im_scale]
        new_ref_rec['boxes'] = ref_roi_rec['boxes'].copy() * ref_im_scale
        new_ref_rec['im_info'] = ref_im_info
        processed_ref_roidb.append(new_ref_rec)
    return processed_ims, processed_ref_ims, processed_roidb, processed_ref_roidb

def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
