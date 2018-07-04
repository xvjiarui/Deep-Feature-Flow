# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

from multiprocessing.pool import ThreadPool as Pool
import cPickle
import os
import time
import mxnet as mx
import numpy as np
import pdb

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes, bbox_overlaps
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from utils.PrefetchingIter import PrefetchingIter
from operator_py.nms_multi_target import num_of_is_full_max

class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch, True)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict, scale) in enumerate(zip(scores_all, boxes_all, data_dict_all, scales)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes

def im_detect(predictor, data_batch, data_names, label_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    label_dict_all = [dict(zip(label_names, ilabel)) for ilabel in data_batch.label]
    scores_all = []
    pred_boxes_all = []
    ref_scores_all = []
    ref_pred_boxes_all = []
    for output, data_dict, label_dict, scale in zip(output_all, data_dict_all, label_dict_all, scales):
        if cfg.TEST.HAS_RPN or cfg.network.ROIDispatch:
            concat_rois = output['concat_rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        if cfg.TEST.LEARN_NMS:
            concat_pred_boxes = output['concat_sorted_bbox_output'].asnumpy()
            # raw_scores = output['sorted_score_output'].asnumpy()
            concat_nms_scores = output['nms_final_score_output'].asnumpy()
            # we used scaled image & roi to train, so it is necessary to transform them back
            concat_pred_boxes = concat_pred_boxes / scale


            pred_boxes, ref_pred_boxes = np.split(concat_pred_boxes, 2)
            scores, ref_scores = np.split(concat_nms_scores, 2)

            pred_boxes_all.append(pred_boxes)
            ref_pred_boxes_all.append(ref_pred_boxes)
            scores_all.append(scores)
            ref_scores_all.append(ref_scores)

            nms_multi_target = output['custom0_nms_multi_target'].asnumpy()
            target, ref_target = np.split(nms_multi_target, 2)
            concat_target_boxes = concat_pred_boxes[np.where(nms_multi_target)[:2]]
            concat_target_scores = concat_nms_scores[np.where(nms_multi_target)[:2]]

            # concat_target_boxes = concat_target_boxes / scale

            # construct gt style nms_multi_target, 0:30 classes
            concat_target_boxes = np.hstack((concat_target_boxes, np.where(nms_multi_target)[1][:, np.newaxis]))
            concat_target_boxes = np.hstack((concat_target_boxes, concat_target_scores[:, np.newaxis]))

            target_boxes, ref_target_boxes = np.split(concat_target_boxes, 2)

            label_dict['nms_multi_target'] = target_boxes
            label_dict['ref_nms_multi_target'] = ref_target_boxes

        else:
            rois, ref_rois = np.split(concat_rois, 2)
            scores = output['cls_prob_reshape_output'].asnumpy()[0]
            bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
            ref_scores = output['cls_prob_reshape_output'].asnumpy()[1]
            ref_bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[1]

            # post processing
            pred_boxes = bbox_pred(rois, bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            pred_boxes /= scale

            ref_pred_boxes = bbox_pred(ref_rois, ref_bbox_deltas)
            ref_pred_boxes = clip_boxes(ref_pred_boxes, im_shape[-2:])
            ref_pred_boxes /= scale

            pred_boxes_all.append(pred_boxes)
            scores_all.append(scores)
            ref_pred_boxes_all.append(ref_pred_boxes)
            ref_scores_all.append(ref_scores)

    return scores_all, pred_boxes_all, ref_scores_all, ref_pred_boxes_all, data_dict_all, label_dict_all


def im_batch_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        im_infos = data_dict['im_info'].asnumpy()
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        rois = output['rois_output'].asnumpy()
        for im_idx in xrange(im_infos.shape[0]):
            bb_idxs = np.where(rois[:,0] == im_idx)[0]
            im_shape = im_infos[im_idx, :2].astype(np.int)

            # post processing
            pred_boxes = bbox_pred(rois[bb_idxs, 1:], bbox_deltas[bb_idxs, :])
            pred_boxes = clip_boxes(pred_boxes, im_shape)

            # we used scaled image & roi to train, so it is necessary to transform them back
            pred_boxes = pred_boxes / scale[im_idx]

            scores_all.append(scores[bb_idxs, :])
            pred_boxes_all.append(pred_boxes)

    return scores_all, pred_boxes_all, data_dict_all

def bbox_equal_count(src_boxes, dst_boxes, epsilon=1e-5):
    def bbox_equal(src_box, dst_box):
        ret = True
        for i in range(4):
            ret = ret and abs(src_box[i] - dst_box[i]) < epsilon
        return ret

    count = 0
    for src_box in src_boxes:
        for dst_box in dst_boxes:
            if bbox_equal(src_box, dst_box):
                 count += 1

    return count

DEBUG = False
def pred_eval(predictor, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True, show_gt=False):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        info_str = imdb.evaluate_detections(all_boxes)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]
    label_names = [k[0] for k in test_data.provide_label[0]]
    num_images = test_data.size

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    #if cfg.TEST.SOFTNMS:
    #    nms = py_softnms_wrapper(cfg.TEST.NMS)
    #else:
    #    nms = py_nms_wrapper(cfg.TEST.NMS)

    if cfg.TEST.SOFTNMS:
        nms = py_softnms_wrapper(cfg.TEST.NMS)
    else:
        nms = py_nms_wrapper(cfg.TEST.NMS)


    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    ref_all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    # class_lut = [[] for _ in range(imdb.num_classes)]
    valid_tally = 0
    valid_sum = 0

    idx = 0
    t = time.time()
    inference_count = 0
    all_inference_time = []
    post_processing_time = []
    nms_full_count = []
    nms_pos_count = []
    is_max_count = []
    all_count = []
    for im_info, ref_im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, ref_scores_all, ref_boxes_all, data_dict_all, label_dict_all = im_detect(predictor, data_batch, data_names, label_names, scales, cfg)


        t2 = time.time() - t
        t = time.time()
        # for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
        nms_full_count_per_batch = 0
        nms_pos_count_per_batch = 0
        global num_of_is_full_max
        is_max_count_per_batch = num_of_is_full_max[0]
        all_count_per_batch = 0
        for delta, (scores, boxes, ref_scores, ref_boxes, data_dict, label_dict) in enumerate(zip(scores_all, boxes_all, ref_scores_all, ref_boxes_all, data_dict_all, label_dict_all)):
            if cfg.TEST.LEARN_NMS:
                for j in range(1, imdb.num_classes):
                    indexes = np.where(scores[:, j-1] > thresh)[0]
                    cls_scores = scores[indexes, j-1:j]
                    cls_boxes = boxes[indexes, j-1, :]
                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    # count the valid ground truth
                    if len(cls_scores) > 0:
                        # class_lut[j].append(idx + delta)
                        valid_tally += len(cls_scores)
                        valid_sum += len(scores)

                    all_boxes[j][idx + delta] = cls_dets

                    if DEBUG:
                        keep = nms(cls_dets)
                        nms_cls_dets = cls_dets[keep, :]
                        target = label_dict['nms_multi_target']
                        target_indices = np.where(target[:, 4] == j-1)
                        target = target[target_indices]
                        nms_full_count_per_batch += bbox_equal_count(nms_cls_dets, target)

                        gt_boxes = label_dict['gt_boxes'][0].asnumpy()
                        gt_boxes = gt_boxes[np.where(gt_boxes[:, 4] == j)[0], :4]
                        gt_boxes /= scales[delta]

                        if len(cls_boxes) != 0 and len(gt_boxes) != 0:
                            overlap_mat = bbox_overlaps(cls_boxes.astype(np.float), gt_boxes.astype(np.float))
                            keep = nms(cls_dets[np.where(overlap_mat > 0.5)[0]])
                            nms_cls_dets = cls_dets[np.where(overlap_mat > 0.5)[0]][keep]
                            nms_pos_count_per_batch += bbox_equal_count(nms_cls_dets, target)
                        all_count_per_batch += len(target)
            else:
                for j in range(1, imdb.num_classes):
                    indexes = np.where(scores[:, j] > thresh)[0]
                    if cfg.TEST.FIRST_N > 0:
                        # todo: check whether the order affects the result
                        sort_indices = np.argsort(scores[:, j])[-cfg.TEST.FIRST_N:]
                        # sort_indices = np.argsort(-scores[:, j])[0:cfg.TEST.FIRST_N]
                        indexes = np.intersect1d(sort_indices, indexes)

                    cls_scores = scores[indexes, j, np.newaxis]
                    cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                    # count the valid ground truth
                    if len(cls_scores) > 0:
                        # class_lut[j].append(idx+delta)
                        valid_tally += len(cls_scores)
                        valid_sum += len(scores)
                        # print np.min(cls_scores), valid_tally, valid_sum
                        # cls_scores = scores[:, j, np.newaxis]
                        # cls_scores[cls_scores <= thresh] = thresh
                        # cls_boxes = boxes[:, 4:8] if cfg.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    if cfg.TEST.SOFTNMS:
                        all_boxes[j][idx + delta] = nms(cls_dets)
                    else:
                        keep = nms(cls_dets)
                        all_boxes[j][idx + delta] = cls_dets[keep, :]
                        # all_boxes[j][idx + delta] = cls_dets

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                if show_gt:
                    gt_boxes = label_dict['gt_boxes'][0]
                    for gt_box in gt_boxes:
                        gt_box = gt_box.asnumpy()
                        gt_cls = int(gt_box[4])
                        gt_box = gt_box/scales[delta]
                        gt_box[4] = 1
                        boxes_this_image[gt_cls] = np.vstack((boxes_this_image[gt_cls], gt_box))

                    if cfg.TEST.LEARN_NMS:
                        target_boxes = label_dict['nms_multi_target']
                        for target_box in target_boxes:
                            print("cur", target_box*scales[delta])
                            target_cls = int(target_box[4])+1
                            target_box[4] = 2 + target_box[5]
                            target_box = target_box[:5]
                            boxes_this_image[target_cls] = np.vstack((boxes_this_image[target_cls], target_box))
                # vis_all_detection(data_dict['ref_data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)
                # vis_double_all_detection(data_dict['data'].asnumpy(), boxes_this_image, data_dict['ref_data'].asnumpy(), ref_boxes_this_image, imdb.classes, scales[delta], cfg)
            if cfg.TEST.LEARN_NMS:
                for j in range(1, imdb.num_classes):
                    indexes = np.where(ref_scores[:, j-1] > thresh)[0]
                    cls_scores = ref_scores[indexes, j-1:j]
                    cls_boxes = ref_boxes[indexes, j-1, :]
                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    # count the valid ground truth
                    if len(cls_scores) > 0:
                        # class_lut[j].append(idx + delta)
                        valid_tally += len(cls_scores)
                        valid_sum += len(ref_scores)
                    ref_all_boxes[j][idx + delta] = cls_dets

                    if DEBUG:
                        pass
                        keep = nms(cls_dets)
                        nms_cls_dets = cls_dets[keep, :]
                        target = label_dict['ref_nms_multi_target']
                        target_indices = np.where(target[:, 4] == j-1)
                        target = target[target_indices]
                        nms_full_count_per_batch += bbox_equal_count(nms_cls_dets, target)

                        gt_boxes = label_dict['ref_gt_boxes'][0].asnumpy()
                        gt_boxes = gt_boxes[np.where(gt_boxes[:, 4] == j)[0], :4]
                        gt_boxes /= scales[delta]

                        if len(cls_boxes) != 0 and len(gt_boxes) != 0:
                            overlap_mat = bbox_overlaps(cls_boxes.astype(np.float), gt_boxes.astype(np.float))
                            keep = nms(cls_dets[np.where(overlap_mat > 0.5)[0]])
                            nms_cls_dets = cls_dets[np.where(overlap_mat > 0.5)[0]][keep]
                            nms_pos_count_per_batch += bbox_equal_count(nms_cls_dets, target)
                        all_count_per_batch += len(target)
            else:
                for j in range(1, imdb.num_classes):
                    indexes = np.where(ref_scores[:, j] > thresh)[0]
                    if cfg.TEST.FIRST_N > 0:
                        # todo: check whether the order affects the result
                        sort_indices = np.argsort(ref_scores[:, j])[-cfg.TEST.FIRST_N:]
                        # sort_indices = np.argsort(-scores[:, j])[0:cfg.TEST.FIRST_N]
                        indexes = np.intersect1d(sort_indices, indexes)

                    cls_scores = ref_scores[indexes, j, np.newaxis]
                    cls_boxes = ref_boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else ref_boxes[indexes, j * 4:(j + 1) * 4]
                    # count the valid ground truth
                    if len(cls_scores) > 0:
                        # class_lut[j].append(idx+delta)
                        valid_tally += len(cls_scores)
                        valid_sum += len(ref_scores)
                        # print np.min(cls_scores), valid_tally, valid_sum
                        # cls_scores = scores[:, j, np.newaxis]
                        # cls_scores[cls_scores <= thresh] = thresh
                        # cls_boxes = boxes[:, 4:8] if cfg.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    if cfg.TEST.SOFTNMS:
                        ref_all_boxes[j][idx + delta] = nms(cls_dets)
                    else:
                        keep = nms(cls_dets)
                        ref_all_boxes[j][idx + delta] = cls_dets[keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([ref_all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(ref_all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        ref_all_boxes[j][idx+delta] = ref_all_boxes[j][idx+delta][keep, :]

            if vis:
                ref_boxes_this_image = [[]] + [ref_all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                if show_gt:
                    gt_boxes = label_dict['ref_gt_boxes'][0]
                    for gt_box in gt_boxes:
                        gt_box = gt_box.asnumpy()
                        gt_cls = int(gt_box[4])
                        gt_box = gt_box/scales[delta]
                        gt_box[4] = 1
                        ref_boxes_this_image[gt_cls] = np.vstack((ref_boxes_this_image[gt_cls], gt_box))

                    if cfg.TEST.LEARN_NMS:
                        target_boxes = label_dict['ref_nms_multi_target']
                        for target_box in target_boxes:
                            print("ref", target_box*scales[delta])
                            target_cls = int(target_box[4]) + 1
                            target_box[4] = 2 + target_box[5]
                            target_box = target_box[:5]
                            ref_boxes_this_image[target_cls] = np.vstack((ref_boxes_this_image[target_cls], target_box))
                vis_double_all_detection(data_dict['data'].asnumpy(), boxes_this_image, data_dict['ref_data'].asnumpy(), ref_boxes_this_image, imdb.classes, scales[delta], cfg)
                # vis_all_detection(data_dict['ref_data'].asnumpy(), ref_boxes_this_image, imdb.classes, scales[delta], cfg)

        if DEBUG:
            nms_full_count.append(nms_full_count_per_batch)
            nms_pos_count.append(nms_pos_count_per_batch)
            is_max_count.append(is_max_count_per_batch)
            all_count.append(all_count_per_batch)
            print("full:{} pos:{} max:{}".format(1.0*sum(nms_full_count)/sum(all_count), 1.0*sum(nms_pos_count)/sum(all_count), 1.0*sum(is_max_count)/sum(all_count)))
        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        post_processing_time.append(t3)
        all_inference_time.append(t1 + t2 + t3)
        inference_count += 1
        if inference_count % 200 == 0:
            valid_count = 500 if inference_count > 500 else inference_count
            print("--->> running-average inference time per batch: {}".format(float(sum(all_inference_time[-valid_count:]))/valid_count))
            print("--->> running-average post processing time per batch: {}".format(float(sum(post_processing_time[-valid_count:]))/valid_count))
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images, t1, t2, t3)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images, t1, t2, t3))

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # np.save('class_lut.npy', class_lut)

    info_str = imdb.evaluate_detections(all_boxes)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))
        # num_valid_classes = [len(x) for x in class_lut]
        logger.info('valid class ratio:{}'.format(np.sum(num_valid_classes)/float(num_images)))
        logger.info('valid score ratio:{}'.format(float(valid_tally)/float(valid_sum+0.01)))


def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-4):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            elif score > 0.9999:
                gt_color = (random.random(), random.random(), random.random())  # generate a random color
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=3.5)
            else:

                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()

def vis_double_all_detection(im_array, detections, ref_im_array, ref_detections, class_names, scale, cfg, threshold=1e-4):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    ref_im = image.transform_inverse(ref_im_array, cfg.network.PIXEL_MEANS)
    fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), tight_layout={'pad': 0})

    figures = {'cur': im, 'ref': ref_im}
    detections_dict = {'cur': detections, 'ref': ref_detections}
    for ind, title in enumerate(figures):
        im = figures[title]
        detections = detections_dict[title]
        axeslist.ravel()[ind].imshow(im)
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
        for j, name in enumerate(class_names):
            if name == '__background__':
                continue
            origin_color = (random.random(), random.random(), random.random())  # generate a random color
            target_color = (random.random(), random.random(), random.random())  # generate a random color
            dets = detections[j]
            for det in dets:
                bbox = det[:4] * scale
                score = det[-1]
                if score < threshold or score==1:
                    continue
                linewidth = 0.5 
                # if score == 1:
                #     linewidth = 3.5
                # elif score == 2:
                #     linewidth = 1.5
                color = origin_color
                if score >= 2:
                    color = target_color
                    linewidth = 2.5
                    score -= 2
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=linewidth, alpha=0.7)
                axeslist.ravel()[ind].add_patch(rect)
                axeslist.ravel()[ind].text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    for i, title in enumerate(figures):
        ind = i+2
        im = figures[title]
        detections = detections_dict[title]
        axeslist.ravel()[ind].imshow(im)
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
        for j, name in enumerate(class_names):
            if name == '__background__':
                continue
            origin_color = (random.random(), random.random(), random.random())  # generate a random color
            target_color = (random.random(), random.random(), random.random())
            dets = detections[j]
            for det in dets:
                bbox = det[:4] * scale
                score = det[-1]
                if score < 1:
                    continue
                linewidth = 0.5 
                if score == 1:
                    linewidth = 3.5
                    color = origin_color
                elif score >= 2:
                    linewidth = 1.5
                    color = target_color
                    score -= 2
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=linewidth, alpha=0.7)
                axeslist.ravel()[ind].add_patch(rect)
                axeslist.ravel()[ind].text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    # plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()

def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def double_im_detect(predictor, data_batch, data_names, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    ref_scores_all = []
    ref_pred_boxes_all = []
    for output, data_dict in zip(output_all, data_dict_all):

        scale = data_dict['im_info'][0, 2]

        rois = output['rois_output'].asnumpy()[:, 1:]
        ref_rois = output['ref_rois_output'].asnumpy()[:, 1:]
        im_shape = data_dict['data'].shape
        ref_im_shape = data_dict['ref_data']
        non_ref_dim = rois.shape[0]

        # save output
        scores = output['cls_prob_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_output'].asnumpy()[0]
        ref_scores = output['cls_prob_output'].asnumpy()[1]
        ref_bbox_deltas = output['bbox_pred_output'].asnumpy()[1]

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        ref_pred_boxes = bbox_pred(ref_rois, ref_bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
        ref_pred_boxes = clip_boxes(ref_pred_boxes, ref_im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale
        ref_pred_boxes = ref_pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)

        ref_scores_all.append(ref_scores)
        ref_pred_boxes_all.append(ref_pred_boxes)

    return scores_all, pred_boxes_all, ref_scores_all, ref_pred_boxes_all, data_dict_all

