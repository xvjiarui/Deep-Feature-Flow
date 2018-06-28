# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng
# Modified by Jiarui XU
# --------------------------------------------------------
"""
Nms Multi-thresh Target Operator selects foreground and background roi,
    and assigns label, bbox_transform to them.
    we choose stable tuple instead of max score ones
"""

import mxnet as mx
import numpy as np
import pdb

from bbox.bbox_transform import bbox_overlaps, translation_dist


class NmsMultiTargetOp(mx.operator.CustomOp):
    def __init__(self, target_thresh):
        super(NmsMultiTargetOp, self).__init__()
        self._target_thresh = target_thresh
        self._num_thresh = len(target_thresh)

    def forward(self, is_train, req, in_data, out_data, aux):
        # bbox, [first_n, num_fg_classes, 4]
        bbox = in_data[0].asnumpy()
        gt_box = in_data[1].asnumpy()
        # score, [first_n, num_fg_classes]
        score = in_data[2].asnumpy()

        bbox_bef = in_data[3].asnumpy()
        gt_box_bef = in_data[4].asnumpy()
        score_bef = in_data[5].asnumpy()

        num_fg_classes = bbox.shape[1]
        batch_image, num_gt, code_size = gt_box.shape
        num_fg_classes = bbox.shape[1]
        assert batch_image == 1, 'only support batch_image=1, but receive %d' % num_gt
        assert code_size == 5, 'code_size of gt should be 5, but receive %d' % code_size
        assert len(score.shape) == 2, 'shape of score is %d instead of 2.' % len(score.shape)
        assert score.shape[1] == num_fg_classes, 'number of fg classes should be same for boxes and scores'
        assert bbox.shape[1] == bbox_bef.shape[1], 'num_fg_calsses should be same among frames'
        # assert gt_box.shape[1] == gt_box_bef.shape[1], 'will gt disappear? {} {}'.format(gt_box.shape[1], gt_box_bef.shape[1])

        def get_max_socre_bboxes(score_list_per_class, num_boxes):
            if len(score_list_per_class) == 0:
                return np.zeros(shape=(num_boxes, self._num_thresh), dtype=np.float32)
            else:
                output_list_per_class = []
                for score in score_list_per_class:
                    num_boxes = score.shape[0]
                    max_score_indices = np.argmax(score, axis=0)
                    # in case all indices are 0
                    valid_bbox_indices = np.where(score)[0]
                    output = np.zeros((num_boxes,))

                    output[np.intersect1d(max_score_indices, valid_bbox_indices)] = 1
                    output_list_per_class.append(output)
                output_per_class = np.stack(output_list_per_class, axis=-1)
                return output_per_class

        def get_scores(bbox, gt_box, score):

            output_list = []
            for cls_idx in range(0, num_fg_classes):
                valid_gt_mask = (gt_box[0, :, -1].astype(np.int32)==(cls_idx+1))

                # [num_valid_gt, 5]
                valid_gt_box = gt_box[0, valid_gt_mask, :]
                num_valid_gt = len(valid_gt_box)

                if num_valid_gt == 0:
                    output_list.append([])
                else:
                    bbox_per_class = bbox[:, cls_idx, :]
                    # score_per_class, [first_n, 1]
                    score_per_class = score[:, cls_idx:cls_idx+1]
                    # [first_n, num_valid_gt]
                    overlap_mat = bbox_overlaps(bbox_per_class.astype(np.float),
                                                valid_gt_box[:,:-1].astype(np.float))

                    eye_matrix = np.eye(num_valid_gt)
                    output_list_per_class = []

                    for thresh in self._target_thresh:
                        # following mAP metric
                        overlap_mask = (overlap_mat > thresh)
                        valid_bbox_indices = np.where(overlap_mask)[0]
                        # require score be 2-dim
                        # [first_n, num_valid_gt]
                        overlap_score = np.tile(score_per_class, (1, num_valid_gt))
                        overlap_score *= overlap_mask
                        max_overlap_indices = np.argmax(overlap_mat, axis=1)
                        # [first_n, num_valid_gt]
                        max_overlap_mask = eye_matrix[max_overlap_indices]
                        overlap_score *= max_overlap_mask

                        output_list_per_class.append(overlap_score)
                    output_list.append(output_list_per_class)

            return output_list

        def get_target(bbox, gt_box, score, bbox_bef, gt_box_bef, score_bef):

            num_boxes = bbox.shape[0]
            num_boxes_bef = bbox_bef.shape[0]
            score_list = get_scores(bbox, gt_box, score)
            score_bef_list = get_scores(bbox_bef, gt_box_bef, score_bef)

            output_list = []
            output_bef_list = []
            for cls_idx in range(0, num_fg_classes):

                valid_gt_mask = (gt_box[0, :, -1].astype(np.int32)==(cls_idx+1))
                valid_gt_box = gt_box[0, valid_gt_mask, :]
                num_valid_gt = len(valid_gt_box)

                valid_gt_bef_mask = (gt_box[0, :, -1].astype(np.int32)==(cls_idx+1))
                valid_gt_bef_box = gt_box[0, valid_gt_mask, :]
                num_valid_gt_bef = len(valid_gt_bef_box)
                # assert len(valid_gt_bef_box) == num_valid_gt, "will gt disappear"

                if num_valid_gt != num_valid_gt_bef:
                    if num_valid_gt_bef > num_valid_gt:
                        num_rm = num_valid_gt_bef - num_valid_gt
                        gt_overlap_mat = bbox_overlaps(valid_gt_bef_box.astype(np.float), 
                            valid_gt_box.astype(np.float))
                        rm_indices = np.argsort(np.sum(gt_overlap_mat, axis=1))[:num_rm]
                        np.delete(valid_gt_bef_box, rm_indices, axis=0)
                        assert valid_gt_bef_box.shape == valid_gt_box.shape, "failed remove, {} -> {}".format(valid_gt_bef_box.shape[0], valid_gt_box.shape[0])
                        print "success remove bef"
                    else:
                        num_rm = num_valid_gt - num_valid_gt_bef
                        gt_overlap_mat = bbox_overlaps(valid_gt_box.astype(np.float), 
                            valid_gt_bef_box.astype(np.float))
                        rm_indices = np.argsort(np.sum(gt_overlap_mat, axis=1))[:num_rm]
                        np.delete(valid_gt_box, rm_indices, axis=0)
                        assert valid_gt_bef_box.shape == valid_gt_box.shape, "failed remove, {} -> {}".format(valid_gt_bef_box.shape[0], valid_gt_box.shape[0])
                        print "success remove"
                score_list_per_class = score_list[cls_idx]
                score_bef_list_per_class = score_bef_list[cls_idx]

                bbox_per_class = bbox[:, cls_idx, :]
                bbox_bef_per_class = bbox_bef[:, cls_idx, :]

                if len(score_list_per_class) == 0 or len(score_bef_list_per_class) == 0:
                    output_list.append(get_max_socre_bboxes(score_list_per_class, num_boxes))
                    output_bef_list.append(get_max_socre_bboxes(score_bef_list_per_class, num_boxes_bef))
                else:
                    output_list_per_class = []
                    output_bef_list_per_class = []

                    for i in range(len(self._target_thresh)):
                        overlap_score = score_list_per_class[i]
                        overlap_score_bef = score_bef_list_per_class[i]
                        output = np.zeros((overlap_score.shape[0],))
                        output_bef = np.zeros((overlap_score_bef.shape[0],))
                        valid_bbox_indices = np.where(overlap_score)[0]
                        valid_bbox_bef_indices = np.where(overlap_score_bef)[0]
                        if np.count_nonzero(overlap_score) == 0 or np.count_nonzero(overlap_score_bef) == 0:
                            output_list_per_class.append(output)
                            output_bef_list_per_class.append(output_bef)
                            continue
                        dist_mat = translation_dist(bbox_per_class[valid_bbox_indices], valid_gt_box[:, :-1])
                        dist_bef_mat = translation_dist(bbox_bef_per_class[valid_bbox_bef_indices], valid_gt_bef_box[:, :-1])
                        for x in range(num_valid_gt):
                            dist_mat_shape = (bbox_per_class[valid_bbox_indices].shape[0], 
                                bbox_bef_per_class[valid_bbox_bef_indices].shape[0], 4)
                            bbox_dist_mat = np.sum((np.tile(np.expand_dims(dist_mat[:, x, :], 1), (1, dist_mat_shape[1],1)) - 
                                np.tile(np.expand_dims(dist_bef_mat[:, x, :], 0), (dist_mat_shape[0], 1, 1)))**2, axis=2)
                            assert bbox_dist_mat.shape == (len(bbox_per_class[valid_bbox_indices]), len(bbox_bef_per_class[valid_bbox_bef_indices]))
                            ind, ind_bef = np.unravel_index(np.argmin(bbox_dist_mat), bbox_dist_mat.shape)
                            output[valid_bbox_indices[ind]] = 1
                            output_bef[valid_bbox_bef_indices[ind_bef]] = 1
                        output_list_per_class.append(output)
                        output_bef_list_per_class.append(output_bef)
                    output_per_class = np.stack(output_list_per_class, axis=-1)
                    output_bef_per_class = np.stack(output_bef_list_per_class, axis=-1)
                    output_list.append(output_per_class)
                    output_bef_list.append(output_bef_per_class)
            # [num_boxes, num_fg_classes, num_thresh]
            blob = np.stack(output_list, axis=1).astype(np.float32, copy=False)
            blob_bef = np.stack(output_list, axis=1).astype(np.float32, copy=False)
            return blob, blob_bef

        blob, blob_bef = get_target(bbox, gt_box, score, bbox_bef, gt_box_bef, score_bef)
        blob = np.concatenate((blob, blob_bef))
        self.assign(out_data[0], req[0], blob)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
        self.assign(in_grad[4], req[4], 0)
        self.assign(in_grad[5], req[5], 0)


@mx.operator.register("nms_multi_target")
class NmsMultiTargetProp(mx.operator.CustomOpProp):
    def __init__(self, target_thresh):
        super(NmsMultiTargetProp, self).__init__(need_top_grad=False)
        self._target_thresh = np.fromstring(target_thresh[1:-1], dtype=float, sep=' ')
        self._num_thresh = len(self._target_thresh)

    def list_arguments(self):
        return ['bbox', 'gt_bbox', 'score', 'bbox_bef', 'gt_bbox_bef', 'score_bef']

    def list_outputs(self):
        return ['nms_multi_target']

    def infer_shape(self, in_shape):
        bbox_shape = in_shape[0]
        gt_box_shape = in_shape[1]
        score_shape = in_shape[2]

        bbox_bef_shape = in_shape[3]
        gt_box_bef_shape = in_shape[4]
        score_bef_shape = in_shape[5]

        assert bbox_shape[0] == score_shape[0], 'ROI number should be same for bbox and score'
        assert bbox_bef_shape[0] == score_bef_shape[0], 'ROI number should be same for bbox and score'

        # assert gt_box_shape == gt_box_bef_shape, 'GT is not consistent!!!'

        num_boxes = bbox_shape[0]
        num_fg_classes = bbox_shape[1]
        output_shape = (num_boxes, num_fg_classes, self._num_thresh)

        num_boxes_bef = bbox_bef_shape[0]
        num_fg_classes = bbox_bef_shape[1]
        output_bef_shape = (num_boxes_bef, num_fg_classes, self._num_thresh)

        output_shape = (num_boxes+num_boxes_bef, num_fg_classes, self._num_thresh)

        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NmsMultiTargetOp(self._target_thresh)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
