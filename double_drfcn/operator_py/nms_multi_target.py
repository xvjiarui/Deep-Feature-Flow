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


num_of_is_full_max = [0]
score_rank_max = [0, 0]
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

        ref_bbox = in_data[3].asnumpy()
        ref_gt_box = in_data[4].asnumpy()
        ref_score = in_data[5].asnumpy()

        num_fg_classes = bbox.shape[1]
        batch_image, num_gt, code_size = gt_box.shape
        num_fg_classes = bbox.shape[1]
        assert batch_image == 1, 'only support batch_image=1, but receive %d' % num_gt
        assert code_size == 5, 'code_size of gt should be 5, but receive %d' % code_size
        assert len(score.shape) == 2, 'shape of score is %d instead of 2.' % len(score.shape)
        assert score.shape[1] == num_fg_classes, 'number of fg classes should be same for boxes and scores'
        assert bbox.shape[1] == ref_bbox.shape[1], 'num_fg_calsses should be same among frames'

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

        def get_scores_per_class(bbox_per_class, gt_box_per_class, score_per_class):
            pass
            # bbox [FIRST_N, 4]
            # gt_box [, 4]
            # score [FIRST_N]
            num_valid_gt = len(gt_box_per_class)
            output_list_per_class = []
            if  num_valid_gt== 0:
                return output_list_per_class

            overlap_mat = bbox_overlaps(bbox_per_class.astype(np.float),
                                        gt_box_per_class[:,:-1].astype(np.float))

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

            return output_list_per_class

        def get_target(bbox, gt_box, score, ref_bbox, ref_gt_box, ref_score):

            global num_of_is_full_max
            num_boxes = bbox.shape[0]
            ref_num_boxes = ref_bbox.shape[0]
            score_list = get_scores(bbox, gt_box, score)
            ref_score_list = get_scores(ref_bbox, ref_gt_box, ref_score)

            output_list = []
            ref_output_list = []
            for cls_idx in range(0, num_fg_classes):

                valid_gt_mask = (gt_box[0, :, -1].astype(np.int32)==(cls_idx+1))
                valid_gt_box = gt_box[0, valid_gt_mask, :]
                num_valid_gt = len(valid_gt_box)

                ref_valid_gt_mask = (ref_gt_box[0, :, -1].astype(np.int32)==(cls_idx+1))
                ref_valid_gt_box = ref_gt_box[0, ref_valid_gt_mask, :]
                ref_num_valid_gt = len(ref_valid_gt_box)

                score_list_per_class = score_list[cls_idx]
                ref_score_list_per_class = ref_score_list[cls_idx]

                bbox_per_class = bbox[:, cls_idx, :]
                ref_bbox_per_class = ref_bbox[:, cls_idx, :]

                if num_valid_gt != ref_num_valid_gt:
                    if ref_num_valid_gt > num_valid_gt:
                        num_rm = ref_num_valid_gt - num_valid_gt
                        ref_num_valid_gt = num_valid_gt
                        gt_overlap_mat = bbox_overlaps(ref_valid_gt_box.astype(np.float), 
                            valid_gt_box.astype(np.float))
                        rm_indices = np.argsort(np.sum(gt_overlap_mat, axis=1))[:num_rm]
                        ref_valid_gt_box = np.delete(ref_valid_gt_box, rm_indices, axis=0)
                        # update ref_score_list_per_class
                        ref_score_list_per_class = get_scores_per_class(ref_bbox_per_class, ref_valid_gt_box, ref_score[:, cls_idx:cls_idx+1])
                        assert ref_valid_gt_box.shape == valid_gt_box.shape, "failed remove ref, {} -> {}".format(ref_valid_gt_box.shape[0], valid_gt_box.shape[0])
                        print "success remove ref"
                    else:
                        num_rm = num_valid_gt - ref_num_valid_gt
                        num_valid_gt = ref_num_valid_gt
                        gt_overlap_mat = bbox_overlaps(valid_gt_box.astype(np.float), 
                            ref_valid_gt_box.astype(np.float))
                        rm_indices = np.argsort(np.sum(gt_overlap_mat, axis=1))[:num_rm]
                        valid_gt_box = np.delete(valid_gt_box, rm_indices, axis=0)
                        # update score_list_per_class
                        score_list_per_class = get_scores_per_class(bbox_per_class, valid_gt_box, score[:, cls_idx:cls_idx+1])
                        assert ref_valid_gt_box.shape == valid_gt_box.shape, "failed remove, {} -> {}".format(ref_valid_gt_box.shape[0], valid_gt_box.shape[0])
                        print "success remove"

                assert num_valid_gt == ref_num_valid_gt, "gt num are not the same"


                if len(score_list_per_class) == 0 or len(ref_score_list_per_class) == 0:
                    output_list.append(get_max_socre_bboxes(score_list_per_class, num_boxes))
                    ref_output_list.append(get_max_socre_bboxes(ref_score_list_per_class, ref_num_boxes))

                else:
                    output_list_per_class = []
                    ref_output_list_per_class = []

                    for i in range(len(self._target_thresh)):
                        overlap_score = score_list_per_class[i]
                        ref_overlap_score = ref_score_list_per_class[i]
                        output = np.zeros((overlap_score.shape[0],))
                        ref_output = np.zeros((ref_overlap_score.shape[0],))
                        if np.count_nonzero(overlap_score) == 0 or np.count_nonzero(ref_overlap_score) == 0:
                            output_list_per_class.append(output)
                            ref_output_list_per_class.append(ref_output)
                            continue
                        for x in range(num_valid_gt):
                            overlap_score_per_gt = overlap_score[:, x]
                            ref_overlap_score_per_gt = ref_overlap_score[:, x]
                            valid_bbox_indices = np.where(overlap_score_per_gt)[0]
                            ref_valid_bbox_indices = np.where(ref_overlap_score_per_gt)[0]
                            target_gt_box = valid_gt_box[x:x+1, :-1]
                            ref_target_gt_box = ref_valid_gt_box[x:x+1, :-1]
                            if len(valid_bbox_indices) == 0 or len(ref_valid_bbox_indices) == 0:
                                continue
                            dist_mat = translation_dist(bbox_per_class[valid_bbox_indices], target_gt_box)[:, 0, :]
                            ref_dist_mat = translation_dist(ref_bbox_per_class[ref_valid_bbox_indices], ref_target_gt_box)[:, 0, :]
                            dist_mat_shape = (bbox_per_class[valid_bbox_indices].shape[0], 
                                ref_bbox_per_class[ref_valid_bbox_indices].shape[0], 4)
                            # print((np.tile(np.expand_dims(dist_mat, 1), (1, dist_mat_shape[1], 1)) - 
                                # np.tile(np.expand_dims(ref_dist_mat, 0), (dist_mat_shape[0], 1, 1)))**2)
                            bbox_dist_mat = np.sum((np.tile(np.expand_dims(dist_mat, 1), (1, dist_mat_shape[1], 1)) - 
                                np.tile(np.expand_dims(ref_dist_mat, 0), (dist_mat_shape[0], 1, 1)))**2, axis=2)
                            assert bbox_dist_mat.shape == (len(bbox_per_class[valid_bbox_indices]), len(ref_bbox_per_class[ref_valid_bbox_indices]))
                            # top_k = 10
                            # translation_thresh = 1.1*np.min(bbox_dist_mat)
                            # top_k = np.sum(bbox_dist_mat < translation_thresh)
                            top_k = int(0.1 * len(bbox_dist_mat.flatten()) + 0.5)
                            top_k = max(1, top_k)
                            top_k = min(top_k, len(bbox_dist_mat.flatten()))
                            # top_k = 1
                            print("{} of out {} stable pair".format(top_k, len(bbox_dist_mat.flatten())))
                            ind_list, ref_ind_list = np.unravel_index(np.argsort(bbox_dist_mat, axis=None)[:top_k], bbox_dist_mat.shape)
                            score_sum_list = []
                            rank_sum_list = []
                            for ind, ref_ind in zip(ind_list, ref_ind_list):
                                score_sum = overlap_score_per_gt[valid_bbox_indices[ind]] + ref_overlap_score_per_gt[ref_valid_bbox_indices[ref_ind]]
                                rank_sum = valid_bbox_indices[ind] + ref_valid_bbox_indices[ref_ind]
                                score_sum_list.append(score_sum)
                                rank_sum_list.append(rank_sum)
                            score_max_idx = np.argmax(np.array(score_sum_list))
                            rank_max_idx = np.argmin(np.array(rank_sum_list))
                            if score_max_idx == rank_max_idx:
                                score_rank_max[0] += 1
                            score_rank_max[1] += 1
                            # max_idx = rank_max_idx
                            max_idx = score_max_idx
                            ind = ind_list[max_idx]
                            ref_ind = ref_ind_list[max_idx]
                            if ind == np.argmax(overlap_score_per_gt[valid_bbox_indices]):
                                # num_of_is_full_max[0] += 1
                                print('cur takes the max')
                            if ref_ind == np.argmax(ref_overlap_score_per_gt[ref_valid_bbox_indices]):
                                # num_of_is_full_max[0] += 1
                                print('ref takes the max')

                            output[valid_bbox_indices[ind]] = 1
                            ref_output[ref_valid_bbox_indices[ref_ind]] = 1
                        output_list_per_class.append(output)
                        ref_output_list_per_class.append(ref_output)
                    output_per_class = np.stack(output_list_per_class, axis=-1)
                    ref_output_per_class = np.stack(ref_output_list_per_class, axis=-1)
                    output_list.append(output_per_class)
                    ref_output_list.append(ref_output_per_class)
            # [num_boxes, num_fg_classes, num_thresh]
            blob = np.stack(output_list, axis=1).astype(np.float32, copy=False)
            ref_blob = np.stack(ref_output_list, axis=1).astype(np.float32, copy=False)
            return blob, ref_blob

        num_of_is_full_max[0] = 0
        blob, ref_blob = get_target(bbox, gt_box, score, ref_bbox, ref_gt_box, ref_score)
        blob = np.concatenate((blob, ref_blob))
        # print("score_rank:{}".format(1.0*score_rank_max[0]/score_rank_max[1]))
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
        return ['bbox', 'gt_bbox', 'score', 'ref_bbox', 'ref_gt_bbox', 'ref_score']

    def list_outputs(self):
        return ['nms_multi_target']

    def infer_shape(self, in_shape):
        bbox_shape = in_shape[0]
        gt_box_shape = in_shape[1]
        score_shape = in_shape[2]

        ref_bbox_shape = in_shape[3]
        ref_gt_box_shape = in_shape[4]
        ref_score_shape = in_shape[5]

        assert bbox_shape[0] == score_shape[0], 'ROI number should be same for bbox and score'
        assert ref_bbox_shape[0] == ref_score_shape[0], 'ROI number should be same for bbox and score'


        num_boxes = bbox_shape[0]
        num_fg_classes = bbox_shape[1]
        output_shape = (num_boxes, num_fg_classes, self._num_thresh)

        ref_num_boxes = ref_bbox_shape[0]
        num_fg_classes = ref_bbox_shape[1]
        ref_output_shape = (ref_num_boxes, num_fg_classes, self._num_thresh)

        output_shape = (num_boxes+ref_num_boxes, num_fg_classes, self._num_thresh)

        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NmsMultiTargetOp(self._target_thresh)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
