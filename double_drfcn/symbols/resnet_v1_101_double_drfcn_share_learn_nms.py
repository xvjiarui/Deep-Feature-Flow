# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuqing Zhu, Shuhao Fu, Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
import math

from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.tile_as import *
from operator_py.nms_multi_target import *
from resnet_v1_101_rcnn_learn_nms_base import resnet_v1_101_rcnn_learn_nms_base as NMS_UTILS
from resnet_v1_101_double_drfcn import resnet_v1_101_double_drfcn



class resnet_v1_101_double_drfcn_share_learn_nms(resnet_v1_101_double_drfcn):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 2e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        feat_range = mx.sym.arange(0, feat_dim / 8)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(8. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
        position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
        div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
        sin_mat = mx.sym.sin(data=div_mat)
        cos_mat = mx.sym.cos(data=div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
        return embedding

    @staticmethod
    def extract_position_matrix(bbox, nongt_dim):
        """ Extract position matrix

        Args:
            bbox: [num_boxes, 4]

        Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        """
        xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                              num_outputs=4, axis=1)
        # [num_fg_classes, num_boxes, 1]
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # [num_fg_classes, num_boxes, num_boxes]
        delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                         rhs=mx.sym.transpose(center_x))
        delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
        delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
        delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                         rhs=mx.sym.transpose(center_y))
        delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
        delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
        delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                           rhs=mx.sym.transpose(bbox_width))
        delta_width = mx.sym.log(delta_width)
        delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                            rhs=mx.sym.transpose(bbox_height))
        delta_height = mx.sym.log(delta_height)
        concat_list = [delta_x, delta_y, delta_width, delta_height]
        for idx, sym in enumerate(concat_list):
            sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim)
            concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
        position_matrix = mx.sym.concat(*concat_list, dim=2)
        return position_matrix

    def attention_module_multi_head(self, roi_feat, position_embedding, nongt_dim, fc_dim, feat_dim, dim=(1024, 1024, 1024), group=16, index=1):
        """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        nongt_roi_feat = mx.symbol.slice_axis(data=roi_feat, axis=0, begin=0, end=nongt_dim)
        # [num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = mx.sym.Reshape(position_embedding, shape=(-3, -2))
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        position_feat_1 = mx.sym.FullyConnected(name='pair_pos_fc1_' + str(index),
                                                data=position_embedding_reshape,
                                                num_hidden=fc_dim)
        position_feat_1_relu = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = mx.sym.Reshape(position_feat_1_relu, shape=(-1, nongt_dim, fc_dim))
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = mx.sym.transpose(aff_weight, axes=(0, 2, 1))

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                       data=roi_feat,
                                       num_hidden=dim[0])
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(1, 0, 2))
        k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                          data=nongt_roi_feat,
                                          num_hidden=dim[1])
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(1, 0, 2))
        v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = mx.sym.transpose(aff_scale, axes=(1, 0, 2))

        assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        weighted_aff = mx.sym.log(mx.sym.maximum(left=aff_weight, right=1e-6)) + aff_scale
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='softmax_' + str(index))
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-3, -2))
        # output_t, [num_rois * fc_dim, feat_dim]
        output_t = mx.symbol.dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = mx.sym.Reshape(output_t, shape=(-1, fc_dim * feat_dim, 1, 1))
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = mx.symbol.Convolution(name='linear_out_' + str(index), data=output_t,
                                           kernel=(1, 1), num_filter=dim[2], num_group=fc_dim)
        output = mx.sym.Reshape(linear_out, shape=(0, 0))
        return output

    def attention_module_nms_multi_head(self, roi_feat, position_mat, num_rois, dim=(1024, 1024, 1024), fc_dim=(64, 16), feat_dim=1024, group=16, index=1):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, num_fg_classes, feat_dim]
            position_mat: [num_fg_classes, num_rois, num_rois, 4]
            num_rois: number of rois
            dim: key, query and linear_out dim
            fc_dim:
            feat_dim:
            group:
            index:

        Returns:
            output: [num_rois, num_fg_classes, fc_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        roi_feat = mx.sym.transpose(roi_feat, axes=(1, 0, 2))
        # roi_feat_reshape, [num_fg_classes*num_rois, feat_dim]
        roi_feat_reshape = mx.sym.Reshape(roi_feat, shape=(-3, -2))
        # position_embedding, [num_fg_classes, num_rois, num_rois, fc_dim[0]]
        position_embedding = NMS_UTILS.extract_pairwise_multi_position_embedding(position_mat, fc_dim[0])
        # [num_fg_classes * num_rois * num_rois, fc_dim[0]]
        position_embedding_reshape =  mx.sym.Reshape(position_embedding, shape=(-1, fc_dim[0]))
        # position_feat_1, [num_fg_classes * num_rois * num_rois, fc_dim[1]]
        position_feat_1 = mx.sym.FullyConnected(name='nms_pair_pos_fc1_' + str(index),
                                                data=position_embedding_reshape,
                                                num_hidden=fc_dim[1])
        # position_feat_1, [num_fg_classes, num_rois, num_rois, fc_dim[1]]
        position_feat_1 = mx.sym.Reshape(position_feat_1, shape=(-1, num_rois, num_rois, fc_dim[1]))
        aff_weight = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_fg_classes, fc_dim[1], num_rois, num_rois]
        aff_weight = mx.sym.transpose(aff_weight, axes=(0, 3, 1, 2))

        ####################### multi head in batch###########################
        assert dim[0] == dim[1], 'Matrix multi requires the same dims!'
        # q_data, [num_fg_classes * num_rois, dim[0]]
        q_data = mx.sym.FullyConnected(name='nms_query_' + str(index), data=roi_feat_reshape, num_hidden=dim[0])
        # q_data, [num_fg_classes, num_rois, group, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, num_rois, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(0, 2, 1, 3))
        # q_data_batch, [num_fg_classes * group, num_rois, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data_batch, shape=(-3, -2))
        k_data = mx.sym.FullyConnected(name='nms_key_' + str(index), data=roi_feat_reshape, num_hidden=dim[1])
        # k_data, [num_fg_classes, num_rois, group, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, num_rois, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(0, 2, 1, 3))
        # k_data_batch, [num_fg_classes * group, num_rois, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data_batch, shape=(-3, -2))
        v_data = roi_feat
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [num_fg_classes * group, num_rois, num_rois]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff

        assert fc_dim[1] == group, 'Check the dimensions in attention!'
        # [num_fg_classes * fc_dim[1], num_rois, num_rois]
        aff_weight_reshape = mx.sym.Reshape(aff_weight, shape=(-3, -2))
        # weighted_aff, [num_fg_classes * fc_dim[1], num_rois, num_rois]
        weighted_aff= mx.sym.log(mx.sym.maximum(left=aff_weight_reshape, right=1e-6)) + aff_scale
        # aff_softmax, [num_fg_classes * fc_dim[1], num_rois, num_rois]
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='nms_softmax_' + str(index))
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-1, fc_dim[1] * num_rois, 0))
        # output_t, [num_fg_classes, fc_dim[1] * num_rois, feat_dim]
        output_t = mx.symbol.batch_dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t_reshape, [num_fg_classes, fc_dim[1], num_rois, feat_dim]
        output_t_reshape = mx.sym.Reshape(output_t, shape=(-1, fc_dim[1], num_rois, feat_dim))
        # output_t_reshape, [fc_dim[1], feat_dim, num_rois, num_fg_classes]
        output_t_reshape = mx.sym.transpose(output_t_reshape, axes=(1, 3, 2, 0))
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output_t_reshape = mx.sym.Reshape(output_t_reshape, shape=(1, fc_dim[1] * feat_dim, num_rois, -1))
        linear_out = mx.symbol.Convolution(name='nms_linear_out_' + str(index),
                                           data=output_t_reshape,
                                           kernel=(1, 1), num_filter=dim[2], num_group=fc_dim[1])
        # [dim[2], num_rois, num_fg_classes]
        linear_out_reshape = mx.sym.Reshape(linear_out, shape=(dim[2], num_rois, -1))
        # [num_rois, num_fg_classes, dim[2]]
        output = mx.sym.transpose(linear_out_reshape, axes=(1, 2, 0))
        return output, aff_softmax

    def get_sorted_bbox_symbol(self, cfg, rois, cls_score, bbox_pred, im_info, suffix=1, is_train=True):


        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        nms_target_thresh = np.fromstring(cfg.network.NMS_TARGET_THRESH, dtype=float, sep=',')
        num_thresh = len(nms_target_thresh)
        nms_eps = 1e-8
        first_n = cfg.TRAIN.FIRST_N if is_train else cfg.TEST.FIRST_N
        num_fg_classes = num_classes - 1
        bbox_means = cfg.TRAIN.BBOX_MEANS if is_train else None
        bbox_stds = cfg.TRAIN.BBOX_STDS if is_train else None
        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N if is_train else cfg.TEST.RPN_POST_NMS_TOP_N

        # remove gt
        cls_score_nongt = mx.sym.slice_axis(data=cls_score, axis=0, begin=0, end=nongt_dim)
        bbox_pred_nongt = mx.sym.slice_axis(data=bbox_pred, axis=0, begin=0, end=nongt_dim)
        bbox_pred_nongt = mx.sym.BlockGrad(bbox_pred_nongt)

        # refine bbox
        # remove batch idx and gt roi
        # sliced_rois, [nongt_dim, 4]
        sliced_rois = mx.sym.slice(data=rois, begin=(0, 1), end=(nongt_dim, None))
        # bbox_pred_nobg, [num_rois, 4*(num_reg_classes-1)]
        bbox_pred_nobg = mx.sym.slice_axis(data=bbox_pred_nongt, axis=1, begin=4, end=None)
        # [num_boxes, 4, num_reg_classes-1]
        refined_bbox = NMS_UTILS.refine_bbox(sliced_rois, bbox_pred_nobg, im_info,
                                        means=bbox_means, stds=bbox_stds
                                        )
        # softmax cls_score to cls_prob, [num_rois, num_classes]
        cls_prob = mx.sym.softmax(data=cls_score_nongt, axis=-1)
        cls_prob_nobg = mx.sym.slice_axis(cls_prob, axis=1, begin=1, end=None)
        sorted_cls_prob_nobg = mx.sym.sort(data=cls_prob_nobg, axis=0, is_ascend=False)
        # sorted_score, [first_n, num_fg_classes]
        sorted_score = mx.sym.slice_axis(sorted_cls_prob_nobg, axis=0,
                                         begin=0, end=first_n, name='sorted_score_{}'.format(suffix))
        # sort by score
        rank_indices = mx.sym.argsort(data=cls_prob_nobg, axis=0, is_ascend=False)
        # first_rank_indices, [first_n, num_fg_classes]
        first_rank_indices = mx.sym.slice_axis(rank_indices, axis=0, begin=0, end=first_n)
        # sorted_bbox, [first_n, num_fg_classes, 4, num_reg_classes-1]
        sorted_bbox = mx.sym.take(a=refined_bbox, indices=first_rank_indices)
        if cfg.CLASS_AGNOSTIC:
            # sorted_bbox, [first_n, num_fg_classes, 4]
            sorted_bbox = mx.sym.Reshape(sorted_bbox, shape=(0, 0, 0), name='sorted_bbox_{}'.format(suffix))
        else:
            cls_mask = mx.sym.arange(0, num_fg_classes)
            cls_mask = mx.sym.Reshape(cls_mask, shape=(1, -1, 1))
            cls_mask = mx.sym.broadcast_to(cls_mask, shape=(first_n, 0, 4))
            # sorted_bbox, [first_n, num_fg_classes, 4]
            sorted_bbox = mx.sym.pick(data=sorted_bbox, name='sorted_bbox_{}'.format(suffix),
                                      index=cls_mask, axis=3)

        # # nms_rank_feat, [first_n, 1024]
        # nms_rank_feat = mx.sym.FullyConnected(name='nms_rank_{}'.format(suffix), data=nms_rank_embedding, num_hidden=128)

        # roi_feat_embedding = mx.sym.FullyConnected(
        #     name='roi_feat_embedding_{}'.format(suffix),
        #     data=fc_all_2_relu,
        #     num_hidden=128)
        # # sorted_roi_feat, [first_n, num_fg_classes, 128]
        # sorted_roi_feat = mx.sym.take(a=roi_feat_embedding, indices=first_rank_indices)

        # # nms_embedding_feat, [first_n, num_fg_classes, 128]
        # nms_embedding_feat = mx.sym.broadcast_add(
        #     lhs=sorted_roi_feat,
        #     rhs=mx.sym.expand_dims(nms_rank_feat, axis=1))

        # return sorted_bbox, sorted_score, nms_embedding_feat

        return sorted_bbox, sorted_score, first_rank_indices

    def get_train_symbol(self, cfg):

        num_anchors = cfg.network.NUM_ANCHORS
        is_train = True
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        gt_boxes = mx.sym.Variable(name="gt_boxes")
        rpn_label = mx.sym.Variable(name='label')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

        ref_data = mx.sym.Variable(name="ref_data")
        ref_im_info = mx.sym.Variable(name="ref_im_info")
        ref_gt_boxes = mx.sym.Variable(name="ref_gt_boxes")
        ref_rpn_label = mx.sym.Variable(name='ref_label')
        ref_rpn_bbox_target = mx.sym.Variable(name='ref_bbox_target')
        ref_rpn_bbox_weight = mx.sym.Variable(name='ref_bbox_weight')


        output_sym_list = []

        concat_data = mx.sym.concat(data, ref_data, dim=0)
        concat_gt_boxes = mx.sym.concat(gt_boxes, ref_gt_boxes, dim=0)
        concat_rpn_label = mx.sym.concat(rpn_label, ref_rpn_label, dim=0)
        concat_rpn_bbox_target = mx.sym.concat(rpn_bbox_target, ref_rpn_bbox_target, dim=0)
        concat_rpn_bbox_weight = mx.sym.concat(rpn_bbox_weight, ref_rpn_bbox_weight, dim=0)
        concat_rpn_cls_score, concat_rpn_bbox_pred, concat_rcnn_feat = self.get_rpn_symbol(concat_data, cfg)

        # prepare rpn data
        concat_rpn_cls_score_reshape = mx.sym.Reshape(
            data=concat_rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    
        # classification
        concat_rpn_cls_prob = mx.sym.SoftmaxOutput(data=concat_rpn_cls_score_reshape, label=concat_rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            concat_rpn_bbox_loss_ = concat_rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(concat_rpn_bbox_pred - concat_rpn_bbox_target))
            concat_rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=concat_rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
        else:
            concat_rpn_bbox_loss_ = concat_rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(concat_rpn_bbox_pred - concat_rpn_bbox_target))
        concat_rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=concat_rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        concat_rpn_cls_act = mx.sym.SoftmaxActivation(
            data=concat_rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        concat_rpn_cls_act_reshape = mx.sym.Reshape(
            data=concat_rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        rpn_cls_act_reshape, ref_rpn_cls_act_reshape = mx.sym.split(concat_rpn_cls_act_reshape, axis=0, num_outputs=2)
        rpn_bbox_pred, ref_rpn_bbox_pred = mx.sym.split(concat_rpn_bbox_pred, axis=0, num_outputs=2)
        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

        if cfg.TRAIN.CXX_PROPOSAL:
            ref_rois = mx.contrib.sym.Proposal(
                cls_prob=ref_rpn_cls_act_reshape, bbox_pred=ref_rpn_bbox_pred, im_info=ref_im_info, name='ref_rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            ref_rois = mx.sym.Custom(
                cls_prob=ref_rpn_cls_act_reshape, bbox_pred=ref_rpn_bbox_pred, im_info=ref_im_info, name='ref_rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

        # Proposal Target num ROI is 300 
         # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
        rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)


         # ROI proposal target
        ref_gt_boxes_reshape = mx.sym.Reshape(data=ref_gt_boxes, shape=(-1, 5), name='ref_gt_boxes_reshape')
        ref_rois, ref_label, ref_bbox_target, ref_bbox_weight = mx.sym.Custom(rois=ref_rois, gt_boxes=ref_gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        concat_rois = mx.sym.concat(rois, ref_rois, dim=0)
        concat_label = mx.sym.concat(label, ref_label, dim=0)
        concat_bbox_target = mx.sym.concat(bbox_target, ref_bbox_target, dim=0)
        concat_bbox_weight = mx.sym.concat(bbox_weight, ref_bbox_weight, dim=0)

        concat_conv_new_1 = mx.sym.Convolution(data=concat_rcnn_feat, kernel=(1, 1), num_filter=256, name="conv_new_1")
        concat_conv_new_1_relu = mx.sym.Activation(data=concat_conv_new_1, act_type='relu', name='conv_new_1_relu')

        offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=concat_conv_new_1_relu, rois=concat_rois, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

        concat_deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=concat_conv_new_1_relu, rois=concat_rois,
                                                                    trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)
    
        concat_fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=concat_deformable_roi_pool, num_hidden=1024)
        concat_fc_all_1 = concat_fc_new_1
        concat_fc_all_1_relu = mx.sym.Activation(data=concat_fc_all_1, act_type='relu', name='fc_all_1_relu')
        concat_fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=concat_fc_all_1_relu, num_hidden=1024)
        concat_fc_all_2 = concat_fc_new_2
        concat_fc_all_2_relu = mx.sym.Activation(data=concat_fc_all_2, act_type='relu', name='fc_all_2_relu')

        # cls_score/bbox_pred
        concat_cls_score = mx.sym.FullyConnected(name='cls_score', data=concat_fc_all_2_relu, num_hidden=num_classes)
        concat_bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=concat_fc_all_2_relu, num_hidden=num_reg_classes * 4)

        fc_all_2_relu, ref_fc_all_2_relu = mx.sym.split(concat_fc_all_2_relu, axis=0, num_outputs=2)
        cls_score, ref_cls_score = mx.sym.split(concat_cls_score, axis=0, num_outputs=2)
        bbox_pred, ref_bbox_pred = mx.sym.split(concat_bbox_pred, axis=0, num_outputs=2)

        # classification
        if cfg.TRAIN.ENABLE_OHEM:
            print 'use ohem!'
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            ref_labels_ohem, ref_bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=ref_cls_score, bbox_pred=ref_bbox_pred, labels=ref_label,
                                                           bbox_targets=ref_bbox_target, bbox_weights=ref_bbox_weight)
            concat_label_ohem = mx.sym.concat(labels_ohem, ref_labels_ohem)
            concat_bbox_weight_ohem = mx.sym.concat(bbox_weights_ohem, ref_bbox_weights_ohem)

            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=concat_cls_score, label=concat_labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
            bbox_loss_ = concat_bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(concat_bbox_pred - concat_bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=concat_cls_score, label=concat_label, normalization='valid')
            bbox_loss_ = concat_bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(concat_bbox_pred - concat_bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = concat_label

        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')

        output_sym_list = [concat_rpn_cls_prob, concat_rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]


        sorted_bbox, sorted_score, first_rank_indices = self.get_sorted_bbox_symbol(cfg, rois, cls_score, bbox_pred, im_info, 1)
        ref_sorted_bbox, ref_sorted_score, ref_first_rank_indices = self.get_sorted_bbox_symbol(cfg, ref_rois, ref_cls_score, ref_bbox_pred, ref_im_info, 2)

        # nms_rank_embedding, [first_n, 1024]
        first_n = cfg.TRAIN.FIRST_N
        nms_rank_embedding = NMS_UTILS.extract_rank_embedding(first_n, 1024)
        ref_nms_rank_embedding = NMS_UTILS.extract_rank_embedding(first_n, 1024)

        concat_sorted_bbox = mx.sym.concat(sorted_bbox, ref_sorted_bbox, dim=0)
        concat_sorted_score = mx.sym.concat(sorted_score, ref_sorted_score, dim=0)

        concat_nms_rank_embedding = mx.sym.concat(nms_rank_embedding, ref_nms_rank_embedding, dim=0)


        # nms_rank_feat, [first_n, 1024]
        concat_nms_rank_feat = mx.sym.FullyConnected(name='nms_rank', data=concat_nms_rank_embedding, num_hidden=128)

        concat_roi_feat_embedding = mx.sym.FullyConnected(
            name='roi_feat_embedding',
            data=concat_fc_all_2_relu,
            num_hidden=128)

        # sorted_roi_feat, [first_n, num_fg_classes, 128]
        roi_feat_embedding, ref_roi_feat_embedding = mx.sym.split(concat_roi_feat_embedding, axis=0, num_outputs=2)
        sorted_roi_feat = mx.sym.take(a=roi_feat_embedding, indices=first_rank_indices)
        ref_sorted_roi_feat = mx.sym.take(a=ref_roi_feat_embedding, indices=ref_first_rank_indices)
        concat_sorted_roi_feat = mx.sym.concat(sorted_roi_feat, ref_sorted_roi_feat, dim=0)

        # nms_embedding_feat, [first_n, num_fg_classes, 128]
        concat_nms_embedding_feat = mx.sym.broadcast_add(
            lhs=concat_sorted_roi_feat,
            rhs=mx.sym.expand_dims(concat_nms_rank_feat, axis=1))

        concat_nms_position_matrix = NMS_UTILS.extract_multi_position_matrix(concat_sorted_bbox)

        # 2*first_n hacking here
        nms_target_thresh = np.fromstring(cfg.network.NMS_TARGET_THRESH, dtype=float, sep=',')
        num_thresh = len(nms_target_thresh)
        nms_eps = 1e-8
        first_n = cfg.TRAIN.FIRST_N
        num_fg_classes = num_classes - 1
        bbox_means = cfg.TRAIN.BBOX_MEANS
        bbox_stds = cfg.TRAIN.BBOX_STDS 
        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N
        first_n *= 2
        # nms_attention_1, [first_n, num_fg_classes, 1024]
        concat_nms_attention_1, concat_nms_softmax_1 = self.attention_module_nms_multi_head(
            concat_nms_embedding_feat, concat_nms_position_matrix,
            num_rois=first_n, index=1, group=16,
            dim=(1024, 1024, 128), fc_dim=(64, 16), feat_dim=128)
        concat_nms_all_feat_1 = concat_nms_embedding_feat + concat_nms_attention_1
        concat_nms_all_feat_1_relu = mx.sym.Activation(data=concat_nms_all_feat_1, act_type='relu', name='nms_all_feat_1_relu')
        # [first_n * num_fg_classes, 1024]
        concat_nms_all_feat_1_relu_reshape = mx.sym.Reshape(concat_nms_all_feat_1_relu, shape=(-3, -2))
        # logit, [first_n * num_fg_classes, num_thresh]
        concat_nms_conditional_logit = mx.sym.FullyConnected(name='nms_logit',
                                                  data=concat_nms_all_feat_1_relu_reshape,
                                                  num_hidden=num_thresh)
        # logit_reshape, [first_n, num_fg_classes, num_thresh]
        concat_nms_conditional_logit_reshape = mx.sym.Reshape(concat_nms_conditional_logit,
                                                   shape=(first_n, num_fg_classes, num_thresh))
        concat_nms_conditional_score = mx.sym.Activation(data=concat_nms_conditional_logit_reshape,
                                              act_type='sigmoid', name='nms_conditional_score')
        concat_sorted_score_reshape = mx.sym.expand_dims(concat_sorted_score, axis=2)
        # sorted_score_reshape = mx.sym.BlockGrad(sorted_score_reshape)
        concat_nms_multi_score = mx.sym.broadcast_mul(lhs=concat_sorted_score_reshape, rhs=concat_nms_conditional_score)

        nms_multi_target = mx.sym.Custom(bbox=sorted_bbox, gt_bbox=gt_boxes, 
                                         score=sorted_score,
                                         ref_bbox=ref_sorted_bbox, ref_gt_bbox = ref_gt_boxes, 
                                         ref_score = ref_sorted_score,
                                         op_type='nms_multi_target', target_thresh=nms_target_thresh)
        nms_pos_loss = - mx.sym.broadcast_mul(lhs=nms_multi_target,
                                              rhs=mx.sym.log(data=(concat_nms_multi_score + nms_eps)))
        nms_neg_loss = - mx.sym.broadcast_mul(lhs=(1.0 - nms_multi_target),
                                              rhs=mx.sym.log(data=(1.0 - concat_nms_multi_score + nms_eps)))
        normalizer = first_n * num_thresh
        nms_pos_loss = cfg.TRAIN.nms_loss_scale * nms_pos_loss / normalizer
        nms_neg_loss = cfg.TRAIN.nms_loss_scale * nms_neg_loss / normalizer
        ##########################  additional output!  ##########################
        output_sym_list.append(mx.sym.BlockGrad(nms_multi_target, name='nms_multi_target_block'))
        output_sym_list.append(mx.sym.BlockGrad(concat_nms_conditional_score, name='nms_conditional_score_block'))
        output_sym_list.append(mx.sym.MakeLoss(name='nms_pos_loss', data=nms_pos_loss,
                                               grad_scale=cfg.TRAIN.nms_pos_scale))
        output_sym_list.append(mx.sym.MakeLoss(name='nms_neg_loss', data=nms_neg_loss))

        self.sym = mx.sym.Group(output_sym_list)
        return self.sym

    def get_test_symbol(self, cfg):

        num_anchors = cfg.network.NUM_ANCHORS
        is_train = False
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        ref_data = mx.sym.Variable(name="ref_data")
        ref_im_info = mx.sym.Variable(name="ref_im_info")

        gt_boxes = mx.sym.Variable(name='gt_boxes')
        ref_gt_boxes = mx.sym.Variable(name='ref_gt_boxes')


        im_info_list = [im_info, ref_im_info]

        output_sym_list = []

        sorted_bbox_feat_list = []

        fc_list = []

        concat_data = mx.sym.concat(data, ref_data, dim=0)
        concat_rpn_cls_score, concat_rpn_bbox_pred, concat_rcnn_feat = self.get_rpn_symbol(concat_data, cfg)
        # rpn_cls_score, rpn_bbox_pred, rcnn_feat = self.get_rpn_symbol(data, cfg)
        # ref_rpn_cls_score, ref_rpn_bbox_pred, ref_rcnn_feat = self.get_rpn_symbol(ref_data, cfg)
        # concat_rpn_cls_score = mx.sym.concat(rpn_cls_score, ref_rpn_cls_score, dim=0)
        # concat_rpn_bbox_pred = mx.sym.concat(rpn_bbox_pred, ref_rpn_bbox_pred, dim=0)
        # concat_rcnn_feat = mx.sym.concat(rcnn_feat, ref_rcnn_feat, dim=0)

        # prepare rpn data
        concat_rpn_cls_score_reshape = mx.sym.Reshape(
            data=concat_rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

        # ROI proposal
        concat_rpn_cls_act = mx.sym.SoftmaxActivation(
            data=concat_rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        concat_rpn_cls_act_reshape = mx.sym.Reshape(
            data=concat_rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        rpn_cls_act_reshape, ref_rpn_cls_act_reshape = mx.sym.split(concat_rpn_cls_act_reshape, axis=0, num_outputs=2)
        rpn_bbox_pred, ref_rpn_bbox_pred = mx.sym.split(concat_rpn_bbox_pred, axis=0, num_outputs=2)
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        if cfg.TEST.CXX_PROPOSAL:
            ref_rois = mx.contrib.sym.Proposal(
                cls_prob=ref_rpn_cls_act_reshape, bbox_pred=ref_rpn_bbox_pred, im_info=ref_im_info, name='ref_rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            ref_rois = mx.sym.Custom(
                cls_prob=ref_rpn_cls_act_reshape, bbox_pred=ref_rpn_bbox_pred, im_info=ref_im_info, name='ref_rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)


        concat_rois = mx.sym.concat(rois, ref_rois, dim=0, name='concat_rois')

        concat_conv_new_1 = mx.sym.Convolution(data=concat_rcnn_feat, kernel=(1, 1), num_filter=256, name="conv_new_1")
        concat_conv_new_1_relu = mx.sym.Activation(data=concat_conv_new_1, act_type='relu', name='conv_new_1_relu')

        offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=concat_conv_new_1_relu, rois=concat_rois, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

        concat_deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=concat_conv_new_1_relu, rois=concat_rois,
                                                                    trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)
    
        concat_fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=concat_deformable_roi_pool, num_hidden=1024)
        concat_fc_all_1 = concat_fc_new_1
        concat_fc_all_1_relu = mx.sym.Activation(data=concat_fc_all_1, act_type='relu', name='fc_all_1_relu')
        concat_fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=concat_fc_all_1_relu, num_hidden=1024)
        concat_fc_all_2 = concat_fc_new_2
        concat_fc_all_2_relu = mx.sym.Activation(data=concat_fc_all_2, act_type='relu', name='fc_all_2_relu')

        # cls_score/bbox_pred
        concat_cls_score = mx.sym.FullyConnected(name='cls_score', data=concat_fc_all_2_relu, num_hidden=num_classes)
        concat_bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=concat_fc_all_2_relu, num_hidden=num_reg_classes * 4)

        concat_cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=concat_cls_score)
        concat_cls_prob = mx.sym.Reshape(data=concat_cls_prob, shape=(2*cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        concat_bbox_pred_reshape = mx.sym.Reshape(data=concat_bbox_pred, name='bbox_pred_reshape',
                                           shape=(2*cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes))

        output_sym_list = [concat_rois, concat_cls_prob, concat_bbox_pred_reshape]

        fc_all_2_relu, ref_fc_all_2_relu = mx.sym.split(concat_fc_all_2_relu, axis=0, num_outputs=2)
        cls_score, ref_cls_score = mx.sym.split(concat_cls_score, axis=0, num_outputs=2)
        bbox_pred, ref_bbox_pred = mx.sym.split(concat_bbox_pred, axis=0, num_outputs=2)

        sorted_bbox, sorted_score, first_rank_indices = self.get_sorted_bbox_symbol(cfg, rois, cls_score, bbox_pred, im_info, 1)
        ref_sorted_bbox, ref_sorted_score, ref_first_rank_indices = self.get_sorted_bbox_symbol(cfg, ref_rois, ref_cls_score, ref_bbox_pred, ref_im_info, 2)

        concat_sorted_bbox = mx.sym.concat(sorted_bbox, ref_sorted_bbox, dim=0, name='concat_sorted_bbox')
        concat_sorted_score = mx.sym.concat(sorted_score, ref_sorted_score, dim=0, name='concat_sorted_score')

        first_n = cfg.TEST.FIRST_N
        nms_rank_embedding = NMS_UTILS.extract_rank_embedding(first_n)
        ref_nms_rank_embedding = NMS_UTILS.extract_rank_embedding(first_n)
        concat_nms_rank_embedding = mx.sym.concat(nms_rank_embedding, ref_nms_rank_embedding, dim=0)


        # nms_rank_feat, [first_n, 1024]
        concat_nms_rank_feat = mx.sym.FullyConnected(name='nms_rank', data=concat_nms_rank_embedding, num_hidden=128)

        concat_roi_feat_embedding = mx.sym.FullyConnected(
            name='roi_feat_embedding',
            data=concat_fc_all_2_relu,
            num_hidden=128)

        # sorted_roi_feat, [first_n, num_fg_classes, 128]
        roi_feat_embedding, ref_roi_feat_embedding = mx.sym.split(concat_roi_feat_embedding, axis=0, num_outputs=2)
        sorted_roi_feat = mx.sym.take(a=roi_feat_embedding, indices=first_rank_indices)
        ref_sorted_roi_feat = mx.sym.take(a=ref_roi_feat_embedding, indices=ref_first_rank_indices)
        concat_sorted_roi_feat = mx.sym.concat(sorted_roi_feat, ref_sorted_roi_feat, dim=0)

        # nms_embedding_feat, [first_n, num_fg_classes, 128]
        concat_nms_embedding_feat = mx.sym.broadcast_add(
            lhs=concat_sorted_roi_feat,
            rhs=mx.sym.expand_dims(concat_nms_rank_feat, axis=1))

        concat_nms_position_matrix = NMS_UTILS.extract_multi_position_matrix(concat_sorted_bbox)

        # 2*first_n hacking here
        nms_target_thresh = np.fromstring(cfg.network.NMS_TARGET_THRESH, dtype=float, sep=',')
        num_thresh = len(nms_target_thresh)
        nms_eps = 1e-8
        first_n = cfg.TEST.FIRST_N
        num_fg_classes = num_classes - 1
        bbox_means = None
        bbox_stds = None 
        nongt_dim = cfg.TEST.RPN_POST_NMS_TOP_N
        first_n *= 2
        # nms_attention_1, [first_n, num_fg_classes, 1024]
        concat_nms_attention_1, concat_nms_softmax_1 = self.attention_module_nms_multi_head(
            concat_nms_embedding_feat, concat_nms_position_matrix,
            num_rois=first_n, index=1, group=16,
            dim=(1024, 1024, 128), fc_dim=(64, 16), feat_dim=128)
        concat_nms_all_feat_1 = concat_nms_embedding_feat + concat_nms_attention_1
        concat_nms_all_feat_1_relu = mx.sym.Activation(data=concat_nms_all_feat_1, act_type='relu', name='nms_all_feat_1_relu')
        # [first_n * num_fg_classes, 1024]
        concat_nms_all_feat_1_relu_reshape = mx.sym.Reshape(concat_nms_all_feat_1_relu, shape=(-3, -2))
        # logit, [first_n * num_fg_classes, num_thresh]
        concat_nms_conditional_logit = mx.sym.FullyConnected(name='nms_logit',
                                                  data=concat_nms_all_feat_1_relu_reshape,
                                                  num_hidden=num_thresh)
        # logit_reshape, [first_n, num_fg_classes, num_thresh]
        concat_nms_conditional_logit_reshape = mx.sym.Reshape(concat_nms_conditional_logit,
                                                   shape=(first_n, num_fg_classes, num_thresh))
        concat_nms_conditional_score = mx.sym.Activation(data=concat_nms_conditional_logit_reshape,
                                              act_type='sigmoid', name='nms_conditional_score')
        concat_sorted_score_reshape = mx.sym.expand_dims(concat_sorted_score, axis=2)
        # sorted_score_reshape = mx.sym.BlockGrad(sorted_score_reshape)

        # testing nms_multi_target
        # concat_nms_multi_score = mx.sym.broadcast_mul(lhs=concat_sorted_score_reshape, rhs=concat_nms_conditional_score)
        concat_nms_multi_score = concat_sorted_score_reshape

        nms_multi_target = mx.sym.Custom(bbox=sorted_bbox, gt_bbox=gt_boxes, 
                                         score=sorted_score,
                                         ref_bbox=ref_sorted_bbox, ref_gt_bbox = ref_gt_boxes, 
                                         ref_score = ref_sorted_score,
                                         op_type='nms_multi_target', target_thresh=nms_target_thresh)

        if cfg.TEST.MERGE_METHOD == -1:
            nms_final_score = mx.sym.mean(data=concat_nms_multi_score, axis=2, name='nms_final_score')
        elif cfg.TEST.MERGE_METHOD == -2:
            nms_final_score = mx.sym.max(data=concat_nms_multi_score, axis=2, name='nms_final_score')
        elif 0 <= cfg.TEST.MERGE_METHOD < num_thresh:
            idx = cfg.TEST.MERGE_METHOD
            nms_final_score = mx.sym.slice_axis(data=concat_nms_multi_score, axis=2, begin=idx, end=idx + 1)
            nms_final_score = mx.sym.Reshape(nms_final_score, shape=(0, 0), name='nms_final_score')
        else:
            raise NotImplementedError('Unknown merge method %s.' % cfg.TEST.MERGE_METHOD)

        output_sym_list.append(concat_sorted_bbox)
        output_sym_list.append(concat_sorted_score)
        output_sym_list.append(nms_final_score)
        output_sym_list.append(nms_multi_target)


        self.sym = mx.sym.Group(output_sym_list)
        return self.sym

    def init_weight_attention_nms_multi_head(self, cfg, arg_params, aux_params, index=1):
        arg_params['nms_pair_pos_fc1_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_weight'])
        arg_params['nms_pair_pos_fc1_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_bias'])
        arg_params['nms_query_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_query_' + str(index) + '_weight'])
        arg_params['nms_query_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_query_' + str(index) + '_bias'])
        arg_params['nms_key_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_key_' + str(index) + '_weight'])
        arg_params['nms_key_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_key_' + str(index) + '_bias'])
        arg_params['nms_linear_out_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_weight'])
        arg_params['nms_linear_out_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_bias'])

    # def init_weight_nms(self, cfg, arg_params,aux_params, index=1):
    #     arg_params['nms_rank_{}_weight'.format(index)] = mx.random.normal(
    #         0, 0.01, shape=self.arg_shape_dict['nms_rank_{}_weight'.format(index)])
    #     arg_params['nms_rank_{}_bias'.format(index)] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_{}_bias'.format(index)])
    #     arg_params['roi_feat_embedding_{}_weight'.format(index)] = mx.random.normal(
    #         0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_{}_weight'.format(index)])
    #     arg_params['roi_feat_embedding_{}_bias'.format(index)] = mx.nd.zeros(
    #         shape=self.arg_shape_dict['roi_feat_embedding_{}_bias'.format(index)])
    #     arg_params['nms_logit_weight'] = mx.random.normal(
    #         0, 0.01, shape=self.arg_shape_dict['nms_logit_weight'])
    #     arg_params['nms_logit_bias'] = mx.nd.full(shape=self.arg_shape_dict['nms_logit_bias'], val=-3.0)

    #     self.init_weight_attention_nms_multi_head(cfg, arg_params, aux_params, index=1)

    # def init_weight_embedding(self, cfg, arg_params, aux_params, index=1):
    #     arg_params['nms_rank_{}_weight'.format(index)] = mx.random.normal(
    #         0, 0.01, shape=self.arg_shape_dict['nms_rank_{}_weight'.format(index)])
    #     arg_params['nms_rank_{}_bias'.format(index)] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_{}_bias'.format(index)])
    #     arg_params['roi_feat_embedding_{}_weight'.format(index)] = mx.random.normal(
    #         0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_{}_weight'.format(index)])
    #     arg_params['roi_feat_embedding_{}_bias'.format(index)] = mx.nd.zeros(
    #         shape=self.arg_shape_dict['roi_feat_embedding_{}_bias'.format(index)])

    def init_weight_embedding(self, cfg, arg_params, aux_params):
        arg_params['nms_rank_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_rank_weight'])
        arg_params['nms_rank_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_bias'])
        arg_params['roi_feat_embedding_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_weight'])
        arg_params['roi_feat_embedding_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['roi_feat_embedding_bias'])

    def init_weight_nms(self, cfg, arg_params, aux_params):
        # arg_params['nms_rank_weight'] = mx.random.normal(
        #     0, 0.01, shape=self.arg_shape_dict['nms_rank_weight'])
        # arg_params['nms_rank_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_bias'])
        # arg_params['roi_feat_embedding_weight'] = mx.random.normal(
        #     0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_weight'])
        # arg_params['roi_feat_embedding_bias'] = mx.nd.zeros(
        #     shape=self.arg_shape_dict['roi_feat_embedding_bias'])
        # self.init_weight_embedding(cfg, arg_params, aux_params, index=1)
        # self.init_weight_embedding(cfg, arg_params, aux_params, index=2)
        self.init_weight_embedding(cfg, arg_params, aux_params)
        arg_params['nms_logit_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_logit_weight'])
        arg_params['nms_logit_bias'] = mx.nd.full(shape=self.arg_shape_dict['nms_logit_bias'], val=-3.0)
        self.init_weight_attention_nms_multi_head(cfg, arg_params, aux_params, index=1)

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_nms(self, arg_params, aux_params)
        # self.init_weight_rpn(self, arg_params, aux_params)
        # self.init_weight_rcnn(self, arg_params, aux_params)