# Frozen 3D VLM eval on ScanNet200: ScanNet-20 as base; novel = CLASS_LABELS_BASE_NOVEL
# minus those bases (CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL), not the full 180 tail.
# Use with: python tools/eval_vlm_novel.py --config-file configs/scannet/vlm-eval-scannet200-sc20base-180novel.py ...

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL,
    CLASS_LABELS_SC20_IN_SC200_ORDER,
)

_base_ = ["../_base_/default_runtime.py"]

test = dict(type="VLM3DZeroShotGFSTester")

task_id = 0  # unused for this split; kept for config compatibility

epoch = 1
eval_epoch = 1
batch_size = 1
num_worker = 8
empty_cache = False
weight = ""

vlm_3d_weight = (
    "_experiments/pretrained_weights/3d_vlm_weight/sparseunet32_636.pth"
)

_num_bn_novel = len(CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL)
_num_classes = 20 + _num_bn_novel

model = dict(
    type="RegisTrainSegmentor",
    num_base_classes=20,
    num_novel_classes=_num_bn_novel,
    backbone_out_channels=64,
    backbone=dict(type="PT-v3m1"),
)

data_root = "_datasets/ScanNet200"
_class_names = list(CLASS_LABELS_SC20_IN_SC200_ORDER) + list(
    CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL
)

data = dict(
    num_bases=20,
    num_base_novels=_num_classes,
    ignore_index=-1,
    names=_class_names,
    train=dict(
        type="ScanNet200Dataset_BASETrain",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "color"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="ScanNet200Dataset_TEST_ScanNet20Base",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "color"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
