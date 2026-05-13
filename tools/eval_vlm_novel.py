"""
Evaluate frozen 3D VLM on the full test split (no trained seg checkpoint).

Two label splits (see --split):

  sc20_180 (default): ScanNet200 with ScanNet-20 as base and novel classes from
    CLASS_LABELS_BASE_NOVEL not in that base (CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL).
    Use config
    configs/scannet/vlm-eval-scannet200-sc20base-180novel.py (or equivalent).

  ifs: incremental few-shot setup from CLASS20_LABELS_NOVEL; requires --task-id
    and a config such as semseg-pt-v3m1-0-gfsregistrain_k5.py.

Logs: {save_path}/test.log
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS20_LABELS_BASE,
    CLASS20_LABELS_NOVEL,
)
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.test import TESTERS


def main_worker(cfg):
    cfg = default_setup(cfg)
    cfg.test = dict(type="VLM3DZeroShotGFSTester")
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    parser = default_argument_parser()
    parser.add_argument(
        "--split",
        type=str,
        choices=("sc20_180", "ifs"),
        default=None,
        help=        "sc20_180: ScanNet-20 base + BASE_NOVEL-complement novel set on ScanNet200 (single run; "
        "see vlm-eval-scannet200-sc20base-180novel.py). "
        "ifs: incremental CLASS20_LABELS_NOVEL (--task-id required). "
        "If omitted: ifs when --task-id is set, else sc20_180.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="IFS task index (required for --split ifs).",
    )
    args = parser.parse_args()

    split = args.split
    if split is None:
        split = "ifs" if args.task_id is not None else "sc20_180"

    cfg = default_config_parser(args.config_file, args.options)

    if split == "ifs":
        if args.task_id is None:
            parser.error("IFS split requires --task-id (or pass --split sc20_180).")
        cfg.task_id = args.task_id
        all_novel_classes = [
            c
            for task_cls in CLASS20_LABELS_NOVEL[: cfg.task_id + 1]
            for c in task_cls
        ]
        cfg.model["num_all_novel_classes"] = len(all_novel_classes)
        cfg.model["num_novel_classes"] = len(CLASS20_LABELS_NOVEL[cfg.task_id])
        cfg.data["num_base_novels"] = len(CLASS20_LABELS_BASE) + len(
            all_novel_classes
        )
        cfg.data["names"] = list(CLASS20_LABELS_BASE) + all_novel_classes
    else:
        cfg.task_id = 0

    os.makedirs(cfg.save_path, exist_ok=True)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
