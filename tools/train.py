"""
Main Training Script

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
    make_exp_dir_and_save_config
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS20_LABELS_NOVEL,
    CLASS20_LABELS_NOVEL_FLAT,
    CLASS20_LABELS_BASE,
    CLASS20_LABELS_BASE_NOVEL,
)

import logging
from pointcept.utils import logger as pc_logger


def reset_pointcept_logger():
    root_logger = logging.getLogger("pointcept")
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    for name in list(pc_logger.logger_initialized.keys()):
        if name == "pointcept" or name.startswith("pointcept."):
            pc_logger.logger_initialized.pop(name, None)


def main_worker(cfg):
    reset_pointcept_logger()
    cfg = default_setup(cfg)
    root_logger = pc_logger.get_root_logger(
        log_file=os.path.join(cfg.save_path, "train.log"),
        file_mode="a" if cfg.resume else "w",
    )
    root_logger.info(f"====Training task {cfg.task_id}====")
    root_logger.info(f"Config has been updated for task {cfg.task_id}")
    root_logger.info(f"Config has been saved here: {cfg.save_path}")

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()
    root_logger.info(f"Training task {cfg.task_id} completed")


def _setup_ifs_incremental_task(cfg, task_id):
    """IFS: grow label space per incremental novel task."""
    cfg.task_id = task_id
    all_novel_classes = [
        c for task_cls in CLASS20_LABELS_NOVEL[: task_id + 1] for c in task_cls
    ]
    cfg.model["num_all_novel_classes"] = len(all_novel_classes)
    cfg.model["num_novel_classes"] = len(CLASS20_LABELS_NOVEL[task_id])
    cfg.data["num_base_novels"] = cfg.data["num_bases"] + len(all_novel_classes)
    cfg.data["names"] = list(CLASS20_LABELS_BASE) + all_novel_classes
    cfg.save_path = os.path.join(cfg.save_path, f"task-{task_id:02d}")
    make_exp_dir_and_save_config(cfg)
    return cfg


def _setup_gfs_run(cfg):
    """GFS: single run with full base+novel label space from the config."""
    if getattr(cfg, "task_id", -1) < 0:
        cfg.task_id = 0
    cfg.model["num_novel_classes"] = len(CLASS20_LABELS_NOVEL_FLAT)
    cfg.data["num_base_novels"] = len(CLASS20_LABELS_BASE_NOVEL)
    cfg.data["names"] = list(CLASS20_LABELS_BASE_NOVEL)
    make_exp_dir_and_save_config(cfg)
    return cfg


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    trainer_type = cfg.train.type
    incremental = getattr(cfg, "incremental", trainer_type == "IFS_VL_Trainer")

    if incremental and trainer_type != "IFS_VL_Trainer":
        raise ValueError(
            f"incremental=True requires IFS_VL_Trainer, got {trainer_type}"
        )

    launch_kwargs = {
        "num_gpus_per_machine": args.num_gpus,
        "num_machines": args.num_machines,
        "machine_rank": args.machine_rank,
        "dist_url": args.dist_url,
    }
    if incremental:
        # task_id == -1: all incremental tasks; 0..N-1: one task only
        if getattr(cfg, "task_id", -1) == -1:
            for task_id in range(len(CLASS20_LABELS_NOVEL)):
                task_cfg = default_config_parser(args.config_file, args.options)
                _setup_ifs_incremental_task(task_cfg, task_id)
                launch(
                    main_worker,
                    **launch_kwargs,
                    cfg=(task_cfg,),
                )
        else:
            _setup_ifs_incremental_task(cfg, cfg.task_id)
            launch(
                main_worker,
                **launch_kwargs,
                cfg=(cfg,),
            )
    else:
        # GFS_VL_Trainer: one job, full label space from config (e.g. CLASS20_LABELS_BASE_NOVEL)
        _setup_gfs_run(cfg)
        launch(
            main_worker,
            **launch_kwargs,
            cfg=(cfg,),
        )


if __name__ == "__main__":
    main()
