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
    CLASS20_LABELS_BASE,
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


def main():
    args = default_argument_parser().parse_args()

    # NOTE: Since the novel has seprate list the base task is not considered here.
    # Will start form task 0 to task N (In practice base should be 0 and then novel 1 to N-1)
    # if task_id != -1:
    start_task_id = 0
    num_tasks = len(CLASS20_LABELS_NOVEL)
    for task_id in range(start_task_id, num_tasks):

        cfg = default_config_parser(args.config_file, args.options)
        cfg.task_id = task_id
        all_novel_classes = [c for task_cls in CLASS20_LABELS_NOVEL[:task_id+1] for c in task_cls]
        cfg.model["num_all_novel_classes"] = len(all_novel_classes)
        cfg.data["num_base_novels"] = cfg.data["num_bases"] + len(all_novel_classes)
        cfg.data["names"] = list(CLASS20_LABELS_BASE) + all_novel_classes
        cfg.save_path = os.path.join(cfg.save_path, f"task-{task_id:02d}")
        make_exp_dir_and_save_config(cfg)

        launch(
            main_worker,
            num_gpus_per_machine=args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            cfg=(cfg,),
        )


    # else:
    #     launch(
    #         main_worker,
    #         num_gpus_per_machine=args.num_gpus,
    #         num_machines=args.num_machines,
    #         machine_rank=args.machine_rank,
    #         dist_url=args.dist_url,
    #         cfg=(cfg,),
    #     )


if __name__ == "__main__":
    main()
