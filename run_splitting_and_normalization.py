import argparse
import logging
import os
import shutil
from typing import Union

from util.config import load_config, check_config_path
from util.log import enable_logging
from util.normalizers.factory import normalizer_factory
from util.splitters.factory import splitter_factory
from util.splitters.splitting_script import split_per_recording

if __name__ == "__main__":
    enable_logging()
    parser = argparse.ArgumentParser(
        description="Run the data splitting and normalization"
        " for the ICASSP challenge",
    )
    parser.add_argument("--config", default=None)

    parser.add_argument(
        "--speech-features",
        choices=["envelope", "mel"],
        nargs="+",
        default=["envelope"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=Union[float, int],
        help="Split data after preprocessing in sets (train/val/test).",
        default=[80, 10, 10],
    )
    parser.add_argument(
        "--no-cut-to-shortest-length", action="store_true", default=False
    )
    parser.add_argument(
        "--split-names",
        nargs="+",
        default=["train", "val", "test"],
    )
    parser.add_argument("--splitter", default="sequential")
    parser.add_argument("--normalizer", default="standardizer")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-from-scratch", action="store_true")
    args = parser.parse_args()

    config_path = check_config_path(args.config)
    config = load_config(config_path)
    if args.start_from_scratch:
        logging.warning(
            f"This will delete all the data in {config['split_folder']}. "
            "Are you sure you want to do this?"
        )
        confirmation = input("Type 'yes' to confirm...")
        if confirmation.lower() != "yes":
            logging.warning("Stopping")
            exit()
        else:
            split_path = os.path.join(
                config["dataset_folder"],
                config["split_folder"],
            )
            if os.path.exists(split_path):
                shutil.rmtree(split_path)
            logging.info(f"Deleted {split_path}")

    normalizer = normalizer_factory(args.normalizer)

    splitter = splitter_factory(
        args.splitter,
        split_data_folder=os.path.join(
            config["dataset_folder"], config["split_folder"]
        ),
        splits=args.splits,
        split_names=args.split_names,
        normalizer=normalizer,
        overwrite=args.overwrite,
    )

    split_per_recording(
        splitter,
        config,
        args.speech_features,
        not args.no_cut_to_shortest_length,
    )
