"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""
from tqdm import tqdm 
import shutil
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import os

REPO_NAME = "your_hf_username/calvin"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
   "training"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    lang_dir = "task_ABC_D/training/lang_annotations/auto_lang_ann.npy"
    lang_data = np.load(lang_dir,allow_pickle=True).item()
    ep_start_end_ids = lang_data["info"]["indx"]
    #prompt
    lang_ann = lang_data["language"]["ann"]
    #think
    lang_task = lang_data["language"]["task"]
    # import ipdb;ipdb.set_trace()

    for raw_dataset_name in RAW_DATASET_NAMES:
        data_dir_npz = Path(data_dir,raw_dataset_name)
        # raw_dataset = sorted(data_dir_npz.rglob("*.npz"))  # 递归查找所有 .npz
        # import ipdb;ipdb.set_trace()
        for lan,task, idx in tqdm(zip(lang_ann,lang_task, ep_start_end_ids), total=len(lang_ann), desc="Processing episodes"):
            for i in range(idx[0], idx[1] + 10):
                npz_name = f'episode_{i:07d}.npz'
                npz_file = os.path.join(data_dir_npz, npz_name)
                # 
                
                # Check that the file exists
                assert os.path.exists(npz_file), f"{npz_file} does not exist"
                
                # Load the .npz file
                episode_data = np.load(npz_file)

                dataset.add_frame(
                    {
                        "image": episode_data["rgb_static"],
                        "wrist_image": episode_data["rgb_gripper"],
                        "state": episode_data["robot_obs"][:7].astype(np.float32),
                        "actions": episode_data["rel_actions"].astype(np.float32),
                        "task": lan,
                        # "thought": task,
                    }
                )
            i = i+1
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
