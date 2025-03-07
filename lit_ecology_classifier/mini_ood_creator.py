from pathlib import Path
import os
import shutil


def create_mini_ood(data_dir: Path, foldernames, new_folder_name: str = "mini_ood"):
    """Give a data directory and a list of foldernames(subfolder) 
    to create a new copy with only the specified folders

    Args:
        data_dir (Path): The data directory
        foldernames (list): A list of foldernames to copy
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    new_dir = data_dir.parent / new_folder_name
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)
    for folder in foldernames:
        shutil.copytree(data_dir / folder, new_dir / folder)
    return new_dir

if __name__ == "__main__":
    data_dir = Path("data/OOD")
    foldernames = ["dinobryon", "keratella_quadrata", "uroglena"]
    new_dir = create_mini_ood(data_dir, foldernames, new_folder_name="mini_ood_2")
    print(f"New directory created at {new_dir}")