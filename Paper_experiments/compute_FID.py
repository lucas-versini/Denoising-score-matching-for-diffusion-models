import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
import os

def calculate_fid(relative_path_1, relative_path_2):
    path = os.getcwd()
    path_1 = os.path.join(path, relative_path_1)
    path_2 = os.path.join(path, relative_path_2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fid_value = calculate_fid_given_paths([path_1, path_2], batch_size = 128, device = device, dims = 2048)
    return fid_value

if __name__ == "__main__":
    relative_path_to_folder1 = "exp/image_samples/cifar_sigmoid_samples"
    relative_path_to_folder2 = "cifar10_images"

    fid_score = calculate_fid(relative_path_to_folder1, relative_path_to_folder2)
    print(f"The FID score is {fid_score:.3f}")
