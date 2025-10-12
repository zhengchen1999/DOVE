import os
import torch
import argparse
import imageio.v3 as iio
from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype
from tqdm import tqdm

def folder_to_video(input_root, output_root, fps, txt_dir):
    os.makedirs(output_root, exist_ok=True)
    to_tensor = ConvertImageDtype(torch.uint8)

    subfolders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

    if '.txt' in txt_dir:
        out_dir = txt_dir
    else:
        out_dir = os.path.join(txt_dir, 'video.txt')

    with open(out_dir, 'w') as f:
        for folder_name in tqdm(subfolders, desc='Processing folders'):
            folder_path = os.path.join(input_root, folder_name)
            parent_folder = os.path.basename(input_root)
            image_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
            ])
            if not image_files:
                continue

            frames = []
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                img = read_image(img_path).permute(1, 2, 0).byte().numpy()  # [H, W, C] uint8
                frames.append(img)

            output_video_path = os.path.join(output_root, f"{folder_name}.mp4")

            iio.imwrite(
                output_video_path,
                frames,
                fps=fps,
                codec='libx264',
                pixelformat='yuv444p',
                macro_block_size=None,
                ffmpeg_params=['-crf', '0'],
            )

            f.write(f"{parent_folder}/{folder_name}.mp4\n")
            print(f"Saved video: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert folders of images into videos.")
    parser.add_argument('--input_root', type=str, required=True, help='Root directory containing subfolders of images')
    parser.add_argument('--output_root', type=str, required=True, help='Directory to save the output videos')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for the output videos')
    parser.add_argument('--txt_dir', type=str, help='Path to save the video path list')
    args = parser.parse_args()

    folder_to_video(args.input_root, args.output_root, args.fps, args.txt_dir)

if __name__ == '__main__':
    main()
