import os
import argparse

def list_videos_images(root_dir, output_file):
    # Supported video formats
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 'jpg', '.jpeg', '.png', '.bmp', '.tiff')

    video_paths = []

    # Traverse the folder
    for dirpath, _, filenames in os.walk(root_dir):
        for f in sorted(filenames):
            if f.lower().endswith(video_exts):
                abs_path = os.path.join(dirpath, f)
                # Relative path including the parent folder
                rel_path = os.path.relpath(abs_path, os.path.dirname(root_dir))
                video_paths.append(rel_path)

    # Save paths to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in video_paths:
            f.write(path + '\n')

    print(f"Saved {len(video_paths)} video paths to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='List all video files in a given folder (paths and txt include parent folder)')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing videos or images')
    parser.add_argument('--save', type=str, default=None, help='Path to save the txt file (default: parent_dir/<folder>.txt)')
    args = parser.parse_args()

    # Default save path: one level above the input directory
    if args.save is None:
        folder_name = os.path.basename(os.path.normpath(args.dir))
        args.save = os.path.join(os.path.dirname(args.dir), f"{folder_name}.txt")

    list_videos_images(args.dir, args.save)

if __name__ == '__main__':
    main()
