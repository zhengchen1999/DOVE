import os
import shutil
import json
import argparse
import subprocess
from pathlib import Path

def run_zero_to_fp32(checkpoint_dir, output_dir):
    print("Running zero_to_fp32.py with --safe_serialization...")
    subprocess.run([
        "python3", "scripts/zero_to_fp32.py",
        checkpoint_dir,
        output_dir,
        "--safe_serialization"
    ], check=True)

def rename_weights(output_dir):
    print("Renaming safetensors files and updating index JSON...")

    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    new_index_file = os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json")
    os.rename(index_file, new_index_file)

    with open(new_index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    new_weight_map = {}
    for k, v in index_data["weight_map"].items():
        new_v = v.replace("model-", "diffusion_pytorch_model-")
        new_weight_map[k] = new_v

    index_data["weight_map"] = new_weight_map

    with open(new_index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    for file in os.listdir(output_dir):
        if file.startswith("model-") and file.endswith(".safetensors"):
            old_path = os.path.join(output_dir, file)
            new_path = os.path.join(output_dir, file.replace("model-", "diffusion_pytorch_model-"))
            os.rename(old_path, new_path)

def prepare_ckpt_structure(output_dir, weights_source_dir, ckpt_output_dir):
    print(f"Copying from {weights_source_dir} to {ckpt_output_dir}...")
    if os.path.exists(ckpt_output_dir):
        shutil.rmtree(ckpt_output_dir)
    shutil.copytree(weights_source_dir, ckpt_output_dir)

    transformer_dir = os.path.join(ckpt_output_dir, "transformer")

    if os.path.exists(transformer_dir):
        print(f"Cleaning transformer directory at {transformer_dir} but keeping config.json...")
        for item in os.listdir(transformer_dir):
            item_path = os.path.join(transformer_dir, item)
            if os.path.basename(item_path) == "config.json":
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        os.makedirs(transformer_dir)

    for file in os.listdir(output_dir):
        src_path = os.path.join(output_dir, file)
        if os.path.isfile(src_path) and file != "config.json":
            shutil.copy(src_path, os.path.join(transformer_dir, file))

    print("Checkpoint structure updated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Input checkpoint folder (e.g., path/checkpoint-12)")
    parser.add_argument("--mid_output_dir", default="", help="Path to store merged FP32 weights")
    parser.add_argument("--weights_source", default=os.path.expanduser("../../pretrained_models/CogVideoX1.5-5B"),
                        help="Path to original CogVideo weights")
    parser.add_argument("--ckpt_output_dir", default="",
                        help="Path to final output ckpt directory")

    args = parser.parse_args()

    if args.mid_output_dir == "":
        mid_output_dir = args.checkpoint_dir + '-fp32'
    else:
        mid_output_dir = args.mid_output_dir

    if args.ckpt_output_dir == "":
        ckpt_output_dir = args.checkpoint_dir.replace('/checkpoint-', '/ckpt-') + '-sft'
    else:
        ckpt_output_dir = args.ckpt_output_dir
    
    if os.path.exists(ckpt_output_dir):
        print(f"[Skipped] {ckpt_output_dir} already exists. Skipping processing.")
        return

    run_zero_to_fp32(args.checkpoint_dir, mid_output_dir)
    rename_weights(mid_output_dir)
    prepare_ckpt_structure(mid_output_dir, args.weights_source, ckpt_output_dir)

    if os.path.exists(mid_output_dir):
        print(f"Removing intermediate directory: {mid_output_dir}")
        shutil.rmtree(mid_output_dir)

    print("All done!")

if __name__ == "__main__":
    main()
