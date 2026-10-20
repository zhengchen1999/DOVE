"""Install the two supplied archives locally without adding weights to Git."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-zip", type=Path, help="The supplied eval_metric.zip")
    parser.add_argument("--ewarp-zip", type=Path, help="The supplied ewarp_v3_code(2).zip")
    args = parser.parse_args()
    if not args.metrics_zip and not args.ewarp_zip:
        parser.error("Specify at least one archive")
    manifest = json.loads((ROOT / "metrics/sources.json").read_text())
    for path, key in [(args.metrics_zip, "eval_metric.zip"), (args.ewarp_zip, "ewarp_v3_code(2).zip")]:
        if path and hashlib.sha256(path.read_bytes()).hexdigest() != manifest["archives_sha256"][key]:
            raise ValueError(f"Archive differs from the supplied reference: {path}")
    if args.metrics_zip:
        destination = ROOT / "third_party"
        with zipfile.ZipFile(args.metrics_zip) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {item.filename}")
                if (len(path.parts) < 3 or path.parts[0] != "eval_metric-master"
                        or path.parts[1] not in {"DOVER", "VBench"} or item.is_dir()):
                    continue
                target = destination / Path(*path.parts)
                data = archive.read(item)
                if target.exists():
                    if target.read_bytes() != data:
                        raise FileExistsError(f"Refusing to replace modified source: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        print(f"Optional metric source: {destination / 'eval_metric-master'}")
    if args.ewarp_zip or args.metrics_zip:
        member = ("ewarp_v3/eval_metric/RAFT/models/raft-things.pth" if args.ewarp_zip
                  else "eval_metric-master/RAFT/models/raft-things.pth")
        with zipfile.ZipFile(args.ewarp_zip or args.metrics_zip) as archive:
            data = archive.read(member)
        if hashlib.sha256(data).hexdigest() != manifest["raft_checkpoint_sha256"]:
            raise ValueError("Unexpected RAFT checkpoint")
        target = ROOT / "pretrained_models/metrics/raft-things.pth"
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace different checkpoint: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"RAFT checkpoint: {target}")


if __name__ == "__main__":
    main()
