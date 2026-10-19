"""Download the Supervision 'vehicles.mp4' clip used in the manuscript (Sec. 3.B)."""
import argparse
import os

import _path  # noqa: F401
from _path import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    args = parser.parse_args()
    from supervision.assets import VideoAssets, download_assets

    os.makedirs(args.out_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(args.out_dir)
    try:
        download_assets(VideoAssets.VEHICLES)
    finally:
        os.chdir(cwd)
    print(f"saved {os.path.join(args.out_dir, VideoAssets.VEHICLES.value)}")


if __name__ == "__main__":
    main()
