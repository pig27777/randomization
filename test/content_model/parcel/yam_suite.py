"""Six prototypes through the existing YAM runner, with unchanged v2.2 actions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experience.yam_platform.angles import execute

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for prototype in range(6):
        execute('parcel.yam_adapter',dict(prototype=prototype,dt=.0005,
                category='compressible' if prototype>=4 else 'bendable'),args.output/f'p{prototype}')
