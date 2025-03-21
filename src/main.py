import argparse
import sys
from train import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description="Flower Classification Project")
    parser.add_argument(
        '--mode', type=str, choices=['train'], default='train',
        help="Mode to run the script: 'train'"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_main()
    else:
        print("Invalid mode selected.")
        sys.exit(1)

if __name__ == "__main__":
    main()