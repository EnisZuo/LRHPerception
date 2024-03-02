import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("test parser")
    parser.add_argument("--test_arg1", default=0, type=int, help="input a int for test_arg1")
    args = parser.parse_args()
    print(args.test_arg1)