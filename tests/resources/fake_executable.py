#!/usr/bin/env python3
"""Fake script to just save command line arguments to file."""
import sys
import argparse
import pathlib


def main():
    """Save arguments to file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=None)
    args = parser.parse_known_args(sys.argv)[0]

    if not args.output_path:
        sys.exit(1)

    output_path = pathlib.Path(args.output_path).expanduser().absolute()
    arg_path = output_path / "arguments.txt"
    with arg_path.open("w") as fout:
        fout.write("\n".join(sys.argv))


if __name__ == "__main__":
    main()
