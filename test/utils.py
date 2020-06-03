import csv
import re
import sys
from struct import unpack

import numpy as np


def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
    return img, height, width


def generate_depth(file_path, baseline=176.252, focal_length=4152.073, doffs=213.084):
    depth_img, height, width = read_pfm(file_path)
    depth_img = np.array(depth_img)
    depths = baseline * focal_length / (depth_img + doffs)
    depths = np.reshape(depths, (height, width))
    depths = np.fliplr([depths])[0]
    return depths


def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def create_depth_map(pfm_file_path, calib=None):
    # dispariy, [shape,scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        print(calib)
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))

        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = generate_depth(pfm_file_path, base_line, fx, doffs)

        return depth_map
