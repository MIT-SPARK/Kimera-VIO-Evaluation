#!/usr/bin/env python

import numpy as np

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                      if smallest == element]

def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:,0:3],axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances),1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances
