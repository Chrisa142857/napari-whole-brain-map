"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations
import torch
import numpy as np
import nibabel as nib

import warnings
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

from multiview_stitcher import spatial_image_utils, io

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer"""

    # implement your writer logic here ...
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    assert isinstance(data, np.ndarray), "Support Numpy or Torch array"
    if not path.endswith('.nii.gz'):
        path = path + '.nii.gz'
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    # return path to any file(s) that were successfully written
    return [path]


def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
    """
    Writes zarr backed dask arrays containing fused images.
    Ignores transform_keys.
    FullLayerData: 3-tuple with (data, meta, layer_type)
    """

    # implement your writer logic here ...
    data = data[0][0]
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    assert isinstance(data, np.ndarray), "Support Numpy or Torch array"
    if not path.endswith('.nii.gz'):
        path = path + '.nii.gz'
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return [path]
