"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import os, json, tempfile
from collections.abc import Iterable
import napari.layers
import torch
import numpy as np
from dask import compute
from napari.utils import notifications
from datetime import datetime
from torch_scatter import scatter_max

from magicgui import widgets
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QLabel, QFileDialog
# from qtpy.QtGui import QPixmap
import xml.etree.ElementTree as ET 
# import xarray as xr
# import dask.array as da
# from napari.layers import Image, Labels
# from napari_stitcher import _reader
# from pathlib import Path

from napari_whole_brain_map import _utils
if TYPE_CHECKING:
    import napari


# define labels for visualization choices
CHOICE_METADATA = 'Original'
CHOICE_REGISTERED = 'Registered'


class BrainmapQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # create temporary directory for storing dask arrays
        self.tmpdir = tempfile.TemporaryDirectory()

        self.all_widgets = []

        self.button_select_list = widgets.Button(text='Select directory')
        self.select_list_label = widgets.HBox(widgets=[
            widgets.Label(
                value='Brain tile list',
                tooltip='Load cell profile to represent whole brain map.',
                name='Select brain'
            ),
            self.button_select_list,
            widgets.Label(
                value='Loading status',
                tooltip='Status of loading cell profile.',
            )
        ])
        self.tile_selection = widgets.Select(choices=[])
        self.select_list_status = widgets.Select(choices=[])
        self.select_list_box = widgets.VBox(widgets=[
            self.select_list_label,
            widgets.HBox(widgets=[
                self.tile_selection,
                self.select_list_status,
            ])
        ])
        self.all_widgets.append(self.select_list_box)

        self.button_load_cprof = widgets.Button(text='Load cell profile',
            tooltip='Load cell profile of selected tiles to RAM.')

        self.all_widgets.append(self.button_load_cprof)

        self.org_res_x_textbox = widgets.FloatText(value=0.75, name='Orgin res (x)')
        self.org_res_y_textbox = widgets.FloatText(value=0.75, name='Orgin res (y)')
        self.org_res_z_textbox = widgets.FloatText(value=4.0, name='Orgin res (z)')
        self.map_res_textbox = widgets.FloatText(value=25, name='Map res (um/vx)')
        self.device = widgets.Text(value='cuda:0', name='Device')
        self.maptype_dropdown = widgets.Dropdown(choices=['cell count', 'avg volume'], name='Map type', allow_multiple=False)
        
        self.map_config_widgets_basic = [widgets.HBox(widgets=[
            self.org_res_x_textbox,
            self.org_res_y_textbox,
            self.org_res_z_textbox,]),widgets.HBox(widgets=[
            self.map_res_textbox,
            self.maptype_dropdown,
            self.device])
        ]
        
        # self.all_widgets.extend(self.map_config_widgets_basic)

        
        # self.map_config_widgets_adv = [
        # ]
        
        self.map_config_widgets_tabs = QTabWidget() 
        self.map_config_widgets_tabs.resize(300, 200) 
        self.map_config_widgets_tabs.addTab(
            widgets.VBox(widgets=self.map_config_widgets_basic).native, "Basic") 
        # self.map_config_widgets_tabs.addTab(
        #     widgets.VBox(widgets=self.map_config_widgets_adv).native, "Advanced") 
        
        self.all_widgets.append(self.map_config_widgets_tabs)


        self.button_gen_brainmap = widgets.Button(text='Generate brainmap',
            tooltip='Generate brainmap based on loaded cell profiles.')

        self.all_widgets.append(self.button_gen_brainmap)

        # self.overlap = widgets.FloatSlider(
        #     value=0.4, min=0, max=0.9999, label='Overlap:')
        self.ncol = widgets.SpinBox(
            value=4, min=1, max=100, label='# of columns:') 
        self.nrow = widgets.SpinBox(
            value=5, min=1, max=100, label='# of rows:') 
        self.stitchtype_dropdown = widgets.Dropdown(choices=['N/A', 'Coarse', 'Refine', 'Manual'], name='Stitch type', allow_multiple=False)
        self.zstitchtype = widgets.Dropdown(choices=['Avg', 'Org'], name='Z-type', allow_multiple=False)
        self.button_arrange_tiles = widgets.Button(text='Arrange tiles')
        # organize widgets        
        self.mosaic_widgets = widgets.VBox(widgets=[
            widgets.HBox(widgets=[
                # self.overlap,
            self.ncol, 
            self.nrow,]), widgets.HBox(widgets=[
            self.stitchtype_dropdown,
            self.zstitchtype,
            self.button_arrange_tiles,])
        ])
        self.all_widgets.append(self.mosaic_widgets)
        self.manual_stitch_path = widgets.Button(text='N/A')
        self.manual_stitch_selection = widgets.HBox(widgets=[
            widgets.Label(value='Select path to manual stitch'),
            self.manual_stitch_path
        ])
        self.all_widgets.append(self.manual_stitch_selection)

        self.button_fuse = widgets.Button(text='Undouble and Fuse',
            tooltip='Fuse brain map as whole.')
        self.undouble_path = widgets.Button(text=f'{os.getcwd()}/tmp/doubled_NIS_label')
        self.undouble_to_fuse = widgets.HBox(widgets=[
            widgets.Label(value='Select path to doubled cell id'),
            self.undouble_path,
            self.button_fuse
        ])
        self.all_widgets.append(self.button_fuse)

        self.container = QWidget()
        self.container.setLayout(QVBoxLayout())

        for w in self.all_widgets:
            if hasattr(w, 'native'):
                self.container.layout().addWidget(w.native)
            else:
                self.container.layout().addWidget(w)

        self.container.setMinimumWidth = 5
        self.layout().addWidget(self.container)

        self.undouble_path.clicked.connect(self.select_undouble_path)
        self.manual_stitch_path.clicked.connect(self.select_manual_path)
        self.button_load_cprof.clicked.connect(self.load_cell_profile_raw)
        self.button_select_list.clicked.connect(self.load_cell_profile_name)
        self.button_gen_brainmap.clicked.connect(self.whole_brainmap)
        self.button_arrange_tiles.clicked.connect(self.arrange_brainmap_layer)
        self.button_fuse.clicked.connect(self.run_fusion)
        self.raw = {}
        self.raw_stitched = {}
        self.seg_shape = {}
        # self.msims = {}
        self.image_layer_bbox = {}
        self.brainmap_layers = []
        self.brainmap_layernames = []
        self.fused_layers = {}
        # self.device.value = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.brainmap_overlap_mask = {}
        self.brainmap_cache = {}
        self.doubled_label = {}
        self.cache_tform = {}

    def arrange_brainmap_layer(self, stitch_type=None, ncol=None, nrow=None):
        if stitch_type is None:
            stitch_type = self.stitchtype_dropdown.current_choice
        if ncol is None:
            ncol = self.ncol.value
        if nrow is None:
            nrow = self.nrow.value

        stitch_path = '/'.join(self.cprof_root.split('/')[:-1]) + '/' + self.btag
        stitch_path = stitch_path.replace('results', 'stitch_by_ptreg').replace('P4/', '')
        overlap_r = 0.4#self.overlap.value
        org_overlap_r = 0.2
        ij_list = [[i, j] for i in range(ncol) for j in range(nrow)]
        # zdepth = max([self.seg_shape[d][0] for d in self.seg_shape])
        # ratio = [s/self.map_res_textbox.value for s in [self.org_res_z_textbox.value, self.org_res_y_textbox.value, self.org_res_x_textbox.value]]
        # trans_slice_bbox = {d: [None for _ in range(zdepth)] for d in self.seg_shape}
        self.cache_tform = {}
        # for di, d in enumerate(self.seg_shape):
        for i, j in ij_list:
            d = f'UltraII[{i:02d} x {j:02d}]'
            if d not in self.seg_shape: continue
            seg_shape = self.seg_shape[d]
            tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
            # i, j = ij_list[di]
            k = f'{i}-{j}'
            tile_lt_x, tile_lt_y = i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)
            org_tile_lt_x, org_tile_lt_y = i*seg_shape[1]*(1-org_overlap_r), j*seg_shape[2]*(1-org_overlap_r)
            if stitch_type == 'N/A':
                if 'N/A' not in self.raw_stitched: self.raw_stitched['N/A'] = {}
                if d not in self.raw_stitched['N/A']: self.raw_stitched['N/A'][d] = self.raw[d]
                tz = 0
                self.cache_tform[d] = [org_tile_lt_x, org_tile_lt_y]
                _, _, _, nis, _ = self.raw[d]
                new_nis = nis.clone()
                new_nis[:, 1::3] = new_nis[:, 1::3] + org_tile_lt_x
                new_nis[:, 2::3] = new_nis[:, 2::3] + org_tile_lt_y
                self.raw_stitched['N/A'][d] = [self.raw_stitched['N/A'][d][0], self.raw_stitched['N/A'][d][1], self.raw_stitched['N/A'][d][2], new_nis, self.raw_stitched['N/A'][d][-1]]
                    
            if stitch_type in ['Coarse', 'Refine']:
                tform_stack_coarse = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_coarse.json', 'r', encoding='utf-8'))
                tz, tx, ty = tform_stack_coarse[k]
                tile_lt_x = tile_lt_x + tx
                tile_lt_y = tile_lt_y + ty
                org_tile_lt_x = org_tile_lt_x + tx
                org_tile_lt_y = org_tile_lt_y + ty
                if 'Coarse' not in self.raw_stitched: self.raw_stitched['Coarse'] = {}
                if d not in self.raw_stitched['Coarse']: self.raw_stitched['Coarse'][d] = self.raw[d]
                self.cache_tform[d] = [org_tile_lt_x, org_tile_lt_y]
                if stitch_type == 'Refine':
                    if 'Refine' not in self.raw_stitched: self.raw_stitched['Refine'] = {}
                    if d in self.raw_stitched['Refine']: continue
                    if os.path.exists(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine.json'):
                        tform_stack_refine = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine.json', 'r', encoding='utf-8'))
                    if os.path.exists(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine_ptreg.json'):
                        tform_stack_ptreg = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
                        print(datetime.now(), "Loaded tform of coarse, refine and pt-reg", len(tform_stack_ptreg), [tform_stack_ptreg[zi].keys() for zi in range(len(tform_stack_ptreg)) if len(tform_stack_ptreg[zi].keys()) > 0][-1])
                    else:
                        tform_stack_ptreg = None
                        print(datetime.now(), "Loaded tform of coarse, refine")
                    # center = self.raw_stitched['Coarse'][d][0].clone()
                    center, _, _, nis, nis_label = self.raw[d]
                    ct_z = (center[:, 0].clone() + tz).long()
                    nis_z = (((nis[:, 0] + nis[:, 3]) / 2).clone() + tz).long()
                    new_nis = []
                    new_nis_label = []
                    pre_refine_lt = None
                    for zi in range(len(tform_stack_refine)):
                        ct_zmask = torch.where(ct_z == zi)[0]
                        if k in tform_stack_refine[zi]:
                            _tx, _ty = tform_stack_refine[zi][k]
                            if (abs(_tx) > tform_xy_max[0] or abs(_ty) > tform_xy_max[1]):
                                if pre_refine_lt is not None:
                                    _tx, _ty = pre_refine_lt
                                else:
                                    for zii in range(zi, len(tform_stack_refine)):
                                        if k not in tform_stack_refine[zii]: continue
                                        _tx, _ty = tform_stack_refine[zii][k]
                                        if abs(_tx) <= tform_xy_max[0] and abs(_ty) <= tform_xy_max[1]:
                                            break
                            else:
                                _tx, _ty = tform_stack_refine[zi][k]
                            
                            if tform_stack_ptreg is not None:
                                if k in tform_stack_ptreg[zi]:
                                    tx = _tx + tform_stack_ptreg[zi][k][0]
                                    ty = _ty + tform_stack_ptreg[zi][k][1]
                            else:
                                tx, ty = _tx, _ty
                        else:
                            _tx, _ty = 0, 0
                            tx, ty = 0, 0

                        self.cache_tform[d] = [min(org_tile_lt_x + tx, self.cache_tform[d][0]), min(org_tile_lt_y + ty, self.cache_tform[d][1])]

                        if len(ct_zmask) != 0: 
                            pre_refine_lt = [_tx, _ty]
                            center[ct_zmask, 1] = center[ct_zmask, 1] + tx
                            center[ct_zmask, 2] = center[ct_zmask, 2] + ty

                        nis_zmask = torch.where(nis_z == zi)[0]
                        if len(nis_zmask) > 0:
                            # print("before trans", nis[nis_zmask].min(0)[0], nis[nis_zmask].max(0)[0])
                            nis[nis_zmask, 1::3] = nis[nis_zmask, 1::3] + org_tile_lt_x + tx
                            nis[nis_zmask, 2::3] = nis[nis_zmask, 2::3] + org_tile_lt_y + ty
                            # print("after trans", nis[nis_zmask].min(0)[0], nis[nis_zmask].max(0)[0], [minz, miny, minx])
                            new_nis.append(nis[nis_zmask])
                            new_nis_label.append(nis_label[nis_zmask])
                    
                    new_nis = torch.cat(new_nis)
                    new_nis_label = torch.cat(new_nis_label)
                    self.raw_stitched['Refine'][d] = [center, self.raw_stitched['Coarse'][d][1], self.raw_stitched['Coarse'][d][2], new_nis, new_nis_label]

            if stitch_type == 'Manual':
                if 'Manual' not in self.raw_stitched: self.raw_stitched['Manual'] = {}
                if 'N/A' not in self.raw_stitched: self.raw_stitched['N/A'] = {}
                if d not in self.raw_stitched['N/A']: self.raw_stitched['N/A'][d] = self.raw[d]
                xshape = ((self.seg_shape[d][1]*(1-org_overlap_r)) * ncol + org_overlap_r*self.seg_shape[d][1])*self.org_res_y_textbox.value
                yshape = ((self.seg_shape[d][2]*(1-org_overlap_r)) * nrow + org_overlap_r*self.seg_shape[d][2])*self.org_res_x_textbox.value
                if d not in self.raw_stitched['Manual']:
                    center, _, _, nis, nis_label = self.raw[d]
                    ct_z = center[:, 0].clone().long()
                    nis_z = (nis[:, 0] + nis[:, 3]) / 2
                    new_nis = []
                    new_nis_label = []
                
                tz = []
                xmlfile_root = self.manual_stitch_path.text
                # xmlfile_root = f'/cajal/Felix/Lightsheet/stitching/Manual_aligned_{self.btag}'
                # if self.btag == 'L57D855P5':
                #     xmlfile_root = '/cajal/ACMUSERS/ziquanw/Lightsheet/imaris_stitch/Manual_aligned_L57D855P5'
                for fn in os.listdir(xmlfile_root):
                    if not fn.endswith('.xml'): continue
                    xmlfile = f'{xmlfile_root}/{fn}'
                    tree = ET.parse(xmlfile) 
                    for i in range(len(tree.getroot()[0])):
                        if tree.getroot()[0][i].tag == 'Image':
                            item = tree.getroot()[0][i].attrib
                            zsplit, ims_fn = item['Filename'].split('\\')[-2:]
                            zmin = int(zsplit.split('-')[0][1:])
                            zmax = int(zsplit.split('-')[1][1:])
                            if d.lower() in ims_fn.lower():
                                minz, miny, minx = get_minxyz(tree, item, xshape, yshape)
                                tz.append(minz)
                                break
                print("list of zmin", tz)
                tz, tile_lt_x, tile_lt_y = np.mean(tz).item(), None, None
                for fn in os.listdir(xmlfile_root):
                    if not fn.endswith('.xml'): continue
                    xmlfile = f'{xmlfile_root}/{fn}'
                    tree = ET.parse(xmlfile) 
                    for i in range(len(tree.getroot()[0])):
                        if tree.getroot()[0][i].tag == 'Image':
                            item = tree.getroot()[0][i].attrib
                            zsplit, ims_fn = item['Filename'].split('\\')[-2:]
                            zmin = int(zsplit.split('-')[0][1:])
                            zmax = int(zsplit.split('-')[1][1:])
                            if d.lower() in ims_fn.lower():
                                minz, miny, minx = get_minxyz(tree, item, xshape, yshape)
                                print(d, zsplit, ims_fn, [minz, miny, minx], )
                                if d not in self.cache_tform:
                                    self.cache_tform[d] = [miny/self.org_res_y_textbox.value, minx/self.org_res_x_textbox.value]
                                else:
                                    self.cache_tform[d] = [min(miny/self.org_res_y_textbox.value, self.cache_tform[d][0]), min(minx/self.org_res_x_textbox.value, self.cache_tform[d][1])]
                                if tile_lt_x is None:
                                    if self.zstitchtype.current_choice == 'Org':
                                        tz, tile_lt_x, tile_lt_y = minz, miny, minx
                                    elif self.zstitchtype.current_choice == 'Avg':
                                        tile_lt_x, tile_lt_y = miny, minx
                                elif d not in self.raw_stitched['Manual']:
                                    ct_zmask = torch.where(torch.logical_and(ct_z >= zmin, ct_z < zmax))[0]
                                    if len(ct_zmask) > 0:
                                        if self.zstitchtype.current_choice == 'Org':
                                            center[ct_zmask, 0] = center[ct_zmask, 0] + (minz-tz)
                                        center[ct_zmask, 1] = center[ct_zmask, 1] + (miny-tile_lt_x)
                                        center[ct_zmask, 2] = center[ct_zmask, 2] + (minx-tile_lt_y)
                                if d not in self.raw_stitched['Manual']:
                                    nis_zmask = torch.where(torch.logical_and(nis_z >= zmin, nis_z < zmax))[0]
                                    if len(nis_zmask) > 0:
                                        # print("before trans", nis[nis_zmask].min(0)[0], nis[nis_zmask].max(0)[0])
                                        if self.zstitchtype.current_choice == 'Org':
                                            nis[nis_zmask, 0::3] = nis[nis_zmask, 0::3] + minz/self.org_res_z_textbox.value
                                        elif self.zstitchtype.current_choice == 'Avg':
                                            nis[nis_zmask, 0::3] = nis[nis_zmask, 0::3] + tz/self.org_res_z_textbox.value
                                        nis[nis_zmask, 1::3] = nis[nis_zmask, 1::3] + miny/self.org_res_y_textbox.value
                                        nis[nis_zmask, 2::3] = nis[nis_zmask, 2::3] + minx/self.org_res_x_textbox.value
                                        # print("after trans", nis[nis_zmask].min(0)[0], nis[nis_zmask].max(0)[0], [minz, miny, minx])
                                        new_nis.append(nis[nis_zmask])
                                        new_nis_label.append(nis_label[nis_zmask])

                                # for z in range(int(zmin+minz), np.ceil(zmax+minz).astype(np.int32)):
                                    # if z >= 0 and z < len(trans_slice_bbox[d]):
                                    #     trans_slice_bbox[d][z] = [miny, minx, miny+seg_shape[1]*self.org_res_y_textbox.value, minx+seg_shape[2]*self.org_res_x_textbox.value]
                                break
            
                if d not in self.raw_stitched['Manual']: 
                    new_nis = torch.cat(new_nis)
                    new_nis_label = torch.cat(new_nis_label)
                    self.raw_stitched['Manual'][d] = [center, self.raw_stitched['N/A'][d][1], self.raw_stitched['N/A'][d][2], new_nis, new_nis_label]
            
            if stitch_type in ['N/A', 'Refine', 'Manual']:
                self.raw[d] = self.raw_stitched[stitch_type][d]
                self.update_brainmap_layer(d)
                
            self.brainmap_layers[self.brainmap_layernames.index(d)].translate[-3:] = [tz, tile_lt_x, tile_lt_y]
            self.brainmap_layers[self.brainmap_layernames.index(d)].refresh()

    def update_select_status(self):
        status = []
        for update_d in self.tile_selection.choices:
            if update_d in self.raw:
                status.append(f'cell #: {format_number(len(self.raw[update_d][0]))} ({update_d})')
            else:
                status.append(f'Not loaded ({update_d})')
        self.select_list_status.choices = status
        print(len(self.tile_list), len(self.tile_selection.choices), len(self.select_list_status.choices), len(status))

    def select_undouble_path(self):
        self.undouble_path.text = str(QFileDialog.getExistingDirectory(self, "Select folder contains [Doubled-Cell-ID].zip"))

    def select_manual_path(self):
        self.manual_stitch_path.text = str(QFileDialog.getExistingDirectory(self, "Select folder contains [Imaris-Stitcher].xml"))

    def load_cell_profile_name(self):
        self.btag_split_key = '_'
        self.btag_split_i = 1
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.cprof_root = folder
        btag = folder.split('/')[-1].split(self.btag_split_key)[self.btag_split_i]
        self.btag = btag
        cprof_tile_name = [d for d in os.listdir(folder) if os.path.exists(f'{folder}/{d}/{btag}_NIScpp_results_zmin0_instance_center.zip')] # and os.path.isdir(f'{folder}/{d}')
        self.tile_list = sorted(cprof_tile_name)
        self.tile_selection.choices = self.tile_list
        self.update_select_status()
        self.button_select_list.text = f'Selected {self.btag}'
        self.button_select_list.enabled = False
        self.manual_stitch_path.text = f'/cajal/Felix/Lightsheet/stitching/Manual_aligned_{self.btag}'


    def load_cell_profile_raw(self):
        zratio = 2.5/4
        self.zstitch_remap = {}
        with _utils.TqdmCallback(tqdm_class=_utils.progress,
                        desc='Loading raw cell profile', bar_format=" "):
            for choicei in range(len(self.tile_selection.current_choice)):

                d = self.tile_selection.current_choice[choicei]
                cprof_d = f'{self.cprof_root}/{d}'
                stack_names = [f for f in os.listdir(cprof_d) if f.endswith('instance_center.zip')]
                _bbox = []
                _label = []
                _vol = []
                zdepth = 0
                for stack_name in sort_fnlist(stack_names):
                    seg_meta = torch.load(f"{cprof_d}/{stack_name.replace('instance_center', 'seg_meta')}")
                    zdepth += seg_meta[0]
                    zstart = int(stack_name.split('zmin')[1].split('_')[0])
                    zstart = int(zstart*zratio)
                    volfn = f"{cprof_d}/{stack_name.replace('instance_center', 'instance_volume')}"
                    labelfn = f"{cprof_d}/{stack_name.replace('instance_center', 'instance_label')}"
                    vol = torch.load(volfn).long()
                    label = torch.load(labelfn).long()
                    bboxfn = f"{cprof_d}/{stack_name.replace('instance_center', 'instance_bbox')}"
                    bbox = torch.load(bboxfn)
                    bbox[:, 0] = bbox[:, 0] * zratio + zstart
                    bbox[:, 3] = bbox[:, 3] * zratio + zstart
                    _vol.append(vol)
                    _bbox.append(bbox)
                    _label.append(label)
                    
                if len(_bbox) == 0: continue
                self.seg_shape[d] = [int(zdepth * zratio), seg_meta[1].item(), seg_meta[2].item()]
                vol = torch.cat(_vol).to(self.device.value)
                bbox = torch.cat(_bbox).to(self.device.value)
                label = torch.cat(_label).to(self.device.value)
                pt = (bbox[:, :3] + bbox[:, 3:]) / 2
                zstitch_remap_fn = f"{cprof_d}/{self.btag}_remap.zip"
                if os.path.exists(zstitch_remap_fn):
                    zstitch_remap = torch.load(zstitch_remap_fn).to(self.device.value)
                    self.zstitch_remap[d] = zstitch_remap
                    print("before remove z-stitched pt", pt.shape, label.shape)
                    pt, pt_label, vol = do_zstitch(zstitch_remap, pt, label, vol=vol)
                    print("after remove z-stitched pt", pt.shape, pt_label.shape)
                else:
                    self.zstitch_remap[d] =  None
                    pt_label = label
                    
                self.raw[d] = [pt, pt_label, vol, bbox.float(), label]
                self.select_list_status.choices = [f'cell #: {format_number(len(self.raw[update_d][0]))}' if update_d in self.raw else f'Not loaded' for update_d in self.tile_list]
                self.update_select_status()


    def whole_brainmap(self):
        for d in self.raw:
            self.update_brainmap_layer(d)

    def update_brainmap_layer(self, name):
        colormap = 'gray'
        blending = 'additive'
        # spatial_dims = [None, None, None, None]
        dim_n = 3
        affine_transform = np.eye(dim_n+1)
        center, _, vol = self.raw[name][:3]
        dshape = self.seg_shape[name]
        brainmap = self.get_brainmap(center, vol, dshape)
        brainmap = brainmap.numpy().astype(np.float32)
        # name = f'{name}:{self.maptype_dropdown.current_choice.replace(' ', '-')}'
        kwargs = {
            'contrast_limits': compute_contrast_limit(brainmap),
            'name': name,
            'colormap': colormap,
            'gamma': 0.9,
            'affine': affine_transform,
            'translate': np.array([0 for dim in range(dim_n)]),
            'scale': np.array([self.map_res_textbox.value for dim in range(dim_n)]),
            'cache': True,
            'blending': blending,
            'multiscale': True,
            'metadata': None,
            'axis_labels': ['x', 'y', 'z']
        }
        print('brainmap.shape', brainmap.shape)
        self.brainmap_cache[name] = brainmap
        multiscale_data = [brainmap, brainmap[::2, ::2, ::2], brainmap[::4, ::4, ::4]]
        # multiscale_data = [xr.DataArray(brainmap, dims=['x', 'y', 'z']), xr.DataArray(brainmap[::2, ::2, ::2], dims=['x', 'y', 'z']), xr.DataArray(brainmap[::4, ::4, ::4], dims=['x', 'y', 'z'])]
        # multiscale_data = brainmap
        if name in self.brainmap_layernames:
            self.brainmap_layers[self.brainmap_layernames.index(name)].data = multiscale_data
        else:
            brainmap_layer = self.viewer.add_image(multiscale_data, **kwargs)
            brainmap_layer.events.connect(self.watch_layer_changes)
            self.brainmap_layers.append(brainmap_layer)
            self.brainmap_layernames.append(name)
            # self.msims[name] = image_layer_to_msim(brainmap_layer, self.viewer)
        self.image_layer_bbox[name] = self.get_image_layer_bbox(self.brainmap_layers[self.brainmap_layernames.index(name)])

    def get_brainmap(self, center, vol=None, dshape=None):
        ratio = [s/self.map_res_textbox.value for s in [self.org_res_z_textbox.value, self.org_res_y_textbox.value, self.org_res_x_textbox.value]]
        center = center.clone().to(self.device.value)
        outlier_mask = (center<0).any(-1)
        # print(center.shape, torch.where(outlier_mask)[0].shape)
        # center = center[outlier_mask]
        if vol is not None:
            vol = vol.clone().float().to(self.device.value)
            # vol = vol[outlier_mask]
        center[:,0] = center[:,0] * ratio[0]
        center[:,1] = center[:,1] * ratio[1]
        center[:,2] = center[:,2] * ratio[2]
        dshape = [max(int(dshape[0]*ratio[0]), int(center[:,0].max()+2)), max(int(dshape[1]*ratio[1]), int(center[:,1].max()+2)), max(int(dshape[2]*ratio[2]), int(center[:,2].max()+2))]
        # dshape = [int(center[:,0].max()+1), int(center[:,1].max()+1), int(center[:,2].max()+1)]
        # outbound_mask = torch.logical_or(center[:, 0].round() > dshape[0]-1, center[:, 1].round() > dshape[1]-1)
        # outbound_mask = torch.logical_or(outbound_mask, center[:, 2].round() > dshape[2]-1)
        # center = center[torch.logical_not(outbound_mask)]
        z = center[:, 0]#.clip(min=0)
        y = center[:, 1]#.clip(min=0)
        x = center[:, 2]#.clip(min=0)
        # print(center.shape, z, x, y)
        loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(self.device.value) 
        loc = loc[(z.round().long(), y.round().long(), x.round().long())] # all nis location in the downsample space
        loc_count = loc.bincount() 
        loc_count = loc_count[loc_count!=0] 
        atlas_loc = loc.unique().to(self.device.value) # unique location in the downsample space
        ## volume avg & local intensity
        vol_avg = None
        if self.maptype_dropdown.current_choice == 'avg volume':
            loc_argsort = loc.argsort().cpu()
            loc_splits = loc_count.cumsum(0).cpu()
            loc_vol = torch.tensor_split(vol[loc_argsort], loc_splits)
            assert len(loc_vol[-1]) == 0
            loc_vol = loc_vol[:-1]
            loc_vol = torch.nn.utils.rnn.pad_sequence(loc_vol, batch_first=True, padding_value=-1)
            loc_fg = loc_vol!=-1
            loc_num = loc_fg.sum(1)
            loc_vol[loc_vol==-1] = 0
            vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2]).float()#.to(self.device.value)
            vol_avg[atlas_loc] = (loc_vol.sum(1) / loc_num).cpu().float()
            # for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
            #     where_loc = torch.where(loc==loci)[0]
            #     vol_avg[loci] = vol[where_loc].mean()
            vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2])#.cpu()
            return vol_avg
        ## density map
        elif self.maptype_dropdown.current_choice == 'cell count':
            density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(self.device.value)
            density[atlas_loc] = loc_count.double() #/ center.shape[0]
            density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
            return density
        
    def watch_layer_changes(self, event):
        """
        Watch changes in transformations.
        """
        # if event.type in ['affine', 'scale', 'translate']:
        l = event.source
        self.image_layer_bbox[l.name] = self.get_image_layer_bbox(self.brainmap_layers[self.brainmap_layernames.index(l.name)])
                
    def run_fusion(self):
        if self.stitchtype_dropdown.current_choice in ['Coarse']: 
            print(f'Not support [{self.stitchtype_dropdown.current_choice}] stitch type')
            return 
        stack_nis = {k: self.raw[k][-2].detach().cpu() for k in self.raw}
        stack_label = {k: self.raw[k][-1].detach().cpu() for k in self.raw}
        tile_center = {k: [self.seg_shape[k][1]//2, self.seg_shape[k][2]//2] for k in self.image_layer_bbox}
        self.doubled_label = nms_undouble_cell(stack_nis, stack_label, tile_center, self.seg_shape, tile_wh=self.seg_shape, tile_lt_loc=self.cache_tform, overlap_r=0.2, btag=self.btag, device=self.device.value, save_path='./tmp')
        lrange = 1000
        undoubled_center = []
        undoubled_label = []
        undoubled_vol = []
        for k in self.doubled_label:
        #########################
            # pt, label, vol = self.raw[k][:3]
        #########################
            pt = stack_nis[k].clone().to(self.device.value)
            pt = (pt[:, :3] + pt[:, 3:]) / 2
            label = stack_label[k].clone().to(self.device.value)
        #########################
            print("before remove doubled pt", self.raw[k][-2].shape, label.shape)
            keep_ind = []
            for labeli in range(0, len(label), lrange):
                label_batch = label[labeli:labeli+lrange]
                label2rm = label_batch[:, None] == self.doubled_label[k][None, :].to(self.device.value)
                do_rm = label2rm.any(1)
                keep_ind.append(torch.arange(labeli, labeli+len(label_batch), device=self.device.value)[torch.logical_not(do_rm)])
            if len(label) > 0:
                keep_ind = torch.cat(keep_ind)
                pt = pt[keep_ind]
        #########################
                # vol = vol[keep_ind]
        #########################
                label = label[keep_ind]
        #########################
                # self.raw[k] = [pt, label, vol, self.raw[k][-2], self.raw[k][-1]]
                # self.update_brainmap_layer(k)
        #########################
            zstitch_remap = self.zstitch_remap[k]
            if zstitch_remap is not None:
                pt, label, _ = do_zstitch(zstitch_remap, pt, label)

            undoubled_center.append(pt)
            undoubled_label.append(label)
            # undoubled_vol.append(vol)
            print("after remove doubled pt", pt.shape, label.shape)

        undoubled_center = torch.cat(undoubled_center)
        undoubled_label = torch.cat(undoubled_label)
        # undoubled_vol = torch.cat(undoubled_vol)
        undoubled_vol = None

        bbox = list(self.image_layer_bbox.values())
        bbox = np.stack(bbox)
        bbox_origin_zero = np.round(bbox.copy()).astype(np.int32)
        bbox_origin_zero = bbox_origin_zero - bbox_origin_zero.min()
        dshape = bbox_origin_zero[:, 3:].max(0) + 2
        fused_image = self.get_brainmap(undoubled_center, undoubled_vol, dshape).detach().cpu().numpy()

        ################
        fused_name = f"fused:{self.maptype_dropdown.current_choice.replace(' ', '-')}"
        fused_images = [fused_image, fused_image[::2, ::2, ::2], fused_image[::4, ::4, ::4]]
        if fused_name not in self.fused_layers:
            colormap = 'gray'
            blending = 'additive'
            dim_n = 3
            affine_transform = np.eye(dim_n+1)
            kwargs = {
                'contrast_limits': compute_contrast_limit(fused_image),
                'name': fused_name,
                'colormap': colormap,
                'gamma': 0.9,
                'affine': affine_transform,
                'translate': np.array([0 for dim in range(dim_n)]),
                'scale': np.array([self.map_res_textbox.value for dim in range(dim_n)]),
                'cache': True,
                'blending': blending,
                'multiscale': True,
                'metadata': None,
                'axis_labels': ['x', 'y', 'z']
            }
            fused_layer = self.viewer.add_image(fused_images, **kwargs)
            self.fused_layers[fused_name] = fused_layer
        else:
            self.fused_layers[fused_name].data = fused_images

    def get_image_layer_bbox(self, layer: napari.layers.Image):
        D, H, W = layer.data[0].shape
        z, y, x = layer.translate
        ratio = [s/self.map_res_textbox.value for s in [self.org_res_z_textbox.value, 1, 1]]
        z, y, x = z*ratio[0], y*ratio[1], x*ratio[2]
        # print(D, H, W, z, y, x, ratio)
        return np.array([z, y, x, z+D, y+H, x+W])

def do_zstitch(zstitch_remap, pt, label, vol=None):
    lrange = 1000
    ## loc: gnn stitch source (current tile) nis index, stitch_remap_loc: index of pairs in the stitch remap list
    loc, stitch_remap_loc = [], []
    for lrangei in range(0, len(label), lrange):
        lo, stitch_remap_lo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
        loc.append(lo+lrangei)
        stitch_remap_loc.append(stitch_remap_lo)
    loc, stitch_remap_loc = torch.cat(loc), torch.cat(stitch_remap_loc)

    ## pre_loc: gnn stitch target (previous tile) nis index, tloc: index of remaining Z stitch pairs after nis being removed by X-Y stitching
    pre_loc, tloc = [], []
    for lrangei in range(0, len(label), lrange):
        pre_lo, tlo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[1, None, stitch_remap_loc])
        pre_loc.append(pre_lo+lrangei)
        tloc.append(tlo)
    pre_loc, tloc = torch.cat(pre_loc), torch.cat(tloc)

    ## source nis is removed from keeping mask
    keep_mask = torch.ones(len(pt)).bool()
    keep_mask[loc] = False
#     keep_masks[stack_name][f'{i}-{j}'] = torch.logical_and(keep_masks[stack_name][f'{i}-{j}'], keep_mask)

    # merge stitched source nis to target nis
    loc = loc[tloc]
    pt[pre_loc] = (pt[loc] + pt[pre_loc]) / 2
    if vol is not None:
        vol[pre_loc] = vol[loc] + vol[pre_loc]

    pt = pt[keep_mask]
    if vol is not None:
        vol = vol[keep_mask]
    label = label[keep_mask]
    return pt, label, vol

def nms_undouble_cell(stack_nis_bbox, stack_nis_label, tile_center, seg_shape, overlap_r, btag, save_path, tile_wh, tile_lt_loc, device='cuda:1'):
    '''
    NMS to remove doubled cells
    '''
    if os.path.exists(f'{save_path}/doubled_NIS_label/{btag}_doubled_label_byNapari.zip'):
        return torch.load(f'{save_path}/doubled_NIS_label/{btag}_doubled_label_byNapari.zip')
    neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    rm_label = {}
    # nms_margin = 0.3
    nms_r = 0.5 #overlap_r + nms_margin
    nms_computed = []
    ncol = 4
    nrow = 5
    bbox_area_wh = {k: stack_nis_bbox[k][:, 4:6].max(0)[0] - stack_nis_bbox[k][:, 1:3].min(0)[0] + 1 for k in stack_nis_bbox}
    max_tile_wh = {k: [max(tile_wh[k][1], bbox_area_wh[k][0]), max(tile_wh[k][2], bbox_area_wh[k][1])] for k in stack_nis_bbox}
    tile_lt_loc = {k: [min(tile_lt_loc[k][0], stack_nis_bbox[k][:, 1].min(0)[0]), min(tile_lt_loc[k][1], stack_nis_bbox[k][:, 2].min(0)[0])] for k in stack_nis_bbox}
    print(max_tile_wh)
    print(tile_lt_loc)
    for k in stack_nis_bbox:
    # for k in ['1-1']:
        torch.cuda.empty_cache()
        i, j = k.split(' x ')
        i, j = int(i[8:10]), int(j[:-1])
        if k not in rm_label: rm_label[k] = []
        bbox_tile = stack_nis_bbox[k].clone().to(device)
        label_tile = stack_nis_label[k].clone().to(device)
        # lt_loc_tile = tile_lt_loc[k]
        for pi, pj in neighbor:
            if f'UltraII[{i+pi:02d} x {j+pj:02d}]' not in stack_nis_bbox: continue
            if f'{i}-{j}-{i+pi}-{j+pj}' in nms_computed: continue
            if f'{i+pi}-{j+pj}-{i}-{j}' in nms_computed: continue
            bbox_nei = stack_nis_bbox[f'UltraII[{i+pi:02d} x {j+pj:02d}]'].clone().to(device)
            label_nei = stack_nis_label[f'UltraII[{i+pi:02d} x {j+pj:02d}]'].clone().to(device)
            nms_computed.append(f'{i}-{j}-{i+pi}-{j+pj}')
            ###################################
            mov_to_tgt = {}
            if pi < 0: # left of mov to right of tgt
                mov_to_tgt[0] = 1
            if pi > 0: # right of mov to left of tgt
                mov_to_tgt[1] = 0
            if pj < 0: # bottom of mov to top of tgt
                mov_to_tgt[2] = 3
            if pj > 0: # top of mov to bottom of tgt
                mov_to_tgt[3] = 2
            # mov_masks = bbox_in_stitching_seam(bbox_tile, lt_loc_tile, [seg_shape[k][1], seg_shape[k][2]], i, j, ncol, nrow, nms_r)
            # lt_loc_nei = tile_lt_loc[f'UltraII[{i+pi:02d} x {j+pj:02d}]']
            # tgt_masks = bbox_in_stitching_seam(bbox_nei, lt_loc_nei, [seg_shape[f'UltraII[{i+pi:02d} x {j+pj:02d}]'][1], seg_shape[f'UltraII[{i+pi:02d} x {j+pj:02d}]'][2]], i+pi, j+pj, ncol, nrow, nms_r)
            mov_masks = bbox_in_stitching_seam(bbox_tile, tile_lt_loc[k], max_tile_wh[k], i, j, ncol, nrow, nms_r)
            tgt_masks = bbox_in_stitching_seam(bbox_nei, tile_lt_loc[f'UltraII[{i+pi:02d} x {j+pj:02d}]'], max_tile_wh[f'UltraII[{i+pi:02d} x {j+pj:02d}]'], i+pi, j+pj, ncol, nrow, nms_r)
            tgt_mask = []
            mov_mask = []
            for mov_mi in range(len(mov_masks)):
                if mov_masks[mov_mi] is None: continue
                if mov_mi not in mov_to_tgt: continue
                tgt_mi = mov_to_tgt[mov_mi]
                if tgt_masks[tgt_mi] is None: continue
                mov_mask.append(mov_masks[mov_mi])
                tgt_mask.append(tgt_masks[tgt_mi])
            assert len(mov_mask) <= 2, len(mov_mask)
            if len(mov_mask) > 1: # corner of tile
                mov_mask = torch.logical_and(mov_mask[0], mov_mask[1])
                tgt_mask = torch.logical_and(tgt_mask[0], tgt_mask[1])
            else: # edge of tile
                mov_mask = mov_mask[0]
                tgt_mask = tgt_mask[0]
            mov_mask = torch.where(mov_mask)[0]
            tgt_mask = torch.where(tgt_mask)[0]
            bbox_tgt = bbox_nei[tgt_mask]
            blabel_tgt = label_nei[tgt_mask]
            bbox_mov = bbox_tile[mov_mask]
            blabel_mov = label_tile[mov_mask]
            #########################
            # bbox_tgt = bbox_nei
            # blabel_tgt = label_nei
            # bbox_mov = bbox_tile
            # blabel_mov = label_tile
            #########################
            ## minimal iou threshold
            print(datetime.now(), f"NMS between {i}-{j}, {i+pi}-{j+pj}", bbox_mov.shape, bbox_tgt.shape)
            if len(bbox_tgt) == 0 or len(bbox_mov) == 0: continue
            # default threshold = 0.001
            rm_ind_tgt, rm_ind_mov = nms_bbox(
                bbox_tgt, bbox_mov, iou_threshold=0.1, 
                tile_tgt_center=tile_center[f'UltraII[{i+pi:02d} x {j+pj:02d}]'], tile_mov_center=tile_center[k], 
                seg_shape=[seg_shape[k][1], seg_shape[k][2]], device=device
            )
            if rm_ind_tgt is None: continue

            rm_mask_tgt = torch.zeros(len(bbox_tgt), device=rm_ind_tgt.device, dtype=bool)
            rm_mask_tgt[rm_ind_tgt] = True
            rm_mask_mov = torch.zeros(len(bbox_mov), device=rm_ind_mov.device, dtype=bool)
            rm_mask_mov[rm_ind_mov] = True

            if f'UltraII[{i+pi:02d} x {j+pj:02d}]' not in rm_label: rm_label[f'UltraII[{i+pi:02d} x {j+pj:02d}]'] = []
                
            rm_label[f'UltraII[{i+pi:02d} x {j+pj:02d}]'].append(blabel_tgt[rm_mask_tgt])
            rm_label[k].append(blabel_mov[rm_mask_mov])

            rm_mask_nei = torch.zeros(len(bbox_nei), device=bbox_nei.device, dtype=bool)
            rm_mask_tile = torch.zeros(len(bbox_tile), device=bbox_tile.device, dtype=bool)
            #####################################
            rm_mask_nei[tgt_mask[rm_ind_tgt]] = True
            rm_mask_tile[mov_mask[rm_ind_mov]] = True
            #####################################
            # rm_mask_nei[rm_ind_tgt] = True
            # rm_mask_tile[rm_ind_mov] = True
            #####################################
            bbox_tile = bbox_tile[torch.logical_not(rm_mask_tile)]
            label_tile = label_tile[torch.logical_not(rm_mask_tile)]
            stack_nis_bbox[f'UltraII[{i+pi:02d} x {j+pj:02d}]'] = stack_nis_bbox[f'UltraII[{i+pi:02d} x {j+pj:02d}]'][torch.logical_not(rm_mask_nei).cpu()]
            stack_nis_label[f'UltraII[{i+pi:02d} x {j+pj:02d}]'] = stack_nis_label[f'UltraII[{i+pi:02d} x {j+pj:02d}]'][torch.logical_not(rm_mask_nei).cpu()]

        stack_nis_bbox[k] = bbox_tile
        stack_nis_label[k] = label_tile
    print(nms_computed)
    for k in rm_label:
        if len(rm_label[k]) > 0:
            rm_label[k] = torch.cat(rm_label[k]).cpu()
        else:
            rm_label[k] = torch.zeros(0)
        print(rm_label[k].shape, "removals being recorded in tile", k)
    os.makedirs(f'{save_path}/doubled_NIS_label', exist_ok=True)
    torch.save(rm_label, f'{save_path}/doubled_NIS_label/{btag}_doubled_label_byNapari.zip')
    torch.cuda.empty_cache()
    return rm_label

def nms_bbox(bbox_tgt, bbox_mov, iou_threshold=0.01, tile_tgt_center=None, tile_mov_center=None, seg_shape=None, device=None):
    # remove touching boundary boxes first
    rm_mask_tgt = (bbox_tgt[:, 1] == 0) | (bbox_tgt[:, 1] == seg_shape[0]) | (bbox_tgt[:, 4] == 0) | (bbox_tgt[:, 4] == seg_shape[0]) | \
    (bbox_tgt[:, 2] == 0) | (bbox_tgt[:, 2] == seg_shape[1]) | (bbox_tgt[:, 5] == 0) | (bbox_tgt[:, 5] == seg_shape[1])
    rm_mask_mov = (bbox_mov[:, 1] == 0) | (bbox_mov[:, 1] == seg_shape[0]) | (bbox_mov[:, 4] == 0) | (bbox_mov[:, 4] == seg_shape[0]) | \
    (bbox_mov[:, 2] == 0) | (bbox_mov[:, 2] == seg_shape[1]) | (bbox_mov[:, 5] == 0) | (bbox_mov[:, 5] == seg_shape[1])
    remain_id_tgt = torch.where(~rm_mask_tgt)[0]
    remain_id_mov = torch.where(~rm_mask_mov)[0]
    bbox_tgt = bbox_tgt[remain_id_tgt]
    bbox_mov = bbox_mov[remain_id_mov]
    # distance to tile center
    tgt_cx = (bbox_tgt[:, 1] + bbox_tgt[:, 4]) / 2
    tgt_cy = (bbox_tgt[:, 2] + bbox_tgt[:, 5]) / 2
    tgt_dis_to_tctr = ((tgt_cx - tile_tgt_center[0])**2 + (tgt_cy - tile_tgt_center[1])**2).sqrt()
    tgt_dis_to_tctr = (tgt_dis_to_tctr - tgt_dis_to_tctr.min()) / (tgt_dis_to_tctr.max() - tgt_dis_to_tctr.min())
    mov_cx = (bbox_mov[:, 1] + bbox_mov[:, 4]) / 2
    mov_cy = (bbox_mov[:, 2] + bbox_mov[:, 5]) / 2
    mov_dis_to_tctr = ((mov_cx - tile_mov_center[0])**2 + (mov_cy - tile_mov_center[1])**2).sqrt()
    mov_dis_to_tctr = (mov_dis_to_tctr - mov_dis_to_tctr.min()) / (mov_dis_to_tctr.max() - mov_dis_to_tctr.min())
    # compute iou
    D = int(bbox_mov.shape[1]/2)
    area_tgt = box_area(bbox_tgt, D)
    area_mov = box_area(bbox_mov, D)
    iou, index = box_iou(bbox_tgt, bbox_mov, D, area1=area_tgt, area2=area_mov)
    if iou is None: return None, None
    ## scatter max among each of mov bbox (use this)
    max_iou, argmax = scatter_max(iou, index[1])
    valid = torch.where(max_iou>0)[0]
    if len(valid) == 0: return None, None
    ## scatter max among each of tgt bbox
    # max_iou, argmax = scatter_max(iou, index[0])
    ## max iou larger than threshold
    ## adaptive threshold based on distance to tile center
    threshold_e = mov_dis_to_tctr[index[1, argmax[valid]]] * tgt_dis_to_tctr[index[0, argmax[valid]]]
    # threshold_e = -1 * threshold_e
    threshold_e = (threshold_e - threshold_e.min()) / (threshold_e.max()- threshold_e.min())
    threshold_e = threshold_e.clip(min=0.05, max=0.5)
    iou_threshold = iou_threshold * threshold_e
    big_iou = torch.where(max_iou[valid] >= iou_threshold)[0]
    if len(big_iou) == 0: return None, None
    remove_ind_mov = index[1, argmax[valid][big_iou]]
    remove_ind_tgt = index[0, argmax[valid][big_iou]]
    ## remove small area of tgt or mov bbox
    # remove_area_tgt = area_tgt[remove_ind_tgt] 
    # remove_area_mov = area_mov[remove_ind_mov] 
    # remove_ind_tgt = remove_ind_tgt[remove_area_tgt < remove_area_mov].unique()
    # remove_ind_mov = remove_ind_mov[remove_area_mov <= remove_area_tgt].unique()
    ## remove tgt or mov bbox randomly
    # rand_choose = torch.rand(len(remove_ind_tgt)) >= 0.5
    # remove_ind_tgt = remove_ind_tgt[rand_choose]
    # remove_ind_mov = remove_ind_mov[torch.logical_not(rand_choose)]
    ## adaptive randomness of removing bbox based on the distance to tile center
    tgt_cx = (bbox_tgt[remove_ind_tgt, 1] + bbox_tgt[remove_ind_tgt, 4]) / 2
    tgt_cy = (bbox_tgt[remove_ind_tgt, 2] + bbox_tgt[remove_ind_tgt, 5]) / 2
    dis_to_tctr = ((tgt_cx - tile_tgt_center[0])**2 + (tgt_cy - tile_tgt_center[1])**2).sqrt()
    dis_to_tctr = (dis_to_tctr - dis_to_tctr.min()) / (dis_to_tctr.max() - dis_to_tctr.min())
    # print(dis_to_tctr.max(), dis_to_tctr.min())
    rand_choose = torch.rand(len(remove_ind_tgt)).to(device) >= dis_to_tctr
    remove_ind_tgt = remove_ind_tgt[rand_choose]
    remove_ind_mov = remove_ind_mov[torch.logical_not(rand_choose)]
    ## use original index
    remove_ind_tgt = torch.cat([remain_id_tgt[remove_ind_tgt], torch.where(rm_mask_tgt)[0]])
    remove_ind_mov = torch.cat([remain_id_mov[remove_ind_mov], torch.where(rm_mask_mov)[0]])
    return remove_ind_tgt, remove_ind_mov

def box_iou(boxes1, boxes2, D=2, area1=None, area2=None):
    if area1 is None:
        area1 = box_area(boxes1, D)
    if area2 is None:
        area2 = box_area(boxes2, D)
    index = []
    inter = []
    lrange = 100
    if lrange*boxes2.shape[0] >= 2147483647: # INT_MAX
        lrange = int(2147483647 / boxes2.shape[0])
    # for i in trange(0, len(boxes1), lrange, desc=f'Compute IoU between {boxes1.shape} and {boxes2.shape} boxes'):
    for i in range(0, len(boxes1), lrange):
        lt = torch.max(boxes1[i:i+lrange, None, :D], boxes2[:, :D])  # [N,M,D]
        rb = torch.min(boxes1[i:i+lrange, None, D:], boxes2[:, D:])  # [N,M,D]
        # print(lt.shape, rb.shape)
        ind1, ind2 = torch.where((rb > lt).all(dim=-1))
        if len(ind1) == 0: continue
        wh = rb[ind1, ind2] - lt[ind1, ind2]
        assert (wh>0).all()
        inter.append(wh.cumprod(-1)[..., -1]) # [N*M]
        ind1 = ind1 + i
        index.append(torch.stack([ind1, ind2])) # [2, N*M]
    if len(inter) == 0: return None, None
    inter = torch.cat(inter)
    index = torch.cat(index, 1)
    union = area1[index[0]] + area2[index[1]] - inter
    iou = inter / union
    return iou, index

def box_area(bbox, D):
    wh = bbox[:, D:] - bbox[:, :D]
    return torch.cumprod(wh, dim=1)[:, -1]

def bbox_in_stitching_seam(bbox, lt_loc, wh, i, j, ncol, nrow, nms_r):
    # mask = [left, right, bottom, top]
    masks = [None, None, None, None]
    if i > 0:
        # 0 ~ nms_r
        mask = bbox[:, 1] < lt_loc[0] + wh[0]*(nms_r)
        masks[0] = mask

    if i < ncol-1:
        # 1-nms_r ~ 1
        mask = bbox[:, 4] > lt_loc[0] + wh[0]*(1-nms_r)
        masks[1] = mask

    if j > 0:
        # 0 ~ nms_r
        mask = bbox[:, 2] < lt_loc[1] + wh[1]*nms_r
        masks[2] = mask

    if j < nrow-1:
        # 1-nms_r ~ 1
        mask = bbox[:, 5] > lt_loc[1] + wh[1]*(1-nms_r)
        masks[3] = mask

    return masks

def compute_contrast_limit(data):
    contrast_limits = [
        compute(np.min(data))[0],
        compute(np.max(data))[0]]
    if contrast_limits[0] == contrast_limits[1]:
        contrast_limits[1] = contrast_limits[1] + 1
    return contrast_limits

def sort_fnlist(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]


def format_number(num):
    if num >= 1e9:
        return f"{num / 1e9:03.2f}B"  # Billion
    elif num >= 1e6:
        return f"{num / 1e6:03.2f}M"  # Million
    elif num >= 1e3:
        return f"{num / 1e3:03.2f}K"  # Thousand
    else:
        return str(num)


def get_minxyz(tree, item, xshape=0, yshape=0):
    if tree.getroot().attrib['Direction'] == 'RightUp':
        return float(item['MinZ']), float(item['MinY']), float(item['MinX'])
    if tree.getroot().attrib['Direction'] == 'RightDown':
        return float(item['MinZ']), yshape-float(item['MinY']), float(item['MinX'])
    if tree.getroot().attrib['Direction'] == 'LeftUp':
        return float(item['MinZ']), float(item['MinY']), xshape-float(item['MinX'])
    if tree.getroot().attrib['Direction'] == 'LeftDown':
        return float(item['MinZ']), yshape-float(item['MinY']), xshape-float(item['MinX'])