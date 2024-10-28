"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import os, json
from collections.abc import Iterable
import torch
import numpy as np
from dask import compute
from napari.utils import notifications

from magicgui import widgets
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QLabel, QFileDialog
from qtpy.QtGui import QPixmap
import xml.etree.ElementTree as ET 

# from multiview_stitcher import (
#     registration,
#     fusion,
#     spatial_image_utils,
#     msi_utils,
#     param_utils
#     )
from napari.layers import Image, Labels

# from napari_stitcher import _reader, viewer_utils, _utils

from pathlib import Path

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
        self.maptype_dropdown = widgets.Dropdown(choices=['cell count', 'avg volume'], name='Map type', allow_multiple=False)
        
        self.map_config_widgets_basic = [widgets.HBox(widgets=[
            self.org_res_x_textbox,
            self.org_res_y_textbox,
            self.org_res_z_textbox,]),widgets.HBox(widgets=[
            self.map_res_textbox,
            self.maptype_dropdown,])
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
        self.button_arrange_tiles = widgets.Button(text='Arrange tiles')
        # organize widgets        
        self.mosaic_widgets = widgets.VBox(widgets=[
            widgets.HBox(widgets=[
                # self.overlap,
            self.ncol, 
            self.nrow,]), widgets.HBox(widgets=[
            self.stitchtype_dropdown,
            self.button_arrange_tiles,])
        ])
        self.all_widgets.append(self.mosaic_widgets)


        self.container = QWidget()
        self.container.setLayout(QVBoxLayout())

        for w in self.all_widgets:
            if hasattr(w, 'native'):
                self.container.layout().addWidget(w.native)
            else:
                self.container.layout().addWidget(w)

        self.container.setMinimumWidth = 5
        self.layout().addWidget(self.container)

        self.button_load_cprof.clicked.connect(self.load_cell_profile_raw)
        self.button_select_list.clicked.connect(self.load_cell_profile_name)
        self.button_gen_brainmap.clicked.connect(self.whole_brainmap)
        self.button_arrange_tiles.clicked.connect(self.arrange_brainmap_layer)

        self.raw = {}
        self.raw_stitched = {}
        self.seg_shape = {}
        self.brainmap_layers = []
        self.brainmap_layernames = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def arrange_brainmap_layer(self):
        xmlfile_root = f'/cajal/Felix/Lightsheet/stitching/Manual_aligned_{self.btag}'
        stitch_path = '/'.join(self.cprof_root.split('/')[:-1]) + '/' + self.btag
        stitch_path = stitch_path.replace('results', 'stitch_by_ptreg').replace('P4/', '')
        overlap_r = 0.2#self.overlap.value
        ij_list = [[i, j] for i in range(self.ncol.value) for j in range(self.nrow.value)]
        for di, d in enumerate(self.seg_shape):
            seg_shape = self.seg_shape[d]
            tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
            i, j = ij_list[di]
            k = f'{i}-{j}'
            tile_lt_x, tile_lt_y = i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)
            if self.stitchtype_dropdown.current_choice == 'N/A':
                if 'N/A' not in self.raw_stitched: self.raw_stitched['N/A'] = {}
                if d not in self.raw_stitched['N/A']: self.raw_stitched['N/A'][d] = self.raw[d]
                tz = 0

            if self.stitchtype_dropdown.current_choice in ['Coarse', 'Refine']:
                tform_stack_coarse = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_coarse.json', 'r', encoding='utf-8'))
                tz, tx, ty = tform_stack_coarse[k]
                tile_lt_x = tile_lt_x + tx
                tile_lt_y = tile_lt_y + ty
                if 'Coarse' not in self.raw_stitched: self.raw_stitched['Coarse'] = {}
                if d not in self.raw_stitched['Coarse']: self.raw_stitched['Coarse'][d] = self.raw[d]
            
                if self.stitchtype_dropdown.current_choice == 'Refine':
                    if 'Refine' not in self.raw_stitched: self.raw_stitched['Refine'] = {}
                    if d in self.raw_stitched['Refine']: continue
                    if os.path.exists(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine.json'):
                        tform_stack_refine = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine.json', 'r', encoding='utf-8'))
                    if os.path.exists(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine_ptreg.json'):
                        tform_stack_refine = json.load(open(f'{stitch_path}/NIS_tranform/{self.btag}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
                    center = self.raw_stitched['Coarse'][d][0].clone()
                    ct_z = (center[:, 0].clone() + tz).long()
                    pre_refine_lt = None
                    for zi in range(len(tform_stack_refine)):
                        ct_zmask = torch.where(ct_z == zi)[0]
                        if len(ct_zmask) == 0: continue
                        if k in tform_stack_refine[zi]:
                            tx, ty = tform_stack_refine[zi][k]
                            if (abs(tx) > tform_xy_max[0] or abs(ty) > tform_xy_max[1]):
                                if pre_refine_lt is not None:
                                    tx, ty = pre_refine_lt
                                else:
                                    for zii in range(zi, len(tform_stack_refine)):
                                        tx, ty = tform_stack_refine[zii][k]
                                        if abs(tx) <= tform_xy_max[0] and abs(ty) <= tform_xy_max[1]:
                                            break
                            else:
                                tx, ty = tform_stack_refine[zi][k]
                        else:
                            tx, ty = 0, 0
                        pre_refine_lt = [tx, ty]
                        center[ct_zmask, 1] = center[ct_zmask, 1] + tx
                        center[ct_zmask, 2] = center[ct_zmask, 2] + ty
                    self.raw_stitched['Refine'][d] = [center, self.raw_stitched['Coarse'][d][1], self.raw_stitched['Coarse'][d][2]]

            if self.stitchtype_dropdown.current_choice == 'Manual':
                if 'Manual' not in self.raw_stitched: self.raw_stitched['Manual'] = {}
                if d not in self.raw_stitched['Manual']:
                    center, _, _ = self.raw_stitched['N/A'][d]
                    xshape = self.seg_shape[d][1] * self.ncol.value
                    ct_z = center[:, 0].clone().long()
                
                tz, tile_lt_x, tile_lt_y = None, None, None
                for fn in os.listdir(xmlfile_root):
                    if not fn.endswith('.xml'): continue
                    xmlfile = f'{xmlfile_root}/{fn}'
                    tree = ET.parse(xmlfile) 
                    for i in range(len(tree.getroot()[0])):
                        if tree.getroot()[0][i].tag == 'Image':
                            item = tree.getroot()[0][i].attrib
                            zsplit, ims_fn = item['Filename'].split('\\')[-2:]
                            if d in ims_fn:
                                minz, miny, minx = get_minxyz(tree, item, xshape)
                                if tile_lt_x is None:
                                    tz, tile_lt_x, tile_lt_y = minz, miny, minx
                                elif d not in self.raw_stitched['Manual']:
                                    zmin = int(zsplit.split('-')[0][1:])
                                    zmax = int(zsplit.split('-')[1][1:])
                                    ct_zmask = torch.where(torch.logical_and(ct_z >= zmin, ct_z < zmax))[0]
                                    if len(ct_zmask) > 0:
                                        center[ct_zmask, 0] = center[ct_zmask, 0] + (minz-tz)
                                        center[ct_zmask, 1] = center[ct_zmask, 1] + (miny-tile_lt_x)
                                        center[ct_zmask, 2] = center[ct_zmask, 2] + (minx-tile_lt_y)
                                break
            
                if d not in self.raw_stitched['Manual']: self.raw_stitched['Manual'][d] = [center, self.raw_stitched['N/A'][d][1], self.raw_stitched['N/A'][d][2]]
            
            if self.stitchtype_dropdown.current_choice in ['Refine', 'Manual']:
                self.raw[d] = self.raw_stitched[self.stitchtype_dropdown.current_choice][d]
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


    def load_cell_profile_raw(self):
        lrange = 1000
        zratio = 2.5/4
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
                vol = torch.cat(_vol).to(self.device)
                bbox = torch.cat(_bbox).to(self.device)
                label = torch.cat(_label).to(self.device)
                pt = (bbox[:, :3] + bbox[:, 3:]) / 2
                zstitch_remap_fn = f"{cprof_d}/{self.btag}_remap.zip"
                if os.path.exists(zstitch_remap_fn):
                    zstitch_remap = torch.load(zstitch_remap_fn).to(self.device)
                    print("before remove z-stitched pt", pt.shape, label.shape)
                    print("z-stitch remap dict shape", zstitch_remap.shape)

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
                    vol[pre_loc] = vol[loc] + vol[pre_loc]

                    pt = pt[keep_mask]
                    vol = vol[keep_mask]
                    label = label[keep_mask]
                    print("after remove z-stitched pt", pt.shape, label.shape)

                self.raw[d] = [pt, label, vol]
                self.select_list_status.choices = [f'cell #: {format_number(len(self.raw[update_d]))}' if update_d in self.raw else f'Not loaded' for update_d in self.tile_list]
                self.update_select_status()


    def whole_brainmap(self):
        for d in self.raw:
            self.update_brainmap_layer(d)

    def update_brainmap_layer(self, name):
        colormap = 'gray'
        blending = 'additive'
        spatial_dims = [3, 3, 3]
        affine_transform = np.eye(len(spatial_dims) + 1)
        center, _, vol = self.raw[name]
        dshape = self.seg_shape[name]
        brainmap = self.get_brainmap(center, vol, dshape)
        brainmap = brainmap.numpy().astype(np.float32)
        kwargs = {
            'contrast_limits': compute_contrast_limit(brainmap),
            'name': name,
            'colormap': colormap,
            'gamma': 0.9,
            'affine': affine_transform,
            'translate': np.array([0 for dim in spatial_dims]),
            'scale': np.array([self.map_res_textbox.value for dim in spatial_dims]),
            'cache': True,
            'blending': blending,
            'multiscale': True,
            'metadata': None,
        }
        multiscale_data = [brainmap, brainmap[::2, ::2, ::2], brainmap[::4, ::4, ::4]]
        if name in self.brainmap_layernames:
            self.brainmap_layers[self.brainmap_layernames.index(name)].data = multiscale_data
        else:
            brainmap_layer = self.viewer.add_image(multiscale_data, **kwargs)
            self.brainmap_layers.append(brainmap_layer)
            self.brainmap_layernames.append(name)

    def get_brainmap(self, center, vol, dshape):
        ratio = [s/self.map_res_textbox.value for s in [self.org_res_z_textbox.value, self.org_res_y_textbox.value, self.org_res_x_textbox.value]]
        center = center.clone().to(self.device)
        vol = vol.float().to(self.device)
        center[:,0] = center[:,0] * ratio[0]
        center[:,1] = center[:,1] * ratio[1]
        center[:,2] = center[:,2] * ratio[2]
        dshape = [int(dshape[0]*ratio[0]), int(dshape[1]*ratio[1]), int(dshape[2]*ratio[2])]
        # dshape = [int(center[:,0].max()+1), int(center[:,1].max()+1), int(center[:,2].max()+1)]
        outbound_mask = torch.logical_or(center[:, 0].round() > dshape[0]-1, center[:, 1].round() > dshape[1]-1)
        outbound_mask = torch.logical_or(outbound_mask, center[:, 2].round() > dshape[2]-1)
        center = center[torch.logical_not(outbound_mask)]
        z = center[:, 0]#.clip(min=0, max=dshape[0]-0.9)
        y = center[:, 1]#.clip(min=0, max=dshape[1]-0.9)
        x = center[:, 2]#.clip(min=0, max=dshape[2]-0.9)
        # print(center.shape, z, x, y)
        loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(self.device) 
        loc = loc[(z.round().long(), y.round().long(), x.round().long())] # all nis location in the downsample space
        loc_count = loc.bincount() 
        loc_count = loc_count[loc_count!=0] 
        atlas_loc = loc.unique().to(self.device) # unique location in the downsample space
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
            vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2]).float()#.to(self.device)
            vol_avg[atlas_loc] = (loc_vol.sum(1) / loc_num).cpu().float()
            # for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
            #     where_loc = torch.where(loc==loci)[0]
            #     vol_avg[loci] = vol[where_loc].mean()
            vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2])#.cpu()
            return vol_avg
        ## density map
        elif self.maptype_dropdown.current_choice == 'cell count':
            density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(self.device)
            density[atlas_loc] = loc_count.double() #/ center.shape[0]
            density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
            return density
        

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
        return float(item['MinZ']), xshape-float(item['MinY']), float(item['MinX'])
    if tree.getroot().attrib['Direction'] == 'LeftUp':
        return float(item['MinZ']), float(item['MinY']), yshape-float(item['MinX'])
    if tree.getroot().attrib['Direction'] == 'LeftDown':
        return float(item['MinZ']), xshape-float(item['MinY']), yshape-float(item['MinX'])