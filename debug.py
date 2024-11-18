import os

from napari import Viewer, run
viewer = Viewer(show=False)
dw, self = viewer.window.add_plugin_dock_widget(
"napari-whole-brain-map", "Brainmap"
)
self.btag_split_key = '_'
self.btag_split_i = 1
folder = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/pair6/220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
self.cprof_root = folder
btag = folder.split('/')[-1].split(self.btag_split_key)[self.btag_split_i]
self.btag = btag
cprof_tile_name = [d for d in os.listdir(folder) if os.path.exists(f'{folder}/{d}/{btag}_NIScpp_results_zmin0_instance_center.zip')] # and os.path.isdir(f'{folder}/{d}')
self.tile_list = sorted(cprof_tile_name)
self.tile_selection.choices = self.tile_list
self.tile_selection.current_choice = self.tile_list
self.stitchtype_dropdown.current_choice = 'Manual'
self()
self.load_cell_profile_raw()
self.whole_brainmap()
self.arrange_brainmap_layer()

run()