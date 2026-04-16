import laspy
import numpy as np
import open3d as o3d
# import pandas as pd
import seaborn as sns
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys

file=laspy.read(r'C:\Users\messa\Menu\my_coding\rmit\Programming Project\dev_loader\PointcloudAnalysis_IFP\preprocessed\bologna\low_featured.laz')

x,y,z=file.x,file.y,file.z

features=[x for x in file.point_format.dimension_names]

labels=np.array(file.classification).astype(int).transpose()

points=np.column_stack((x,y,z))

pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(points)

np.unique_counts(labels)

no_class=np.where(labels==0)
sidewalk=np.where(labels==2)
veg=np.where(labels==5)
furniture=np.where(labels==8)
street=np.where(labels==11)
temp=np.where(labels==13)
cars=np.where(labels==15)

palette = {
    0:  [0.8,0.8,0.8], # Not yet classified
    2:  [0.25, 0.25, 0.25], # Sidewalk
    5:  [0.0, 0.4, 0.0], # High Veg
    8:  [1.0, 0.9, 0.0], # Furniture
    11: [0.15, 0.15, 0.15], # Street
    13: [1.0, 0.4, 0.7], # Temp
    15: [1.0, 0.6, 0.0], # Vehicles
}

colors=np.array([palette.get(l) for l in labels])

label_layers={}
for count in palette:
    slicing_indices=np.where(labels==count)
    sliced_points=points[slicing_indices[0]]
    sliced_colours=colors[slicing_indices[0]]
    tempPointCloud=o3d.geometry.PointCloud()
    tempPointCloud.points=o3d.utility.Vector3dVector(sliced_points)
    tempPointCloud.colors=o3d.utility.Vector3dVector(sliced_colours)
    label_layers[count]=tempPointCloud

def refresh_function(layer_name:str,checked:bool,view_name,window_name):
    view_name.scene.show_geometry(layer_name,checked)
    window_name.post_redraw()

def on_layout(ctx):
    dimensions=win.content_rect
    em=win.theme.font_size
    sidebar_width=10*em
    view.frame=gui.Rect(dimensions.x,dimensions.y,dimensions.width-sidebar_width,dimensions.height)
    sidebar.frame=gui.Rect(dimensions.get_right()-sidebar_width,dimensions.y,sidebar_width,dimensions.height)

app=gui.Application.instance
app.initialize()
win=app.create_window('testing',1920,1080)
view=gui.SceneWidget()
view.scene=rendering.Open3DScene(win.renderer)
for count in label_layers:
    view.scene.add_geometry(f'layer_{count}',label_layers[count],rendering.MaterialRecord())
    
# creating the sidebar
sidebar=gui.Vert(5)

for count in palette:
    cb=gui.Checkbox(f'layer_{count}')
    sidebar.add_child(cb)
    cb.checked=True
    cb.set_on_checked(lambda x, name=f'layer_{count}':refresh_function(name,x,view,win))
    
bounds=view.scene.bounding_box
view.setup_camera(60,bounds,bounds.get_center())
    
win.add_child(view)
win.add_child(sidebar)
win.set_on_layout(on_layout)
app.run()
