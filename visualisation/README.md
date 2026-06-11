In the visualisation module, two basic steps are being executed - 

`Pointcloud conversion --> Visualisation using Potree`.

Pointcloud conversion converts the classified points along with sidewalk widths into a format usable to Potree. This step is executed in the `visualise()` function.

The function converts two sets of pointcloud points -
1. segmented classified points for the entire city
2. segmented classified points for the sidewalk

The latter is used for highlighting the sidewalk points specifically along with the sidewalk segments in the potree visualisation, and is displayed at an offset of `.7` along the Z-axis. This is done so that when highlighting sidewalk segments in the visualisation, only the sidewalk points are being highlighted . This creates a differentiation between the street and sidewalk points. If only one pointcloud is loaded, the entire pointcloud is coloured based on the segments. This allows us to view the segments, but assigns the same colours that are also found on the sidewalk to the entire city, making the visualisation more difficult to understand.

Having converted both of these sets, the converted points are found in the pointclouds folder in a folder named after the city given in the terminal command (eg. --city bologna), with the points for the city and sidewalk found in their respective folders.
```text
visualisation/
├── potree_vis/
│   ├── build/
│   ├── libs/
│   ├── pointclouds/
│   │   └── <city>/
│   │       ├──city
│   │       └──sidewalk
```
The visualisation is determined by the index.html file found in the same visualisation folder. 

In order to see visualisations of the points that you have converted, simply change the path defined in the line - `Potree.loadPointCloud("pointclouds/bologna/city/cloud.js", "City")...` to the path of the new city that has been converted. Similarly you can do the same for the sidewalk points. 