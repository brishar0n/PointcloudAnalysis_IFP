To run the entire pipeline, run the following command: `python .\pipeline.py {path to dataset} --city {name of city used} --all`
example: `python .\pipeline.py datasets\bologna_subsampled.laz --city bologna --all`

Ensure that all datasets that are to be preprocessed go into a dataset folder taht follows this hierarchy:
```text
POINTCLOUDANALYSIS_IFP/
├── classification/
├── datasets/
├── metrics/
├── preprocessing/
├── processing/
├── visualisation/
│   ├── potree_vis/
│   │   ├── build/
│   │   ├── libs/
│   │   └── index.html
│   └── PotreeConverter_1.7_windows_x64/
├── .gitignore
├── pipeline.py
├── README.md
└── requirements.txt
```

All preprocessed files are found in the `classification\preprocessed` folder.

**For the classification process:**
Each city's input is a single file: `preprocessed/<city>/low_featured.laz`

If these files are available to use, since preprocessing is very time consuming, you could comment out the `preprocessing() function` from the pipeline.py in the main function to simply run the classification -> processing -> width_metrics -> visualisation pipeline. 

**For visualisation, once pipeline.py has been completely executed, change the cwd using `cd` in the terminal to `visualisation\potree_vis`**
The output for required potree conversion of pointclouds is found in the `visualisation\potree_vis\pointclouds` folder, following the hierarchy:
```text
visualisation/
├── potree_vis/
│   ├── build/
│   ├── libs/
│   ├── pointclouds/
│   │   └── <city>/
│   │       ├──city
│   │       └──sidewalk
│   └── index.html
└── PotreeConverter_1.7_windows_x64   

where city and sidewalk folders in the <city> folder, contain the converted pointclouds consisting of all points in the city, and only sidewalk points respectively.

```
Having changed the cwd, run the command `python -m http.server 8000`, which runs runs the potree visualisation on the localhost which can be seen using the link: **http://localhost:8000/**.