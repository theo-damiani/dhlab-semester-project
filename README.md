# Automatic georeferencing of historical cadastres

Abstract-This semester project has been hosted by the Digital Humanities Laboratory at EPFL founded by Professor
Frédéric Kaplan. It will focus on the georeferencing, i.e. spatial recalibration of 19th-century "Napoleonic"
cadastral sheets. This project attempts to address the need for the automatic grouping of huge corpora
like the Napoleonic cadastres. Indeed, a single map at the level of a city or a region would be a significant
tool to facilitate the studies of historians and geographers.
Our method focuses on alignment by pattern matching. A genetic algorithm is used to search for the
match of a pattern in another image. Several patches are extracted from the anchor image, and then, the
best match is employed to compute the homography from the anchor to the target image.
Our best results were obtained with a restricted GA, the loss function Intersection Over Union and a disruptive
selection of the best match.

##

<p align="center">
  <img src="results/stats/al001002.png" alt="Berney 001-002 Alignement" width="500"/>  
</p>

##
Requirements:
```
geneticalgorithm
opencv-python
scikit-image
Pillow
scipy
numpy
tqdm
json
ast
csv
```
##

For testing purpose:
  - Open RUN-MODEL.ipynb:
      - Specify which cadastres you want to align with the variable list_cadastres_name.
      - Run the notebook. Results will be saved in /results/csv/.
  - Create a new folder in results/csv/ and drag your results in it.
  - Open READ-EXP-STATS.ipynb:
      - Specify the folder of your results in the variable EXPNAME.
      - Run the notebook to display plots like in the report.
  - For specific result on one alignement see the notebook READ-EXP.ipynb
