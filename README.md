# pyProBound
Python package for scoring sequences by a ProBound model

This is an interface package to score sequences by models of transcription factor affinity produced by ProBound (Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0). 

This is only an interface. The functional part of the scoring is provided by a (sligthly modified) ProBoundTools Java program available from https://github.com/Caeph/ProBoundTools.git. The original program can be found at https://github.com/BussemakerLab/ProBoundTools. 

## Instalation
Not done yet. Pip package planned.

### Requirements
Python>=3.9 with numpy, jpype 1.4.0 and pandas. Installed Java in your path.

### Pip installation
Will be done.

### From source
Clone https://github.com/Caeph/ProBoundTools.git and compile it using Maven (details here: https://github.com/BussemakerLab/ProBoundTools). Move the compiled jar with dependencies to the pyProBound directory (next to the Python scripts and json files.)

## Usage
See the jupyter notebooks in the test_input directory.
