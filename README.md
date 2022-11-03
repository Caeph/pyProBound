# pyProBound
Python package for scoring sequences by a ProBound model

This is an interface package to score sequences by models of transcription factor affinity produced by ProBound (Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0). 

This is only an interface. The functional part of the scoring is provided by a (sligthly modified) ProBoundTools Java program available from https://github.com/Caeph/ProBoundTools.git. The original program can be found at https://github.com/BussemakerLab/ProBoundTools. 

## Requirements
Python>=3.9 with numpy, jpype 1.4.0 and pandas. Installed Java in your path.

## Usage
Package contains two classes:
**MotifCentral**: a class representing already available models from https://motifcentral.org/.
You can get the *fit_id* of a model you need, using filters as a taxon, source study or a transcription molecule.

It is currently based on v1.0.0 of the database.

The command
```
from pyProBound import MotifCentral

database = MotifCentral()
database.filter(taxa=["Drosophila melanogaster"], 
                publications=["Isakova2017", "Jolma2013"])
```
returns a Pandas DataFrame containing rows that comply with filters.

**ProBoundModel**: a class representing a binding model.

You can get a model from the MotifCentral database, if you know its fit ID.
```
from pyProBound import ProBoundModel
model = ProBoundModel(1000, motifcentral=True) 
```

You can use the ProBound webserver (http://pbdemo.x3dna.org) and fit your own model. 
You can then load it from the model json.
```
model = ProBoundModel("fit.sox2.json", fitjson=True)
```
where ```"fit.sox2.json"``` is a path to the file. 

Finally, you can load a prepared model from json.
```
model = ProBoundModel("model.json")
```
where ```"model.json"``` is a path to the file.

Using the loaded model, you can manipulate it (selecting and removing binding modes, ...). 
You can also score sequences by the following methods:

```
model.score_affinity_sum(sequences)
model.score_binding_mode_scores(seqs, score_format=...)
```
See the example jupyter notebooks in the github repository for more details.