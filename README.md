# Functa for Earth Observation Data

> Using Functa (Neural Fields) for learning compressed representations of observation and reanalysis data.


## Running Scripts


**Help From Scripts**

```python
python functa4eo/_src/train.py --help
```

**Example Experiment Run** 
```bash
# SSH + SIREN MultiHead Model
python functa4eo/_src/train.py data=ssh model=siren_mh
# SST + SIREN MultiHead Model
python functa4eo/_src/train.py data=temperature model=siren_mh
```