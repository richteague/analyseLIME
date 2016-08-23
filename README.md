# analyseLIME

Scripts and functions to help analyse [LIME](https://github.com/lime-rt/lime) output. Works best with output from [makeLime](https://github.com/richteague/makeLime), however any LIME output is possible.

---

## `analysecube.py`

Basic functions to plot and analyse a datacube from LIME. Will read in the cube and allows one to plot zeroth and first moment maps, along with calculate radial profiles of the zeroth moment. Simply read the file with:

```python
from analysecube import cube 
datacube = cube('path/to/cube.fits')
```

---

## `readLAMDA.py`

Provides a class to read in the [LAMDA](http://home.strw.leidenuniv.nl/~moldata/) rate files and provide easy access to the variables. An instance is created simplying by providing the path to the collisional rate file.

```python
rates = ratefile('path/to/file.dat')
```

Then you have access to all the attributes, for example `file.molecule` will return the name of the molecule of the file. This is used extensively in the `readpopulation.py` class.

---

## `readpopulation.py`

Class to read in LIME population output file and analyse the physical structure.

```python
from readpopulation import population as pop
model = pop('path/to/popfile.dat', 'path/to/collisionalrates.dat')
```

Allows one to estimate where the emission arises from in the disk, as well as calculate flux weighted variables. 
