# analyseLIME

Scripts to help analyse [LIME](https://github.com/lime-rt/lime) output.

---
---

## `readLAMDA`

Provides a class to read in the [LAMDA](http://home.strw.leidenuniv.nl/~moldata/) rate files and provide easy access to the variables.

An instance is created with

```python
ratefile('path/to/file')
```
---
---


#### `analyseCube.py`
Class to read in a LIME datacube and run basic analysis on it.





#### `readPopulation.py`
Class to read in LIME's `population.out` files and functions to explore emission and level populations.



*Note* These scripts make implicity assumptions about the data structures that you read in. Check that they are compatible.
