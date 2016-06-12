# pygenetic
##### A fast and simple implementation of the genetic algorithm in Python.



---
Import into Python:

```python
import numpy as np
from numba import jit

from GeneticAlgorithm import *
```


Set up data variables:

```python

array_1, array_2, array_3 = [...], [...], [...]

train_valid_ratio = 0.9  # split into 90% training and 10% validation
train_valid_split = int(len(array_1)*train_valid_ratio)  # under the assumption they are all equal in size

array_1_training, array_1_validation = array_1[:train_valid_split], array_1[train_valid_split:]
array_2_training, array_2_validation = array_2[:train_valid_split], array_2[train_valid_split:]
array_3_training, array_3_validation = array_3[:train_valid_split], array_3[train_valid_split:]

```


We are going to try and minimize the right hand side (ie: the error) in the following equation:

<img src="http://www.sciweavers.org/upload/Tex2Img_1465766617/render.png" align="center" border="0" />

Here is an example of one way to approach this using _pygenetic_:


```python
@jit
def objective(x):
    att = x.get_attributes()
    
    # our function has the form (w1*x1+b1)^m1 + ...
    w1, w2 = att[0], att[1]  # multiplicative weights
    b1, b2 = att[2], att[3]  # additional biases
    m1, m2 = att[4], att[5]  # exponents of the bracket, ie: (w*x + b)

    t = w1 * ((array_1_training) + b1) ** m1) + \
        w2 * ((array_2_training) + b2) ** m2)

    if not np.isfinite(t).all(): return 1e6
    r2 = r2_score(array_3_training, t)
    return 1 - r2


@jit
def testing(x):
    att = x.get_attributes()
    w1, w2 = att[0], att[1]
    b1, b2 = att[2] * 10 - 5, att[3] * 10 - 5
    m1, m2 = att[4] + 1, att[5] + 1

    t = w1 * ((array_1_validation) + b1) ** m1) + \
        w2 * ((array_2_validation) + b2) ** m2)

    if not np.isfinite(t).all(): return 1e6
    mse = np.sqrt(array_3_validation, t))
    r2 = r2_score(array_3_validation, t)
    return mse, 1 - r2

```

Example of running the algorithm to optimize RMSE:

```python

population = GeneticAlgorithm.Population()
population.create_population(attributes_min=-10, attributes_max=10, attributes_size=6, population_size=25000)
population.sort_population(function=objective, maximize=False)

min_tmp_rmse, min_tmp_r2 = 1e6, 1e6
min_attributes = None

for I in range(1000):
    population.evolve_population(number_to_keep=25)
    population.crossover_population()
    population.mutate_population(0.1, 0.3, 0.3)
    population.sort_population(function=objective, maximize=False)

    t = testing(population.get_population()[0])
    if t[1] < min_tmp_r2:
        min_tmp_rmse, min_tmp_r2 = t[0], t[1]
        min_attributes = population.get_population()[0].get_attributes()

    print(I + 1, objective(population.get_population()[0]), "\t", t[0], "\t", 1-t[1])
```
