# PyGenes
##### A fast and simple implementation of the genetic algorithm in Python.



---
Example of use:


```python
@jit
def objective(x):
    att = x.get_attributes()
    w1, w2 = att[0], att[1]
    b1, b2 = att[2] * 10 - 5, att[3] * 10 - 5
    m1, m2 = att[4] + 1, att[5] + 1

    t = w1 * ((np.array(training['dew_point_ukmet'].values) + b1) ** m1) + \
        w2 * ((np.array(training['dew_point_gfs'].values) + b2) ** m2)

    if not np.isfinite(t).all(): return 1e6
    r2 = r2_score(training['target_Dew_Point'], t)
    return 1 - r2


@jit
def testing(x):
    att = x.get_attributes()
    w1, w2 = att[0], att[1]
    b1, b2 = att[2] * 10 - 5, att[3] * 10 - 5
    m1, m2 = att[4] + 1, att[5] + 1

    t = w1 * ((np.array(validation['dew_point_ukmet'].values) + b1) ** m1) + \
        w2 * ((np.array(validation['dew_point_gfs'].values) + b2) ** m2)

    if not np.isfinite(t).all(): return 1e6
    mse = np.sqrt(mean_squared_error(validation['target_Dew_Point'], t))
    r2 = r2_score(validation['target_Dew_Point'], t)
    return mse, 1 - r2


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
