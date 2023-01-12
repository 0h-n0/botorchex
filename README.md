# botorchex

Botorch extention library including custom acquistion functions and surrogate models.

## Installation 

```
$ pip install botorchex
```

## Ease to use

Botorch compatible interface.

## Implementation List
### Custom Acqusition function

* Multi Objective Monte-Carlo Probability Improvement
This acquistion function can deacrease more computational resource(wall-time) comparing to other multi objective acqusition function. This performance especially is shown in the more than 3 objctive cases. However, the convergence speed is longer than the others and there is no theoretical background. 

```python
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from botorchex.acquisition.multi_objective.monte_carlo import qMultiProbabilityOfImprovement

model1 = SingleTaskGP(train_X, train_Y[0, :])
model2 = SingleTaskGP(train_X, train_Y[1, :])
# we assume the outputs are independent each other.
best_f = train_Y.max(dim=1)
modes = ModelListGP([model1, model2])
qPI = qMultiProbabilityOfImprovement(models, best_f)
qmpi = qMPI(test_X)
```

If you want to know more examples, you can check the example([multi_objective_bo.ipynb](https://github.com/0h-n0/botorchex/blob/main/tutorials/multi_objective_bo.ipynb))

### Custom Surrogates

* GNN based surrogates?

### Referances

* https://botorch.org/
* https://gpytorch.ai/
