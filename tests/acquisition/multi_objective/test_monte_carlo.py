from itertools import product

import torch
from botorch.utils.testing import BotorchTestCase
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from botorchex.acquisition.multi_objective.monte_carlo import (
    qMultiProbabilityOfImprovement,
)

class TestqMultiProbabilityOfImprovement(BotorchTestCase):
    def setUp(self):
        self.ref_point = [0.0, 0.0, 0.0]
        self.Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        self.pareto_Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
            ],
            device=self.device,
        )

    def test_q_probability_of_improvement(self):
        tkwargs = {"device": self.device}
        for dtype, m in product(
            (torch.float, torch.double),
            (1, 2, 3),
        ):
            tkwargs["dtype"] = dtype
            Y = self.Y_raw[:, :m].to(**tkwargs)
            X_baseline = torch.rand(Y.shape[0], 1, **tkwargs)
            # the event shape is `b x q + r x m` = 1 x 1 x 2
            models = []
            # samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            for i in range(m):
                mm = SingleTaskGP(
                    X_baseline,
                    torch.index_select(
                        Y,
                        1,
                        torch.tensor(
                            [
                                i,
                            ]
                        ),
                    ),
                )
                models.append(mm)
            # basic test
            models = ModelListGP(*models)

    def test_fesibility_multi_q_probability_of_improvement(self):
        tkwargs = {"device": self.device}

        def test_function(x):
            return -(x**2)

        for dtype, m in product(
            (torch.float, torch.double),
            (1,),
        ):
            x = torch.rand(16, 1, m) * 3
            y = test_function(x)
            tkwargs["dtype"] = dtype
            Y = y.to(**tkwargs)
            X_baseline = x.to(**tkwargs)
            # the event shape is `b x q + r x m` = 1 x 1 x 2
            models = []
            # samples = torch.zeros(1, 1, 1, device=self.device, dtype=dtype)
            for i in range(m):
                _train_x = torch.index_select(
                    X_baseline,
                    2,
                    torch.tensor(
                        [
                            i,
                        ]
                    ),
                )
                _train_y = torch.index_select(
                    Y,
                    2,
                    torch.tensor(
                        [
                            i,
                        ]
                    ),
                )
                mm = SingleTaskGP(_train_x, _train_y)
                models.append(mm)

            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            best_f = torch.amax(y, dim=[0, 1])  # reduce batch_shape x sample_shape
            models = ModelListGP(*models)
            acqf = qMultiProbabilityOfImprovement(
                model=models,
                best_f=best_f,
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            # acqf.model._posterior._samples = samples
            res = acqf(X)
            expected_shape = torch.Size([16])
            self.assertEqual(res.shape, expected_shape)