#!/usr/bin/env python3
r"""
Monte-Carlo Acquisition Functions for Multi-objective Bayesian optimization.

"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import torch
from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class qMultiProbabilityOfImprovement(MultiObjectiveMCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement.
    Probability of improvement over the current best observed value,
    Only supports the case of q=1. Requires the posterior to be Gaussian.
    The model must be multi-outcome.
    Estimates the probability of improvement over the current best observed
    values by sampling from the joint posterior distribution of the q-batch.
    MC-based estimates of a probability involves taking expectation of an
    indicator function; to support auto-differntiation, the indicator is    
    replaced with a sigmoid function with temperature parameter `eta`.
    two objective case:
    `qMPI(X) = P(Y1 >= best_f) * P(Y2 >= best_f2),`
    ` Y1 ~ f(X), Y2 ~ f(X), X = (x_1,...,x_q)`
    Example:
        >>> model1 = SingleTaskGP(train_X, train_Y[0, :])
        >>> model2 = SingleTaskGP(train_X, train_Y[1, :])
        >>> # we assume the outputs are independent each other.
        >>> best_f = train_Y.max(dim=1)
        >>> modes = ModelListGP([model1, model2])
        >>> qPI = qMultiProbabilityOfImprovement(models, best_f)
        >>> qmpi = qMPI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[List[float], Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        constraints: Optional[Callable[[Tensor], Tensor]] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
    ) -> None:
        r"""q-Multi Probability of Improvement.
        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can
                be a `batch_shape x objectvies`-shaped tensor,
                which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler(not test): The sampler used to draw base samples.
                See `MCAcquisitionFunction` more details.
            objective(not test): The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform(not test): A PosteriorTransform (optional).
            X_pending(not test):  A `m x d x objectives`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            constraints(not test): A list of callables, each mapping
                a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            eta(not test): The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.posterior_transform = posterior_transform
        self.best_f = best_f
        self.eta = eta

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d x objectives`-dim Tensor of t-batches with
                `q` `d`-dim design points each.
        Returns:
            A `batch_shape'`-dim Tensor of Probability of Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """

        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(
            samples, X=X
        )  # `sample_shape x batch_shape x q x models`-dim
        max_obj = obj.max(dim=-2)[0]  # `sample_shape x batch_shape x models`-dim
        impr = max_obj - self.best_f.to(max_obj)
        vals = torch.sigmoid(impr / self.eta).mean(dim=0)  # mean of samples
        prod_val = torch.prod(vals, 1)  # batch_shape
        return prod_val