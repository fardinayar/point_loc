from itertools import product
from typing import List, Optional, Sequence, Union, Callable
from torch.distributions import MultivariateNormal

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from tabulate import tabulate
from point_loc.registry import METRICS
from point_loc.datasets import matrix_utils

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

    
def _tensor_to_upper_triangular_matrix(tensor):
    """Convert a flat tensor to an upper triangular matrix."""
    n = tensor.size(0)
    if n == 0:
        return torch.zeros((0, 0), dtype=tensor.dtype)

    # Calculate the size of the matrix
    m = int(((-1 + (1 + 8 * n) ** 0.5) / 2))  # Solving m(m + 1)/2 = n

    # Create an empty matrix
    matrix = torch.zeros((m, m), dtype=tensor.dtype)

    # Create indices for the upper triangular part
    indices = torch.triu_indices(m, m, offset=0)
    
    # Fill the upper triangular part using advanced indexing
    matrix[indices[0], indices[1]] = tensor
    
    tabulate_matrix = [[char for char in 'abcxyz']]
    for i, char in enumerate('abcxyz'):
        tabulate_matrix.append([char] + matrix[i].tolist())

    return "\n" + tabulate(tabulate_matrix, headers="firstrow",  tablefmt="orgtbl") + "\n"



@METRICS.register_module()
class MeanAbsoluteError(BaseMetric):
    r"""Mean Absolute Error (MAE) evaluation metric.

    The MAE is the average of the absolute differences between predictions and
    actual values. It measures the average magnitude of the errors in a set of
    predictions, without considering their direction.

    .. math::

        \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from point_loc.evaluation import MeanAbsoluteError
        >>> y_pred = torch.tensor([[3.0, -0.5, 2.0, 7.0], [3.0, -0.5, 2.0, 7.0]])
        >>> y_true = torch.tensor([[2.5, 0.0, 2.0, 8.0], [2.5, 0.0, 3.0, 8.0]])
        >>> MeanAbsoluteError.calculate(y_pred, y_true)
        tensor(0.5)
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        assert 'gt_values' in data_samples[0], "Regression metrics assume that ground truth are stored in a field named gt_values"
        assert 'pred_values' in data_samples[0], "Regression metrics assume that model predictions are stored in a field named gt_values"
        
        for data_sample in data_samples:
            result = dict()
            result['pred'] = data_sample['pred_values'].cpu()
            result['gt'] = data_sample['gt_values'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List[dict]):
        metrics = {}

        # Concatenate all predictions and targets
        pred = torch.cat([res['pred'].unsqueeze(0) for res in results])
        target = torch.cat([res['gt'].unsqueeze(0) for res in results])
        mae = self.calculate(pred, target)
        metrics['\n mean absolute error'] = mae  # Convert to Python scalar

        return metrics

    def calculate(
        self,
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        print_visualizar: Callable = _tensor_to_upper_triangular_matrix,
    ) -> torch.Tensor:
        """Calculate the Mean Absolute Error (MAE) for each dimension.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction results.
            target (torch.Tensor | np.ndarray | Sequence): The target values.

        Returns:
            torch.Tensor: The MAE values for each dimension, converted to an upper triangular matrix.
        """
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.float32)

        assert len(pred.shape) == 2, f"preds and targets are of dimentation {pred.shape} {target.shape}"
        
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}")

        # Calculate MAE for each dimension
        mae = torch.mean(torch.abs(pred - target), dim=0)

        # Convert to upper triangular matrix
        return print_visualizar(mae)


@METRICS.register_module()
class RelativeDelta(MeanAbsoluteError):

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)


    def compute_metrics(self, results: List[dict]):
        metrics = {}

        # Concatenate all predictions and targets
        pred = torch.cat([res['pred'].unsqueeze(0) for res in results])
        target = torch.cat([res['gt'].unsqueeze(0) for res in results])
        delta_5, delta_10, delta_20 = self.calculate(pred, target)
        metrics['\n delta_5'] = delta_5
        metrics['\n delta_10'] = delta_10
        metrics['\n delta_20'] = delta_20

        return metrics

    def calculate(
        self,
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        print_visualizar: Callable = _tensor_to_upper_triangular_matrix,
    ):
        """Calculate the relative delta metrics for each dimension.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction results.
            target (torch.Tensor | np.ndarray | Sequence): The target values.

        Returns:
            Tuple[str, str, str]: The delta_5, delta_10, and delta_20 values for each dimension, 
            converted to upper triangular matrices and formatted as strings.
        """
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.float32)

        assert len(pred.shape) == 2, f"preds and targets are of dimension {pred.shape} {target.shape}"
        
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}")

        # Create a mask for elements where target is larger than 0.01
        mask = torch.abs(target) > 0.01

        # Calculate relative error only for masked elements
        relative_error = torch.where(mask, torch.abs((pred - target) / target) * 100, torch.zeros_like(pred))

        # Calculate thresholds
        a1 = torch.where(mask, (relative_error < 5).float(), torch.zeros_like(pred))
        a2 = torch.where(mask, (relative_error < 10).float(), torch.zeros_like(pred))
        a3 = torch.where(mask, (relative_error < 20).float(), torch.zeros_like(pred))

        # Calculate mean only for masked elements
        a1_mean = a1.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
        a2_mean = a2.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
        a3_mean = a3.sum(dim=0) / mask.sum(dim=0).clamp(min=1)

        return print_visualizar(a1_mean), print_visualizar(a2_mean), print_visualizar(a3_mean)

@METRICS.register_module()
class KLDivergence(MeanAbsoluteError):
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
    def compute_metrics(self, results: List[dict]):
        metrics = {}

        # Concatenate all predictions and targets
        pred = torch.cat([res['pred'].unsqueeze(0) for res in results])
        target = torch.cat([res['gt'].unsqueeze(0) for res in results])
        kl = self.calculate(pred, target)
        metrics['\n kl_divergence'] = kl
        return metrics
    
    def calculate(
        self,
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> torch.Tensor:
        """Calculate the Mean Absolute Error (MAE) for each dimension.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction covariance matrix as upper traingular vector matrix.
            target (torch.Tensor | np.ndarray | Sequence): The target covariance matrix as upper traingular vector matrix..

        """
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.float32)
        
        # iterate over pred-target pairs
        kl_sum = 0
        
        for pr, tr in zip(pred, target):

            pr = matrix_utils.vector_to_symmetric_matrix(pr) + torch.eye(6) * 1e-5  # small constant
            tr = matrix_utils.vector_to_symmetric_matrix(tr) + torch.eye(6) * 1e-5  # small constant
            kl = torch.distributions.kl.kl_divergence(
                torch.distributions.MultivariateNormal(loc=torch.zeros(6), covariance_matrix=tr),
                torch.distributions.MultivariateNormal(loc=torch.zeros(6), covariance_matrix=pr)
            )
            kl_sum += kl
            
        return kl_sum / len(pred)
        
        

        