from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from point_loc.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


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
        >>> from mmpretrain.evaluation import MeanAbsoluteError
        >>> y_pred = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> y_true = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> MeanAbsoluteError.calculate(y_pred, y_true)
        tensor(0.5)
    """
    default_prefix: Optional[str] = 'mae'

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

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        pred = torch.cat([res['pred'] for res in results])
        target = torch.cat([res['gt'] for res in results])

        mae = self.calculate(pred, target)
        metrics['mae'] = mae.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence]
    ) -> torch.Tensor:
        """Calculate the Mean Absolute Error (MAE).

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results.
            target (torch.Tensor | np.ndarray | Sequence): The target values.

        Returns:
            torch.Tensor: The MAE value.
        """
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.float32)
        mae = torch.mean(torch.abs(pred - target))
        return mae