from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

from deepchem.data import Dataset
from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.losses import Loss
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.utils.typing import LossFn


class NequIPModel(TorchModel):
    """Minimal model wrapper for atomistic NequIP-style graph batches.

    This class does not implement the NequIP architecture or native NequIP loss
    yet. It batches variable-size ``GraphData`` samples with ``BatchGraphData``
    and prepares graph-level energy labels and atom-level force labels.
    """

    _SUPPORTED_TASKS = {"energy", "forces"}

    def __init__(self,
                 tasks: Sequence[str],
                 model: torch.nn.Module,
                 loss: Union[Loss, LossFn],
                 batch_size: int = 100,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 **kwargs: Any) -> None:
        if model is None:
            raise ValueError(
                "NequIPModel requires a torch.nn.Module until the native "
                "NequIP architecture is implemented.")
        if loss is None:
            raise ValueError(
                "NequIPModel requires a loss callable until the native NequIP "
                "loss is implemented.")

        self.tasks: List[str] = list(tasks)
        self._validate_tasks(self.tasks)
        self.task_to_index: Dict[str, int] = {
            task: task_index for task_index, task in enumerate(self.tasks)
        }

        output_types = kwargs.pop("output_types", ["prediction"])
        super().__init__(model,
                         loss=loss,
                         output_types=output_types,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         **kwargs)

    def _validate_tasks(self, tasks: Sequence[str]) -> None:
        """Validate the task names supported by this wrapper."""
        unsupported_tasks = [
            task for task in tasks if task not in self._SUPPORTED_TASKS
        ]
        if unsupported_tasks:
            raise ValueError(
                "Unsupported NequIPModel tasks: %s. Supported tasks are %s." %
                (unsupported_tasks, sorted(self._SUPPORTED_TASKS)))
        if len(set(tasks)) != len(tasks):
            raise ValueError("NequIPModel tasks must not contain duplicates.")

    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = True
    ) -> Iterable[Tuple[List[Any], List[Any], List[Any]]]:
        """Create batches without padding graph samples.

        Parameters are the same as ``TorchModel.default_generator()``. The
        ``pad_batches`` argument is ignored and ``False`` is always passed to
        ``Dataset.iterbatches()``.
        """
        del mode, pad_batches
        for _ in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=False):
                del ids_b
                yield ([X_b], [y_b], [w_b])

    def _prepare_batch(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[BatchGraphData, Dict[str, torch.Tensor], Dict[str,
                                                             torch.Tensor]]:
        """Convert a DeepChem batch into batched atomistic graph tensors.

        The input batch must contain ``GraphData`` samples in ``inputs[0]``.
        Labels and weights are returned as dictionaries keyed by task name. If
        labels are missing, empty dictionaries are returned for prediction.
        """
        inputs, labels, weights = batch
        graphs = list(inputs[0])
        batched_graph = BatchGraphData(graphs).numpy_to_torch(self.device)

        if self._is_missing(labels):
            return batched_graph, {}, {}

        label_array = np.asarray(labels[0], dtype=object)
        if label_array.ndim > 1 and label_array.shape[1] < len(self.tasks):
            raise ValueError("Label array has shape %s, but tasks are %s." %
                             (label_array.shape, self.tasks))
        if self._is_missing(weights):
            weight_array = np.ones((label_array.shape[0], len(self.tasks)),
                                   dtype=np.float32)
        else:
            weight_array = np.asarray(weights[0], dtype=np.float32)
        if weight_array.ndim > 1 and weight_array.shape[1] < len(self.tasks):
            raise ValueError("Weight array has shape %s, but tasks are %s." %
                             (weight_array.shape, self.tasks))

        label_tensors: Dict[str, torch.Tensor] = {}
        weight_tensors: Dict[str, torch.Tensor] = {}

        if "energy" in self.task_to_index:
            energy_index = self.task_to_index["energy"]
            if label_array.ndim == 1:
                energy = label_array
            else:
                energy = label_array[:, energy_index]
            energy = np.asarray(energy, dtype=np.float32).reshape(-1, 1)
            label_tensors["energy"] = torch.as_tensor(energy,
                                                      device=self.device)

            if weight_array.ndim == 1:
                energy_weights = weight_array
            else:
                energy_weights = weight_array[:, energy_index]
            energy_weights = np.asarray(energy_weights,
                                        dtype=np.float32).reshape(-1, 1)
            weight_tensors["energy"] = torch.as_tensor(energy_weights,
                                                       device=self.device)

        if "forces" in self.task_to_index:
            forces_index = self.task_to_index["forces"]
            force_labels: List[np.ndarray] = []
            force_weights: List[np.ndarray] = []
            for sample_index, graph in enumerate(graphs):
                if label_array.ndim == 1:
                    forces = label_array[sample_index]
                else:
                    forces = label_array[sample_index, forces_index]
                forces = np.asarray(forces, dtype=np.float32)
                expected_shape = (graph.num_nodes, 3)
                if forces.shape != expected_shape:
                    raise ValueError(
                        "Force labels for sample %d have shape %s. Expected %s."
                        % (sample_index, forces.shape, expected_shape))
                force_labels.append(forces)

                if weight_array.ndim == 1:
                    sample_weight = weight_array[sample_index]
                else:
                    sample_weight = weight_array[sample_index, forces_index]
                force_weights.append(
                    np.full((graph.num_nodes, 1),
                            sample_weight,
                            dtype=np.float32))

            if force_labels:
                force_labels_array = np.concatenate(force_labels, axis=0)
                force_weights_array = np.concatenate(force_weights, axis=0)
            else:
                force_labels_array = np.zeros((0, 3), dtype=np.float32)
                force_weights_array = np.zeros((0, 1), dtype=np.float32)
            label_tensors["forces"] = torch.as_tensor(force_labels_array,
                                                      device=self.device)
            weight_tensors["forces"] = torch.as_tensor(force_weights_array,
                                                       device=self.device)

        return batched_graph, label_tensors, weight_tensors

    def _is_missing(self, values: Any) -> bool:
        return values is None or len(values) == 0 or values[0] is None
