import numpy as np
import pytest
import torch

from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.torch_models.nequip import NequIPModel


class _DummyModule(torch.nn.Module):

    def forward(self, graph):
        batch_size = int(torch.max(graph.graph_index).item()) + 1
        return torch.zeros((batch_size, 1), device=graph.graph_index.device)


def _dummy_loss(outputs, labels, weights):
    del labels, weights
    return outputs[0].sum()


def _make_model(tasks, **kwargs):
    return NequIPModel(tasks=tasks,
                       model=_DummyModule(),
                       loss=_dummy_loss,
                       **kwargs)


def _make_graphs():
    graph1 = GraphData(
        node_features=np.array([[1], [8]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_features=np.ones((2, 3), dtype=np.float32),
        node_pos_features=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                   dtype=np.float32),
        edge_distances=np.ones((2, 1), dtype=np.float32))
    graph2 = GraphData(node_features=np.array([[6], [1], [1]],
                                              dtype=np.float32),
                       edge_index=np.array([[0, 1, 2], [1, 2, 0]],
                                           dtype=np.int64),
                       edge_features=2 * np.ones((3, 3), dtype=np.float32),
                       node_pos_features=np.array(
                           [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]],
                           dtype=np.float32),
                       edge_distances=2 * np.ones((3, 1), dtype=np.float32))
    return [graph1, graph2]


def _make_energy_labels():
    y = np.empty((2, 1), dtype=object)
    y[0, 0] = 1.5
    y[1, 0] = -2.0
    return y


def _make_energy_force_labels():
    y = np.empty((2, 2), dtype=object)
    y[0, 0] = 1.5
    y[0, 1] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32)
    y[1, 0] = -2.0
    y[1, 1] = np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
                       dtype=np.float32)
    return y


def test_prepare_batch_energy_only():
    model = _make_model(tasks=["energy"])
    graphs = _make_graphs()
    y = _make_energy_labels()
    w = np.array([[0.5], [0.25]], dtype=np.float32)

    _, labels, weights = model._prepare_batch(([graphs], [y], [w]))

    assert labels["energy"].shape == (2, 1)
    assert weights["energy"].shape == (2, 1)
    torch.testing.assert_close(labels["energy"], torch.tensor([[1.5], [-2.0]]))
    torch.testing.assert_close(weights["energy"], torch.tensor([[0.5], [0.25]]))


def test_prepare_batch_batches_graphs_and_prepares_energy_force_targets():
    model = _make_model(tasks=["energy", "forces"])
    graphs = _make_graphs()
    y = _make_energy_force_labels()
    w = np.array([[0.5, 0.3], [0.25, 0.4]], dtype=np.float32)

    inputs, labels, weights = model._prepare_batch(([graphs], [y], [w]))

    assert isinstance(inputs, BatchGraphData)
    assert torch.equal(inputs.graph_index, torch.tensor([0, 0, 1, 1, 1]))
    assert inputs.node_pos_features.shape == (5, 3)
    expected_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]])
    assert torch.equal(inputs.edge_index, expected_edge_index)
    assert inputs.edge_distances.shape == (5, 1)
    assert labels["energy"].shape == (2, 1)
    assert labels["forces"].shape == (5, 3)
    assert weights["energy"].shape == (2, 1)
    assert weights["forces"].shape == (5, 1)
    torch.testing.assert_close(labels["energy"], torch.tensor([[1.5], [-2.0]]))
    torch.testing.assert_close(
        labels["forces"],
        torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.2, 0.0, 0.0],
                      [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]))
    torch.testing.assert_close(weights["energy"], torch.tensor([[0.5], [0.25]]))
    torch.testing.assert_close(
        weights["forces"], torch.tensor([[0.3], [0.3], [0.4], [0.4], [0.4]]))


def test_prepare_batch_force_shape_mismatch_raises():
    model = _make_model(tasks=["energy", "forces"])
    graphs = _make_graphs()
    y = _make_energy_force_labels()
    y[1, 1] = np.zeros((2, 3), dtype=np.float32)
    w = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Force labels.*sample 1.*Expected"):
        model._prepare_batch(([graphs], [y], [w]))


@pytest.mark.parametrize("labels, weights, error_match", [
    (_make_energy_labels(), np.ones(
        (2, 2), dtype=np.float32), "Label array has shape.*tasks"),
    (_make_energy_force_labels(), np.ones(
        (2, 1), dtype=np.float32), "Weight array has shape.*tasks"),
])
def test_prepare_batch_task_column_mismatch_raises(labels, weights,
                                                   error_match):
    model = _make_model(tasks=["energy", "forces"])
    graphs = _make_graphs()

    with pytest.raises(ValueError, match=error_match):
        model._prepare_batch(([graphs], [labels], [weights]))


def test_prepare_batch_prediction_path_has_empty_labels_and_weights():
    model = _make_model(tasks=[])
    graphs = _make_graphs()

    inputs, labels, weights = model._prepare_batch(([graphs], [None], [None]))

    assert isinstance(inputs, BatchGraphData)
    assert labels == {}
    assert weights == {}


@pytest.mark.parametrize("kwargs, error_match", [
    ({
        "tasks": ["stress"],
        "model": _DummyModule(),
        "loss": _dummy_loss
    }, "Unsupported NequIPModel tasks"),
    ({
        "tasks": ["energy"],
        "model": None,
        "loss": _dummy_loss
    }, "requires a torch.nn.Module"),
    ({
        "tasks": ["energy"],
        "model": _DummyModule(),
        "loss": None
    }, "requires a loss callable"),
])
def test_constructor_validation(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        NequIPModel(**kwargs)


class _DummyDataset:

    def __init__(self):
        self.pad_batches_seen = []

    def iterbatches(self, batch_size, deterministic, pad_batches):
        del batch_size, deterministic
        self.pad_batches_seen.append(pad_batches)
        graphs = np.asarray(_make_graphs(), dtype=object)
        weights = np.ones((2, 1), dtype=np.float32)
        ids = np.array(["a", "b"], dtype=object)
        yield graphs, _make_energy_labels(), weights, ids


def test_default_generator_does_not_pad_batches():
    model = _make_model(tasks=["energy"], batch_size=4)
    dataset = _DummyDataset()

    batch = next(model.default_generator(dataset, pad_batches=True))

    assert dataset.pad_batches_seen == [False]
    assert len(batch[0][0]) == 2


def test_torch_models_export_path():
    from deepchem.models.torch_models import NequIPModel as ExportedNequIPModel

    assert ExportedNequIPModel is NequIPModel
