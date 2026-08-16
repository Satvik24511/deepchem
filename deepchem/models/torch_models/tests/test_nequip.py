"""Tests for the NequIP energy backbone."""

import copy
import importlib.util

import numpy as np
import pytest
import torch

from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.torch_models.nequip import NequIP

pytestmark = pytest.mark.torch


def _graph(atomic_numbers, positions, edges):
    positions = np.asarray(positions, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    vectors = (positions[edges[1]] -
               positions[edges[0]] if edges.shape[1] else np.empty(
                   (0, 3), dtype=np.float32))
    distances = np.linalg.norm(vectors, axis=1, keepdims=True)
    return GraphData(np.asarray(atomic_numbers,
                                dtype=np.float32).reshape(-1, 1),
                     edges,
                     vectors,
                     positions,
                     edge_distances=distances)


def _two_graphs():
    graph_a = _graph([1, 8], [[0, 0, 0], [0.9, 0.2, 0.1]], [[0, 1], [1, 0]])
    graph_b = _graph([6, 1, 1], [[0, 0, 0], [0.7, -0.1, 0.2], [-0.2, 0.8, 0.1]],
                     [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    return graph_a, graph_b


def _batch(graphs, device="cpu", requires_grad=False):
    batch = BatchGraphData(graphs).numpy_to_torch(device)
    batch.node_pos_features.requires_grad_(requires_grad)
    return batch


def _model(device="cpu"):
    torch.manual_seed(11)
    return NequIP(cutoff=2.0,
                  num_layers=2,
                  l_max=1,
                  num_features=4,
                  num_bessels=3).to(device).eval()


def _require_dgl():
    pytest.importorskip("dgl")


@pytest.mark.parametrize("kwargs", [{
    "cutoff": 0
}, {
    "cutoff": True
}, {
    "cutoff": float("inf")
}, {
    "cutoff": 2,
    "num_layers": 0
}, {
    "cutoff": 2,
    "l_max": -1
}, {
    "cutoff": 2,
    "num_features": 0
}, {
    "cutoff": 2,
    "num_bessels": 0
}])
def test_nequip_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        NequIP(**kwargs)


def test_nequip_input_validation():
    model = _model()
    graph_a, _ = _two_graphs()
    valid = _batch([graph_a])

    invalid_cases = []
    missing_positions = copy.deepcopy(valid)
    missing_positions.node_pos_features = None
    invalid_cases.append(missing_positions)
    wrong_features = copy.deepcopy(valid)
    wrong_features.node_features = torch.ones(2, 2)
    invalid_cases.append(wrong_features)
    non_integral = copy.deepcopy(valid)
    non_integral.node_features[0, 0] = 1.5
    invalid_cases.append(non_integral)
    invalid_atomic_number = copy.deepcopy(valid)
    invalid_atomic_number.node_features[0, 0] = 0
    invalid_cases.append(invalid_atomic_number)
    wrong_positions = copy.deepcopy(valid)
    wrong_positions.node_pos_features = torch.zeros(2, 2)
    invalid_cases.append(wrong_positions)
    wrong_edges = copy.deepcopy(valid)
    wrong_edges.edge_index = torch.zeros(3, 2, dtype=torch.long)
    invalid_cases.append(wrong_edges)
    invalid_node_index = copy.deepcopy(valid)
    invalid_node_index.edge_index[0, 0] = 2
    invalid_cases.append(invalid_node_index)
    invalid_graph_index = copy.deepcopy(valid)
    invalid_graph_index.graph_index[:] = 1
    invalid_cases.append(invalid_graph_index)

    for graph in invalid_cases:
        with pytest.raises((TypeError, ValueError)):
            model(graph)


def test_nequip_missing_dgl_error():
    if importlib.util.find_spec("dgl") is not None:
        pytest.skip("DGL is installed")
    graph_a, _ = _two_graphs()
    with pytest.raises(ImportError, match="requires DGL"):
        _model()(_batch([graph_a]))


def test_nequip_different_sized_batch():
    _require_dgl()
    batch = _batch(_two_graphs())
    output = _model()(batch)
    assert batch.graph_index.tolist() == [0, 0, 1, 1, 1]
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_nequip_batch_isolation():
    _require_dgl()
    graph_a, graph_b = _two_graphs()
    model = _model()
    together = model(_batch([graph_a, graph_b]))
    separate = torch.cat([model(_batch([graph_a])), model(_batch([graph_b]))])
    assert torch.allclose(together, separate, atol=1e-6, rtol=1e-6)


def test_nequip_ignores_static_geometry():
    _require_dgl()
    graph_a, graph_b = _two_graphs()
    original = _batch([graph_a, graph_b])
    corrupted = _batch([graph_a, graph_b])
    corrupted.edge_features.fill_(1234)
    corrupted.edge_distances.fill_(-5678)
    model = _model()
    assert torch.equal(model(original), model(corrupted))


@pytest.mark.parametrize("transform", [
    torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
    torch.diag(torch.tensor([-1., 1., 1.])),
])
def test_nequip_orthogonal_invariance(transform):
    _require_dgl()
    graph_a, graph_b = _two_graphs()
    original = _batch([graph_a, graph_b])
    transformed = _batch([graph_a, graph_b])
    transformed.node_pos_features = transformed.node_pos_features @ transform.T
    model = _model()
    assert torch.allclose(model(original),
                          model(transformed),
                          atol=2e-5,
                          rtol=2e-5)


def test_nequip_translation_invariance():
    _require_dgl()
    graph_a, graph_b = _two_graphs()
    original = _batch([graph_a, graph_b])
    translated = _batch([graph_a, graph_b])
    shifts = torch.tensor([[1.2, -0.4, 0.7], [-0.8, 0.3, 1.1]])
    translated.node_pos_features += shifts[translated.graph_index]
    model = _model()
    assert torch.allclose(model(original),
                          model(translated),
                          atol=1e-6,
                          rtol=1e-6)


@pytest.mark.parametrize("transform", [
    torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
    torch.diag(torch.tensor([-1., 1., 1.])),
])
def test_nequip_force_orthogonal_equivariance(transform):
    _require_dgl()
    graph_a, graph_b = _two_graphs()
    original = _batch([graph_a, graph_b], requires_grad=True)
    transformed = _batch([graph_a, graph_b])
    transformed.node_pos_features = (
        transformed.node_pos_features @ transform.T).detach().requires_grad_()
    model = _model()
    energy = model(original)
    force = -torch.autograd.grad(energy.sum(), original.node_pos_features)[0]
    transformed_energy = model(transformed)
    transformed_force = -torch.autograd.grad(transformed_energy.sum(),
                                             transformed.node_pos_features)[0]
    assert torch.allclose(energy, transformed_energy, atol=2e-5, rtol=2e-5)
    assert torch.allclose(transformed_force,
                          force @ transform.T,
                          atol=3e-5,
                          rtol=3e-5)


def test_nequip_second_order_autograd():
    _require_dgl()
    graph = _batch(_two_graphs(), requires_grad=True)
    model = _model()
    energy = model(graph)
    forces = -torch.autograd.grad(
        energy.sum(), graph.node_pos_features, create_graph=True)[0]
    force_loss = (forces**2).mean()
    force_loss.backward()
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(
        torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in parameter_gradients)


def test_nequip_single_atom():
    graph = _graph([2], [[0, 0, 0]], np.empty((2, 0), dtype=np.int64))
    output = _model()(_batch([graph]))
    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nequip_cuda_device_and_gradients():
    _require_dgl()
    graph = _batch(_two_graphs(), device="cuda", requires_grad=True)
    model = _model("cuda")
    output = model(graph)
    forces = -torch.autograd.grad(output.sum(), graph.node_pos_features)[0]
    assert output.device.type == "cuda"
    assert forces.device.type == "cuda"
