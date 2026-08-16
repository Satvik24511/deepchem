"""Non-periodic NequIP energy backbone."""

import math
from typing import Dict, Tuple, cast

import torch
from torch import nn

from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.torch_models.layers import Fiber, SE3SelfInteraction
from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r


class _BesselBasis(nn.Module):
    """Fixed spherical Bessel-like radial basis."""

    def __init__(self, cutoff: float, num_bessels: int) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("orders", torch.arange(1, num_bessels + 1).float())

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = distances / self.cutoff
        scale = math.sqrt(2.0 / self.cutoff) * math.pi * self.orders
        return scale / self.cutoff * torch.sinc(x * self.orders)


class _PolynomialCutoff(nn.Module):
    """Smooth polynomial cutoff envelope used by NequIP."""

    def __init__(self, cutoff: float, polynomial_order: int = 6) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.polynomial_order = polynomial_order

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = distances / self.cutoff
        p = self.polynomial_order
        envelope = (1 - (p + 1) * (p + 2) / 2 * x**p + p *
                    (p + 2) * x**(p + 1) - p * (p + 1) / 2 * x**(p + 2))
        return torch.where(x < 1, envelope, torch.zeros_like(envelope))


class _NaturalParityPairwiseConv(nn.Module):
    """One natural-parity tensor-product path between two degrees."""

    def __init__(self, degree_in: int, degree_out: int, num_features: int,
                 num_bessels: int) -> None:
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        first_degree = abs(degree_in - degree_out)
        all_degrees = range(first_degree, degree_in + degree_out + 1)
        # Natural parity requires (-1)^l_out = (-1)^(l_in + J).
        self.basis_indices = tuple(
            i for i, angular_degree in enumerate(all_degrees)
            if (degree_in + degree_out + angular_degree) % 2 == 0)
        num_weights = (num_features * num_features * len(self.basis_indices))
        self.radial = nn.Sequential(nn.Linear(num_bessels, num_features),
                                    nn.SiLU(),
                                    nn.Linear(num_features, num_weights))
        self.num_features = num_features

    def forward(self, radial_features: torch.Tensor, envelope: torch.Tensor,
                basis: Dict[str, torch.Tensor]) -> torch.Tensor:
        basis_tensor = basis[f"{self.degree_in},{self.degree_out}"]
        basis_tensor = basis_tensor[..., list(self.basis_indices)]
        weights = self.radial(radial_features) * envelope
        weights = weights.view(-1, self.num_features, 1, self.num_features, 1,
                               len(self.basis_indices))
        return (weights * basis_tensor).sum(-1).reshape(
            -1, self.num_features * (2 * self.degree_out + 1),
            self.num_features * (2 * self.degree_in + 1))


class _NaturalParityConvolution(nn.Module):
    """Natural-parity equivariant convolution with torch aggregation."""

    def __init__(self, f_in: Fiber, f_out: Fiber, num_features: int,
                 num_bessels: int) -> None:
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.pairwise = nn.ModuleDict({
            f"{degree_in},{degree_out}":
            _NaturalParityPairwiseConv(degree_in, degree_out, num_features,
                                       num_bessels)
            for degree_in in f_in.degrees for degree_out in f_out.degrees
        })

    def forward(self, features: Dict[str, torch.Tensor],
                radial_features: torch.Tensor, envelope: torch.Tensor,
                basis: Dict[str, torch.Tensor], source: torch.Tensor,
                destination: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_nodes = next(iter(features.values())).shape[0]
        output = {
            str(degree): next(iter(features.values())).new_zeros(
                num_nodes, self.f_out.structure_dict[degree], 2 * degree + 1)
            for degree in self.f_out.degrees
        }
        if source.numel() == 0:
            return output

        for degree_out in self.f_out.degrees:
            for degree_in in self.f_in.degrees:
                kernel = self.pairwise[f"{degree_in},{degree_out}"](
                    radial_features, envelope, basis)
                source_features = features[str(degree_in)][source].reshape(
                    source.shape[0], -1, 1)
                messages = torch.bmm(kernel, source_features).reshape(
                    source.shape[0], self.f_out.structure_dict[degree_out],
                    2 * degree_out + 1)
                output[str(degree_out)] = output[str(degree_out)].index_add(
                    0, destination, messages)
        return output


class _NequIPGate(nn.Module):
    """Scalar activation and invariant scalar gates for higher degrees."""

    def __init__(self, num_features: int, l_max: int) -> None:
        super().__init__()
        self.l_max = l_max
        self.gates = (nn.Linear(num_features, num_features *
                                l_max) if l_max > 0 else None)

    def forward(self, features: Dict[str,
                                     torch.Tensor]) -> Dict[str, torch.Tensor]:
        scalars = features["0"]
        output = {"0": torch.nn.functional.silu(scalars)}
        if self.gates is None:
            return output
        gates = torch.sigmoid(self.gates(scalars.squeeze(-1)))
        gates = gates.view(scalars.shape[0], self.l_max, -1)
        for degree in range(1, self.l_max + 1):
            output[str(degree)] = (features[str(degree)] *
                                   gates[:, degree - 1, :, None])
        return output


class _NequIPInteractionBlock(nn.Module):
    """Tensor-product convolution, self interaction, residual, and gate."""

    def __init__(self, f_in: Fiber, f_out: Fiber, num_features: int,
                 num_bessels: int, l_max: int, residual: bool) -> None:
        super().__init__()
        self.f_out = f_out
        self.convolution = _NaturalParityConvolution(f_in, f_out, num_features,
                                                     num_bessels)
        self.self_interaction = SE3SelfInteraction(f_in, f_in)
        self.gate = _NequIPGate(num_features, l_max)
        self.residual = residual

    def forward(self, features: Dict[str, torch.Tensor],
                radial_features: torch.Tensor, envelope: torch.Tensor,
                basis: Dict[str, torch.Tensor], source: torch.Tensor,
                destination: torch.Tensor) -> Dict[str, torch.Tensor]:
        messages = self.convolution(features, radial_features, envelope, basis,
                                    source, destination)
        local = self.self_interaction(features)
        for degree in messages:
            if degree in local:
                messages[degree] = messages[degree] + local[degree]
            if self.residual:
                messages[degree] = messages[degree] + features[degree]
        return self.gate(messages)


class NequIP(nn.Module):
    """Natural-parity NequIP energy backbone.

    NequIP is an equivariant interatomic potential based on learned radial
    functions and spherical-harmonic tensor products.  This implementation is
    the non-periodic energy backbone only.

    Parameters
    ----------
    cutoff : float
        Distance cutoff.  Input topology must contain the desired directed
        neighbor edges and every edge must be shorter than this value.
    num_layers : int, default 4
        Number of equivariant interaction blocks.
    l_max : int, default 1
        Maximum angular degree in the natural-parity hidden representation.
    num_features : int, default 32
        Multiplicity of every angular degree.
    num_bessels : int, default 8
        Number of fixed Bessel radial basis functions.

    Notes
    -----
    ``forward`` requires a torch-converted
    :class:`deepchem.feat.graph_data.BatchGraphData` with atomic numbers in
    ``node_features`` of shape ``(N, 1)``, Cartesian positions in
    ``node_pos_features`` of shape ``(N, 3)``, directed topology in
    ``edge_index`` of shape ``(2, E)``, and membership in ``graph_index``.
    It returns total energies with shape ``(number_of_graphs, 1)``.  Forces can
    be obtained by differentiating the summed energy with respect to
    ``node_pos_features``.  PBC, stress, and the ``TorchModel`` training
    wrapper are not included.

    References
    ----------
    .. [1] Batzner, S. et al. "E(3)-Equivariant Graph Neural Networks for
       Data-Efficient and Accurate Interatomic Potentials." Nature
       Communications 13, 2453 (2022).
    """

    def __init__(self,
                 cutoff: float,
                 num_layers: int = 4,
                 l_max: int = 1,
                 num_features: int = 32,
                 num_bessels: int = 8) -> None:
        super().__init__()
        if (not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or
                not math.isfinite(cutoff) or cutoff <= 0):
            raise ValueError("cutoff must be a finite positive number")
        for name, value in (("num_layers", num_layers), ("l_max", l_max),
                            ("num_features", num_features), ("num_bessels",
                                                             num_bessels)):
            minimum = 0 if name == "l_max" else 1
            if not isinstance(value, int) or isinstance(
                    value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        self.cutoff = float(cutoff)
        self.l_max = l_max
        scalar_fiber = Fiber(dictionary={0: num_features})
        hidden_fiber = Fiber(
            dictionary={degree: num_features for degree in range(l_max + 1)})

        self.atomic_embedding = nn.Embedding(119, num_features)
        self.bessel_basis = _BesselBasis(self.cutoff, num_bessels)
        self.cutoff_envelope = _PolynomialCutoff(self.cutoff)
        self.interactions = nn.ModuleList([
            _NequIPInteractionBlock(
                scalar_fiber if layer == 0 else hidden_fiber,
                hidden_fiber,
                num_features,
                num_bessels,
                l_max,
                residual=layer > 0) for layer in range(num_layers)
        ])
        self.readout = nn.Sequential(nn.Linear(num_features, num_features),
                                     nn.SiLU(), nn.Linear(num_features, 1))

    def _validate_graph(
        self, graph: BatchGraphData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(graph, BatchGraphData):
            raise TypeError("graph must be a BatchGraphData")
        required = ("node_features", "node_pos_features", "edge_index",
                    "graph_index")
        for name in required:
            if getattr(graph, name, None) is None:
                raise ValueError(f"graph.{name} is required")
            if not isinstance(getattr(graph, name), torch.Tensor):
                raise TypeError(
                    f"graph.{name} must be a torch tensor; call numpy_to_torch()"
                )

        node_features = cast(torch.Tensor, graph.node_features)
        positions = cast(torch.Tensor, graph.node_pos_features)
        edge_index = cast(torch.Tensor, graph.edge_index)
        graph_index = cast(torch.Tensor, graph.graph_index)
        num_nodes = node_features.shape[0]
        if node_features.ndim != 2 or node_features.shape[1] != 1:
            raise ValueError("node_features must have shape (N, 1)")
        if positions.ndim != 2 or positions.shape != (num_nodes, 3):
            raise ValueError("node_pos_features must have shape (N, 3)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if graph_index.ndim != 1 or graph_index.shape[0] != num_nodes:
            raise ValueError("graph_index must have shape (N,)")
        if num_nodes == 0:
            raise ValueError("graph must contain at least one atom")
        tensors = (node_features, positions, edge_index, graph_index)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all graph tensors must be on the same device")
        if node_features.device != self.atomic_embedding.weight.device:
            raise ValueError("graph and model must be on the same device")
        if not node_features.is_floating_point():
            raise TypeError("node_features must be floating point")
        if not positions.is_floating_point():
            raise TypeError("node_pos_features must be floating point")
        if not torch.isfinite(positions).all().item():
            raise ValueError("node_pos_features must be finite")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if graph_index.dtype != torch.long:
            raise TypeError("graph_index must have dtype torch.long")

        rounded_numbers = node_features.round()
        if not torch.isfinite(node_features).all().item() or not torch.equal(
                node_features, rounded_numbers):
            raise ValueError(
                "node_features must contain integral atomic numbers")
        atomic_numbers = rounded_numbers[:, 0].long()
        if (atomic_numbers < 1).any().item() or (atomic_numbers >
                                                 118).any().item():
            raise ValueError("atomic numbers must be in the range [1, 118]")
        if edge_index.numel() > 0:
            if (edge_index < 0).any().item() or (edge_index >=
                                                 num_nodes).any().item():
                raise ValueError("edge_index contains an invalid node index")
            if not torch.equal(graph_index[edge_index[0]],
                               graph_index[edge_index[1]]):
                raise ValueError("edges may not connect different graphs")
        if (graph_index < 0).any().item():
            raise ValueError("graph_index must contain zero-based identifiers")
        identifiers = torch.unique(graph_index)
        expected = torch.arange(identifiers.shape[0], device=graph_index.device)
        if not torch.equal(identifiers, expected):
            raise ValueError(
                "graph_index identifiers must be contiguous from zero")
        return atomic_numbers, positions, edge_index, graph_index

    def forward(self, graph: BatchGraphData) -> torch.Tensor:
        """Predict one total energy for each structure in ``graph``."""
        atomic_numbers, positions, edge_index, graph_index = self._validate_graph(
            graph)
        source, destination = edge_index
        edge_vectors = positions[destination] - positions[source]
        distances = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
        if (distances == 0).any().item():
            raise ValueError("input edges must connect distinct positions")
        if (distances >= self.cutoff).any().item():
            raise ValueError("all input edges must be shorter than cutoff")

        basis: Dict[str, torch.Tensor] = {}
        if source.numel() > 0:
            try:
                import dgl
            except ModuleNotFoundError as error:
                raise ImportError(
                    "NequIP requires DGL for its equivariant angular basis"
                ) from error
            dgl_graph = dgl.graph((source, destination),
                                  num_nodes=positions.shape[0],
                                  device=positions.device)
            dgl_graph.edata["edge_attr"] = edge_vectors
            basis, _ = get_equivariant_basis_and_r(
                dgl_graph,
                self.l_max,
                compute_gradients=positions.requires_grad)

        radial_features = self.bessel_basis(distances)
        envelope = self.cutoff_envelope(distances)
        features = {"0": self.atomic_embedding(atomic_numbers).unsqueeze(-1)}
        for interaction in self.interactions:
            features = interaction(features, radial_features, envelope, basis,
                                   source, destination)

        atomic_energies = self.readout(features["0"].squeeze(-1))
        num_graphs = int(graph_index.max().item()) + 1
        return atomic_energies.new_zeros(
            (num_graphs, 1)).index_add(0, graph_index, atomic_energies)
