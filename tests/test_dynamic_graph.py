"""Tests for P5-A sparse hidden-representation graph refinement."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from src.graph_refinement import (
    apply_node_confidence_gate,
    blend_sparse_adjs,
    build_semantic_knn_adj,
    refine_graphs,
    rank_normalize_confidence,
)
from src.model import EDLMultiViewGCN


def sparse_identity(size):
    indices = torch.arange(size).repeat(2, 1)
    return torch.sparse_coo_tensor(
        indices, torch.ones(size), (size, size)
    ).coalesce()


class SemanticGraphTests(unittest.TestCase):
    def test_rank_normalization_is_monotonic_and_class_scale_independent(self):
        raw = torch.tensor([0.001, 0.01, 0.1, 0.5])
        normalized = rank_normalize_confidence(raw, confidence_floor=0.2)
        self.assertAlmostEqual(normalized.min().item(), 0.2, places=6)
        self.assertAlmostEqual(normalized.max().item(), 1.0, places=6)
        self.assertTrue(torch.all(normalized[1:] > normalized[:-1]))
        self.assertAlmostEqual(normalized.mean().item(), 0.6, places=6)

    def test_confidence_gate_scales_edges_by_endpoint_confidence(self):
        adjacency = sp.csr_matrix(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        )
        gated = apply_node_confidence_gate(
            adjacency, torch.tensor([1.0, 0.25]), confidence_floor=0.0
        )
        expected = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(gated.toarray(), expected)

    def test_confidence_floor_prevents_zero_edges(self):
        adjacency = sp.csr_matrix(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        )
        gated = apply_node_confidence_gate(
            adjacency, torch.zeros(2), confidence_floor=0.2
        )
        np.testing.assert_allclose(
            gated.toarray(), np.array([[0.0, 0.2], [0.2, 0.0]])
        )

    def test_semantic_graph_is_sparse_symmetric_and_finite(self):
        embedding = torch.rand(20, 6)
        graph = build_semantic_knn_adj(embedding, k=3)
        self.assertTrue(graph.is_sparse)
        self.assertEqual(graph.shape, (20, 20))
        self.assertTrue(torch.isfinite(graph.values()).all())
        torch.testing.assert_close(graph.to_dense(), graph.to_dense().T)
        self.assertLessEqual(graph._nnz(), 20 * (2 * 3 + 1))

    def test_invalid_k_is_rejected(self):
        embedding = torch.rand(5, 2)
        with self.assertRaises(ValueError):
            build_semantic_knn_adj(embedding, k=0)
        with self.assertRaises(ValueError):
            build_semantic_knn_adj(embedding, k=5)


class GraphBlendTests(unittest.TestCase):
    def test_base_weight_one_preserves_base_graph(self):
        base = sparse_identity(4)
        semantic = build_semantic_knn_adj(torch.rand(4, 3), k=1)
        mixed = blend_sparse_adjs(base, semantic, base_weight=1.0)
        torch.testing.assert_close(mixed.to_dense(), base.to_dense())

    def test_convex_blend_matches_dense_reference(self):
        base = sparse_identity(6)
        semantic = build_semantic_knn_adj(torch.rand(6, 3), k=2)
        mixed = blend_sparse_adjs(base, semantic, base_weight=0.75)
        expected = 0.75 * base.to_dense() + 0.25 * semantic.to_dense()
        torch.testing.assert_close(mixed.to_dense(), expected)

    def test_refinement_keeps_one_sparse_graph_per_view(self):
        model = EDLMultiViewGCN([4, 5], num_classes=3, hidden=6, dropout=0.0)
        views = [torch.rand(12, 4), torch.rand(12, 5)]
        base = [sparse_identity(12), sparse_identity(12)]
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(views, base)
        refined, stats = refine_graphs(embeddings, base, k=2, base_weight=0.8)
        self.assertEqual([item.shape for item in embeddings], [(12, 6), (12, 6)])
        self.assertEqual(len(refined), 2)
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(graph.is_sparse for graph in refined))

    def test_refinement_reports_per_view_confidence(self):
        embeddings = [torch.rand(10, 4), torch.rand(10, 4)]
        base = [sparse_identity(10), sparse_identity(10)]
        confidences = [torch.full((10,), 0.4), torch.full((10,), 0.7)]
        refined, stats = refine_graphs(
            embeddings,
            base,
            k=2,
            node_confidences=confidences,
            confidence_floor=0.2,
        )
        self.assertEqual(len(refined), 2)
        self.assertAlmostEqual(stats[0]["mean_confidence"], 0.4, places=6)
        self.assertAlmostEqual(stats[1]["mean_confidence"], 0.7, places=6)



if __name__ == "__main__":
    unittest.main()
