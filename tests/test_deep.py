import torch

from spi_audit.deep import SpatialResidualCNN


def test_spatial_cnn_preserves_map_shape():
    model = SpatialResidualCNN(lookback_months=12, width=8)
    inputs = torch.zeros((3, 12, 13, 13))
    assert model(inputs).shape == (3, 1, 13, 13)
