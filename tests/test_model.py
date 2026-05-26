"""Architecture tests for the CNN."""

import torch

from models.cnn import HandwritingCNN


def test_forward_pass_shape():
    model = HandwritingCNN(num_classes=62)
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 62)


def test_param_count_in_range():
    model = HandwritingCNN(num_classes=62)
    n = model.count_params()
    # Sanity: should be in the low hundreds of thousands, not millions
    assert 100_000 < n < 2_000_000


def test_grad_flows():
    model = HandwritingCNN(num_classes=62)
    x = torch.randn(4, 1, 28, 28)
    target = torch.randint(0, 62, (4,))
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_different_num_classes():
    for nc in [10, 26, 62, 100]:
        model = HandwritingCNN(num_classes=nc)
        out = model(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, nc)
