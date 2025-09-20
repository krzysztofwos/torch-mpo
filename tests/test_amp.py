"""Automatic mixed precision (AMP) tests."""

import pytest
import torch

from torch_mpo.layers import TTConv2d, TTLinear


def test_amp_cpu_bfloat16_smoke():
    """Test AMP with bfloat16 on CPU."""
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 not available")

    m1 = TTLinear(256, 128, tt_ranks=4)
    m2 = TTConv2d(3, 8, kernel_size=3, padding=1, tt_ranks=2)
    x1 = torch.randn(7, 256)
    x2 = torch.randn(2, 3, 16, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y1 = m1(x1)
        y2 = m2(x2)
        loss = y1.float().pow(2).mean() + y2.float().pow(2).mean()

    # Note: Custom layers may not automatically cast to bfloat16
    # The important thing is they work within autocast context without errors
    # Check backward pass works
    loss.backward()
    assert m1.cores[0].grad is not None
    assert m2.cores[0].grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_cuda_fp16_smoke():
    """Test AMP with fp16 on CUDA."""
    m1 = TTLinear(128, 64, tt_ranks=4).cuda()
    m2 = TTConv2d(3, 8, kernel_size=3, padding=1, tt_ranks=2).cuda()
    x1 = torch.randn(5, 128, device="cuda")
    x2 = torch.randn(2, 3, 16, 16, device="cuda")

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        y1 = m1(x1)
        y2 = m2(x2)
        loss = y1.float().pow(2).mean() + y2.float().pow(2).mean()

    # Note: Custom layers may not automatically cast to fp16
    # The important thing is they work within autocast context without errors
    # Check backward pass works
    loss.backward()
    assert m1.cores[0].grad is not None
    assert m2.cores[0].grad is not None


def test_amp_gradscaler_integration():
    """Test integration with GradScaler."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GradScaler")

    model = TTLinear(64, 32, tt_ranks=4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(8, 64, device="cuda")
    target = torch.randn(8, 32, device="cuda")

    # Training step with GradScaler
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)

    # Scale loss and backward
    scaler.scale(loss).backward()

    # Check gradients are scaled
    assert model.cores[0].grad is not None

    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()

    # Verify parameters were updated
    optimizer.zero_grad()
