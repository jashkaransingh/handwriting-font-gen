"""Sanity tests for the augmentation pipeline."""

import numpy as np
import pytest

from data.augment import (AugmentationPipeline, elastic_distortion,
                          gaussian_noise, morphological, random_rotate,
                          random_shear, random_shift, stroke_thickness)


@pytest.fixture
def sample_image():
    img = np.zeros((28, 28), dtype=np.uint8)
    img[8:20, 8:20] = 255
    return img


def test_rotate_preserves_shape(sample_image):
    out = random_rotate(sample_image, max_angle=15.0,
                        rng=np.random.default_rng(0))
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_shift_preserves_shape(sample_image):
    out = random_shift(sample_image, max_pixels=3,
                       rng=np.random.default_rng(0))
    assert out.shape == sample_image.shape


def test_shear_preserves_shape(sample_image):
    out = random_shear(sample_image, max_shear=0.3,
                       rng=np.random.default_rng(0))
    assert out.shape == sample_image.shape


def test_noise_in_valid_range(sample_image):
    out = gaussian_noise(sample_image, std=10.0,
                         rng=np.random.default_rng(0))
    assert out.min() >= 0
    assert out.max() <= 255


def test_morphological_returns_uint8(sample_image):
    out = morphological(sample_image, rng=np.random.default_rng(0))
    assert out.dtype == np.uint8
    assert out.shape == sample_image.shape


def test_elastic_distortion_preserves_shape(sample_image):
    out = elastic_distortion(sample_image, alpha=8.0, sigma=4.0,
                             rng=np.random.default_rng(0))
    assert out.shape == sample_image.shape


def test_stroke_thickness_returns_uint8(sample_image):
    out = stroke_thickness(sample_image, rng=np.random.default_rng(0))
    assert out.dtype == np.uint8


def test_pipeline_medium_severity(sample_image):
    pipe = AugmentationPipeline(severity="medium", seed=0)
    out = pipe(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_pipeline_severities_compose():
    img = np.zeros((28, 28), dtype=np.uint8)
    img[10:18, 10:18] = 255
    for severity in ["light", "medium", "heavy"]:
        pipe = AugmentationPipeline(severity=severity, seed=42)
        out = pipe(img)
        assert out.shape == img.shape


def test_pipeline_determinism():
    """Same seed produces same output."""
    img = np.zeros((28, 28), dtype=np.uint8)
    img[8:20, 8:20] = 255
    pipe1 = AugmentationPipeline(severity="medium", seed=123)
    pipe2 = AugmentationPipeline(severity="medium", seed=123)
    out1 = pipe1(img.copy())
    out2 = pipe2(img.copy())
    np.testing.assert_array_equal(out1, out2)


def test_pipeline_produces_variation():
    """Same input through pipeline twice should produce different outputs."""
    img = np.zeros((28, 28), dtype=np.uint8)
    img[8:20, 8:20] = 255
    pipe = AugmentationPipeline(severity="medium", seed=42)
    a = pipe(img.copy())
    b = pipe(img.copy())
    # Not equal because the rng state advances
    assert not np.array_equal(a, b)
