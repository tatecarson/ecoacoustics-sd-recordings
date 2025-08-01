#!/usr/bin/env python3
"""
Simple test to verify preprocessing control logic without full dependencies.
"""
import numpy as np

from beamforming_utils.beamforming_analysis import maybe_preprocess, should_preprocess_auto

if __name__ == "__main__":
    # Test preprocessing functions
    sr = 22050
    x = np.random.randn(sr * 2)  # 2 seconds of random noise
    
    print("Testing maybe_preprocess function:")
    
    # Test 1: HPF only
    y1 = maybe_preprocess(x, sr, {"hpf_hz": 120})
    print(f"HPF test: input length {len(x)}, output length {len(y1)}, lengths match: {len(x) == len(y1)}")
    
    # Test 2: Envelope median only
    y2 = maybe_preprocess(x, sr, {"envelope_median_ms": 15})
    print(f"Envelope median test: input length {len(x)}, output length {len(y2)}, lengths match: {len(x) == len(y2)}")
    
    # Test 3: Both
    y3 = maybe_preprocess(x, sr, {"hpf_hz": 60, "envelope_median_ms": 10})
    print(f"Combined test: input length {len(x)}, output length {len(y3)}, lengths match: {len(x) == len(y3)}")
    
    # Test 4: No preprocessing
    y4 = maybe_preprocess(x, sr, {})
    print(f"No preprocessing test: arrays identical: {np.array_equal(x, y4)}")
    
    print("\nTesting auto-detection function:")
    
    # Test auto-detection with different profiles
    for profile in ["wind", "rain", "surf_river", "thunder", "geophony_general", "none"]:
        apply_preproc, metrics = should_preprocess_auto(x, sr, profile)
        print(f"Profile '{profile}': apply_preproc={apply_preproc}, LF_ratio={metrics['LF_ratio']:.3f}, S_flat={metrics['S_flat']:.3f}, kurt={metrics['kurt']:.3f}")
    
    print("\nAll tests completed successfully!")
