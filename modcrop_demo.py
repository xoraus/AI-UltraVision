#!/usr/bin/env python3
"""
Demonstration of the modcrop function and why it's necessary for SRCNN.

This script helps understand:
1. How modcrop works mathematically
2. Why cropping is needed for SRCNN architecture
3. The impact of different scale factors
4. Visual comparison of before/after dimensions

Run this script to see modcrop in action!
"""

import numpy as np

def modcrop(img, scale):
    """
    Crop image to make dimensions divisible by scale factor.
    
    Args:
        img (numpy.ndarray): Input image with shape (H, W, C)
        scale (int): Scale factor (typically 3 for SRCNN)
        
    Returns:
        numpy.ndarray: Cropped image with dimensions divisible by scale
    """
    tmpsz = img.shape
    sz = tmpsz[0:2]  # Get height and width
    sz = sz - np.mod(sz, scale)  # Remove remainder when divided by scale
    img = img[0:sz[0], 0:sz[1]]  # Crop to new dimensions
    return img

def analyze_modcrop(height, width, scale):
    """Analyze what modcrop will do to given dimensions."""
    print(f"\n--- Analysis for {height}x{width} image with scale factor {scale} ---")
    
    # Calculate remainders
    h_remainder = height % scale
    w_remainder = width % scale
    
    # Calculate new dimensions
    new_height = height - h_remainder
    new_width = width - w_remainder
    
    # Calculate pixels lost
    h_lost = h_remainder
    w_lost = w_remainder
    total_pixels_before = height * width
    total_pixels_after = new_height * new_width
    pixels_lost = total_pixels_before - total_pixels_after
    percent_lost = (pixels_lost / total_pixels_before) * 100
    
    print(f"Original dimensions: {height} x {width} = {total_pixels_before:,} pixels")
    print(f"Scale factor: {scale}")
    print(f"Height: {height} % {scale} = {h_remainder} remainder → crop {h_lost} pixels")
    print(f"Width: {width} % {scale} = {w_remainder} remainder → crop {w_lost} pixels")
    print(f"New dimensions: {new_height} x {new_width} = {total_pixels_after:,} pixels")
    print(f"Pixels lost: {pixels_lost:,} ({percent_lost:.2f}% of image)")
    
    # Check if dimensions are now divisible
    print(f"Verification: {new_height} % {scale} = {new_height % scale}, {new_width} % {scale} = {new_width % scale}")
    print("✓ Both dimensions now divisible by scale factor!" if (new_height % scale == 0 and new_width % scale == 0) else "✗ Error in calculation!")

def demonstrate_srcnn_constraints():
    """Demonstrate why SRCNN needs specific dimension constraints."""
    print("\n" + "="*60)
    print("SRCNN ARCHITECTURE CONSTRAINTS DEMONSTRATION")
    print("="*60)
    
    print("\nSRCNN Network Architecture:")
    print("1. Patch Extraction: 9×9 conv, 128 filters, 'valid' padding → reduces by 8 pixels")
    print("2. Non-linear Mapping: 3×3 conv, 64 filters, 'same' padding → no size change")  
    print("3. Reconstruction: 5×5 conv, 1 filter, 'valid' padding → reduces by 4 pixels")
    print("Total size reduction: 8 + 0 + 4 = 12 pixels per dimension")
    
    print("\nExample with a 100×100 input image:")
    input_size = 100
    after_layer1 = input_size - 8  # 9x9 valid padding
    after_layer2 = after_layer1    # 3x3 same padding  
    after_layer3 = after_layer2 - 4 # 5x5 valid padding
    
    print(f"Input: {input_size}×{input_size}")
    print(f"After layer 1 (9×9 valid): {after_layer1}×{after_layer1}")
    print(f"After layer 2 (3×3 same): {after_layer2}×{after_layer2}")
    print(f"After layer 3 (5×5 valid): {after_layer3}×{after_layer3}")
    print(f"Final output: {after_layer3}×{after_layer3}")
    
    print(f"\nThis is why the network expects specific input/output size relationships!")

def main():
    """Main demonstration function."""
    print("MODCROP FUNCTION DEMONSTRATION")
    print("="*50)
    print("Understanding why we crop images before SRCNN processing")
    
    # Test cases from common image dimensions
    test_cases = [
        (176, 197, 3),  # User's original example
        (512, 512, 3),  # Square power-of-2 image
        (1920, 1080, 3), # HD video frame
        (224, 224, 3),  # Common ML input size
        (100, 150, 2),  # Small test image with scale 2
    ]
    
    for height, width, scale in test_cases:
        analyze_modcrop(height, width, scale)
    
    # Demonstrate actual modcrop function
    print("\n" + "="*60)
    print("PRACTICAL MODCROP DEMONSTRATION")
    print("="*60)
    
    # Create test images and apply modcrop
    print("\nTesting modcrop function with numpy arrays:")
    
    test_image = np.random.randint(0, 255, (176, 197, 3), dtype=np.uint8)
    print(f"Original test image shape: {test_image.shape}")
    
    for scale in [2, 3, 4]:
        cropped = modcrop(test_image, scale)
        pixels_lost = test_image.size - cropped.size
        percent_lost = (pixels_lost / test_image.size) * 100
        print(f"Scale {scale}: {test_image.shape} → {cropped.shape} (lost {pixels_lost:,} pixels, {percent_lost:.2f}%)")
    
    # Show SRCNN constraints
    demonstrate_srcnn_constraints()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("✓ Modcrop ensures dimensions are divisible by scale factor")
    print("✓ Required for SRCNN architecture compatibility")  
    print("✓ Minimal impact on image content (<2% pixels typically)")
    print("✓ Essential for proper super-resolution processing")
    print("\nThe small crop is worth it for the significant quality improvement!")

if __name__ == "__main__":
    main()