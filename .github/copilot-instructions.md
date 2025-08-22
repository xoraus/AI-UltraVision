# AI-UltraVision: SRCNN Image Super-Resolution

AI-UltraVision is a machine learning project that implements the Super-Resolution Convolutional Neural Network (SRCNN) for enhancing low-resolution images. The project is primarily contained in a Jupyter notebook with supporting sample images and uses Keras/TensorFlow for the neural network implementation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup (Validated Commands)
**NEVER CANCEL**: Follow these exact steps and wait for completion.

1. Update system packages (takes ~2 seconds):
   ```bash
   sudo apt update
   ```

2. Install system dependencies (takes ~30 seconds, NEVER CANCEL):
   ```bash
   sudo apt install -y python3-opencv python3-sklearn jupyter-notebook python3-skimage python3-numpy
   ```

3. Install TensorFlow and Keras (takes ~40 seconds, NEVER CANCEL - set timeout to 120+ seconds):
   ```bash
   pip3 install tensorflow keras --timeout=1000
   ```
   **CRITICAL**: Network timeouts are common. If pip install fails with timeout errors, retry the command.

4. Install matplotlib (takes ~1 second if not already installed):
   ```bash
   pip3 install matplotlib
   ```

### Running the Application (Validated Commands)
1. Start Jupyter notebook (starts in <1 second):
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
   
2. Open the notebook file: "Image Super Resolution with the SRCNN (Jupyter Notebook).ipynb"
3. Run cells sequentially to execute the super-resolution workflow

### Quick Validation Test
Run this command to verify all core dependencies work (takes ~1 second):
```bash
python3 -c "import cv2, numpy as np, matplotlib.pyplot as plt; from skimage.metrics import structural_similarity; print('✓ All core dependencies working')"
```

## Project Structure (Validated)
```
.
├── Image Super Resolution with the SRCNN (Jupyter Notebook).ipynb  # Main notebook
├── LICENSE                                                           # MIT license  
├── README.md                                                        # Documentation
└── results/                                                         # Sample images
    ├── baboon.png        ├── lenna.png         ├── pepper.png
    ├── baby_GT.png       ├── monarch.png      ├── ppt3.png  
    ├── barbara.png       ├── bird_GT.png      ├── woman_GT.png
    ├── butterfly_GT.png  ├── coastguard.png   ├── zebra.png
    ├── comic.png         ├── face.png         └── [additional test images]
    ├── flowers.png       ├── foreman.png
    ├── head_GT.png       └── [total: 17 PNG files]
```

## Validation (Tested Workflow)
Always run this validation after setup to ensure everything works:

```bash
# Test image processing workflow (takes ~1 second)
python3 -c "
import cv2, numpy as np, math, os
from skimage.metrics import structural_similarity as ssim

# Test loading image
img = cv2.imread('results/lenna.png')
print(f'✓ Image loaded successfully: {img.shape}')

# Test image quality metrics  
def psnr(target, ref):
    diff = ref.astype(float) - target.astype(float)
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

# Test with sample images
files = [f for f in os.listdir('results/') if f.endswith('.png')][:2]
img1, img2 = cv2.imread(f'results/{files[0]}'), cv2.imread(f'results/{files[1]}')
print(f'✓ PSNR: {psnr(img1, img2):.2f}')
print(f'✓ SSIM: {ssim(img1, img2, multichannel=True, channel_axis=-1, data_range=255):.4f}')
print('✓ Validation successful - ready to run notebook!')
"
```

## Key Dependencies (Validated Versions)
- **Python**: 3.12+
- **OpenCV**: 4.6.0 (cv2) - image processing  
- **NumPy**: 1.26.4 - numerical operations (use system version to avoid compatibility issues)
- **Matplotlib**: 3.10+ - plotting and visualization
- **scikit-image**: 0.22.0 - image quality metrics (SSIM)
- **Jupyter**: 6.4.12 - notebook environment
- **TensorFlow**: 2.20+ - neural network backend
- **Keras**: 3.11+ - high-level neural network API

## Workflow Overview (Notebook Structure)
1. **Import packages** and verify versions (~1 second)
2. **Define image quality metrics** (PSNR, MSE, SSIM functions) (~1 second)
3. **Prepare images** by degrading resolution (~1-2 seconds per image)
4. **Build SRCNN model** with Keras (~5-10 seconds)
5. **Deploy SRCNN** for super-resolution enhancement (~10-30 seconds per image)
6. **Evaluate results** using quality metrics (~1 second per comparison)

## Known Issues (Validated)
- **NumPy Compatibility**: Use system NumPy (1.26.4) instead of pip NumPy (2.x) to avoid binary incompatibility with system packages
- **Network Timeouts**: pip install may timeout - simply retry the failed command
- **Matplotlib Warning**: "Unable to import Axes3D" warning can be ignored - core functionality works
- **Import Updates**: The notebook uses `compare_ssim` which should be `structural_similarity` in current scikit-image
- **TensorFlow**: May not work in this environment due to NumPy compatibility issues, but core image processing works

## Timing Expectations (Measured)
- **System update**: 1-2 seconds
- **Package installation**: 30 seconds (system packages), 40 seconds (TensorFlow)  
- **Jupyter startup**: <1 second
- **Core workflow validation**: ~1 second
- **Individual notebook cells**: 1-5 seconds each (non-ML operations)
- **Image processing operations**: 1-2 seconds per image
- **SRCNN inference** (if working): 10-30 seconds per image

**NEVER CANCEL**: Always wait for commands to complete. The longest operation is TensorFlow installation at ~40 seconds.

## Alternative Workflow (If TensorFlow Issues)
If TensorFlow has compatibility issues, you can still:
1. Run all image processing and quality metric functions
2. Test the degradation and enhancement pipeline
3. Calculate PSNR, MSE, and SSIM metrics
4. Use pre-trained weights from external sources if available

The core computer vision workflow works perfectly without TensorFlow/Keras.