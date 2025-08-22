# AI-UltraVision: SRCNN Image Super-Resolution

AI-UltraVision is a machine learning project that implements the Super-Resolution Convolutional Neural Network (SRCNN) for enhancing low-resolution images. The project is primarily contained in a Jupyter notebook with supporting sample images and uses Keras/TensorFlow for the neural network implementation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- Update system packages: `sudo apt update`
- Install system dependencies:
  ```bash
  sudo apt install -y python3-opencv python3-sklearn jupyter-notebook
  ```
- Install Python dependencies:
  ```bash
  pip3 install numpy matplotlib tensorflow keras scikit-image --timeout=1000
  ```
  **NEVER CANCEL**: Python package installation can take 15-30 minutes for TensorFlow. Set timeout to 60+ minutes.

### Running the Application
- Start Jupyter notebook: `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`
- Open the notebook file: "Image Super Resolution with the SRCNN (Jupyter Notebook).ipynb"
- Run cells sequentially to execute the super-resolution workflow

## Project Structure
- **Main notebook**: "Image Super Resolution with the SRCNN (Jupyter Notebook).ipynb" - Contains all code and documentation
- **Sample images**: `results/` directory - PNG files for testing (baboon.png, lenna.png, etc.)
- **README.md** - Project documentation with theory and examples
- **LICENSE** - MIT license

## Validation
- Always test the complete workflow by running all notebook cells
- Verify image processing by checking that PSNR, MSE, and SSIM metrics are calculated correctly
- Test with at least one sample image from the results/ directory
- Ensure plots and visualizations render correctly in the notebook

## Common Tasks

### Repository Root Structure
```
.
├── Image Super Resolution with the SRCNN (Jupyter Notebook).ipynb
├── LICENSE
├── README.md
└── results/
    ├── baboon.png
    ├── lenna.png
    ├── pepper.png
    └── [additional sample images]
```

### Key Dependencies and Versions
The notebook expects these Python packages:
- Keras (for neural network architecture)
- OpenCV (cv2) for image processing
- NumPy for numerical operations
- Matplotlib for plotting and visualization
- scikit-image for image quality metrics (SSIM)
- Jupyter for running the notebook

### Workflow Overview
1. **Import packages** and verify versions
2. **Define image quality metrics** (PSNR, MSE, SSIM)
3. **Prepare images** by degrading resolution
4. **Build SRCNN model** with Keras
5. **Deploy SRCNN** for super-resolution enhancement
6. **Evaluate results** using quality metrics

## Known Issues
- TensorFlow installation may fail due to network timeouts - use system packages (python3-opencv) where possible
- Notebook was originally written for Python 2.7 but should work with Python 3.12+
- Some deprecated imports (compare_ssim) may need updating to current scikit-image API

## Timing Expectations
- **Environment setup**: 5-10 minutes for system packages, 15-30 minutes for Python packages
- **Notebook execution**: 2-5 minutes per cell, total runtime 10-20 minutes depending on image processing
- **Image processing**: Individual image enhancement takes 30 seconds to 2 minutes per image

Always validate that package installations complete successfully before proceeding to run the notebook.