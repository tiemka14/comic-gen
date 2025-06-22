# RunPod Setup Guide for Comic-Gen LoRA Training

This guide will walk you through setting up and running LoRA training on RunPod for your comic style and characters using only scripts (no notebooks required).

## 🚀 Quick Start

1. **Launch RunPod Instance** (5 minutes)
2. **Clone Repository with Submodules** (5 minutes)
3. **Upload Your Comic Images** (10 minutes)
4. **Run Training Script** (2-8 hours depending on dataset)
5. **Download Your Trained Model** (5 minutes)

## 📋 Prerequisites

- RunPod account (sign up at [runpod.io](https://runpod.io))
- Credit card for GPU instance costs (~$0.50-2.00/hour)
- Your comic images (20-100 high-quality panels)
- Basic familiarity with Python scripts and terminal

## 🖥️ Step 1: Launch RunPod Instance

### 1.1 Choose GPU Instance
- **Recommended**: RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **Budget Option**: RTX 3080 (10GB VRAM) - may need smaller batch size
- **High-End**: A100 80GB for large datasets

### 1.2 Select Template
Choose a template with:
- ✅ PyTorch pre-installed
- ✅ Jupyter Lab/Notebook (for file upload/terminal, not for notebooks)
- ✅ CUDA support
- ✅ Python 3.8+

**Recommended Template**: `RunPod PyTorch 2.0.1`

### 1.3 Instance Configuration
```
GPU: RTX 4090 (or your choice)
CPU: 8+ cores
RAM: 32GB+
Storage: 100GB+ (SSD preferred)
```

### 1.4 Launch and Connect
1. Click "Deploy"
2. Wait for instance to start (2-5 minutes)
3. Click "Connect" → "HTTP Service" → "Jupyter Lab" (for file browser/terminal)

## 📁 Step 2: Prepare Your Workspace

### 2.1 Clone the Repository with Submodules
In Jupyter Lab, open a terminal and run:

```bash
# Clone the comic-gen repository with submodules
git clone --recursive https://github.com/your-username/comic-gen.git
cd comic-gen

# If you already cloned without --recursive, initialize submodules manually:
git submodule update --init --recursive
```

**Troubleshooting: If you cloned without --recursive**

If you already cloned the repository without the `--recursive` flag, you'll see an empty `kohya/` directory. Here's how to fix it:

```bash
# Check if submodules are initialized
ls -la kohya/

# If kohya/ is empty or shows "submodule" files, initialize submodules:
git submodule update --init --recursive

# Verify the submodule is properly set up
ls -la kohya/
# You should see files like train_network.py, requirements.txt, etc.

# If you still have issues, try:
git submodule init
git submodule update
```

**Alternative: Re-clone with submodules**
```bash
# If submodule initialization fails, you can re-clone:
cd ..
rm -rf comic-gen
git clone --recursive https://github.com/your-username/comic-gen.git
cd comic-gen
```

**Verify Submodule Setup**
```bash
# After cloning or initializing, verify everything is set up correctly:
ls -la kohya/
# Should show files like: train_network.py, requirements.txt, networks/, etc.

# Check submodule status
git submodule status
# Should show: kohya <commit-hash> (path)
```

### 2.2 Install Dependencies
```bash
# Install main project dependencies
pip install -r requirements.txt

# Install Kohya SS dependencies (now in submodule)
cd kohya
pip install -r requirements.txt
cd ..

# Install additional PyTorch packages if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install xformers
pip install opencv-python pillow pyyaml
pip install jupyter ipywidgets
```

### 2.3 Verify Installation
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify Kohya submodule is properly set up
ls -la kohya/
```

## 🖼️ Step 3: Prepare Your Dataset

### 3.1 Upload Your Images
1. **Create input directory**:
   ```bash
   mkdir -p input_images
   ```

2. **Upload your comic images**:
   - Use Jupyter Lab file browser or terminal
   - Drag and drop your images to `input_images/`
   - Supported formats: PNG, JPG, JPEG, WebP

### 3.2 Image Requirements
- **Quantity**: 20-100 images (more = better quality)
- **Quality**: High resolution, clear artwork
- **Style**: Consistent art style and character designs
- **Content**: Various poses, expressions, scenes
- **Format**: PNG/JPG, any size (will be resized automatically)

### 3.3 Character Information
Prepare a list of your comic characters in the script or config as needed.

## 🎯 Step 4: Configure and Run Training (Script-Based)

All training is done via the script `scripts/train_lora.py`.

**Run the training script from the project root directory:**

```bash
python scripts/train_lora.py --comic_name "your_comic" --epochs 100 --learning_rate 1e-4
```

**Arguments:**
- `--comic_name` (required): Name of your comic (used for model naming)
- `--input_dir` (optional): Directory containing your comic images (default: `input_images`)
- `--epochs` (optional): Number of training epochs (default: 100)
- `--learning_rate` (optional): Learning rate (default: 1e-4)
- `--lora_rank` (optional): LoRA rank (default: 16)
- `--batch_size` (optional): Batch size per device (default: 1)
- `--prepare_only`: Only prepare dataset, don't train
- `--test_only`: Only test existing model

**Example:**
```bash
python scripts/train_lora.py --comic_name "my_comic" --epochs 120 --learning_rate 5e-5
```

**Note:**
- Make sure your images are in the `input_images/` directory (or specify with `--input_dir`).
- The script will handle dataset preparation, training, testing, and model export.
- You do **not** need to use Jupyter notebooks for any part of the workflow.

## 🎨 Step 5: Test and Download Your Model

### 5.1 Locate Your Model
```bash
# Your trained model will be at:
outputs/your_comic_name/your_comic_name_lora.safetensors
```

### 5.2 Download Options
1. **Jupyter Lab**: Right-click → Download
2. **Terminal**: Use `scp` or `rsync`
3. **Cloud Storage**: Upload to Google Drive/Dropbox

### 5.3 Model Size
- **LoRA file**: ~50-200MB
- **Full model**: ~4GB (if you want the complete model)

## 🔧 Step 6: Integration with Comic-Gen

### 6.1 Update Configuration
Copy your model to the local comic-gen setup:

```bash
# Copy model to your local setup
cp outputs/your_comic_name/your_comic_name_lora.safetensors /path/to/comic-gen/models/comic_lora.safetensors
```

### 6.2 Test in Gradio App
```bash
cd /path/to/comic-gen
python app/app.py
```

## ✅ Tested Environment

This setup has been successfully tested on RunPod with the following environment. The `requirements.txt` file is based on these versions:

- **Python**: 3.10
- **CUDA**: 11.8 (inferred from torch build)
- **Key Libraries**:
  - `torch==2.7.0`
  - `bitsandbytes==0.46.0`
  - `triton==3.3.1`
  - `xformers==0.0.30`
  - `accelerate==0.30.0`
  - `diffusers==0.25.0`

Following the setup steps and using the provided `requirements.txt` should create a compatible environment for training.

## 🔄 Managing the Kohya Submodule

### Updating Kohya
If you need the latest Kohya features:

```bash
# Update Kohya to latest version
git submodule update --remote kohya
git add kohya
git commit -m "Update Kohya submodule to latest version"

# Reinstall dependencies if needed
cd kohya
pip install -r requirements.txt
cd ..
```

### Using Specific Kohya Version
If you need a specific version for compatibility:

```bash
cd kohya
git checkout <commit-hash-or-tag>
cd ..
git add kohya
git commit -m "Pin Kohya to specific version"
```

## 💰 Cost Optimization

### Instance Selection
| Instance | Cost/Hour | Best For |
|----------|-----------|----------|
| RTX 3080 | $0.30-0.50 | Small datasets, testing |
| RTX 4090 | $0.60-1.00 | Medium datasets, good balance |
| A100 40GB | $1.50-2.50 | Large datasets, fast training |

### Time Estimates
| Dataset Size | Training Time | Cost (RTX 4090) |
|--------------|---------------|-----------------|
| 20 images    | 2-3 hours     | $1.20-1.80      |
| 50 images    | 4-6 hours     | $2.40-3.60      |
| 100 images   | 6-8 hours     | $3.60-4.80      |

### Money-Saving Tips
1. **Use spot instances** (30-50% cheaper)
2. **Monitor training** to stop early if needed
3. **Test on smaller instance** first
4. **Download models** immediately after training

## 🆘 Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size in config
"per_device_train_batch_size": 1
"gradient_accumulation_steps": 8
```

#### Poor Training Quality
```python
# Increase dataset size or quality
# Adjust learning rate
"learning_rate": 5e-5  # Try lower rate
```

#### Slow Training
```bash
# Enable optimizations
pip install xformers
# Use mixed precision
"mixed_precision": "fp16"
```

#### Submodule Issues
```bash
# If Kohya submodule is not working:
git submodule update --init --recursive
cd kohya
git checkout main
cd ..

# If submodule is in detached HEAD state:
cd kohya
git checkout main
git pull origin main
cd ..

# If submodule shows as modified:
git submodule update --init --recursive --force

# If you get "fatal: No url found for submodule path":
git submodule init
git submodule update

# Verify submodule is properly set up:
ls -la kohya/
git submodule status
```

#### Connection Issues
- Use RunPod's built-in terminal
- Save work frequently
- Use cloud storage for backups

### Getting Help
- **RunPod Discord**: Community support
- **Kohya SS Issues**: GitHub repository
- **Comic-Gen Issues**: Your repository

## 🎉 Success Checklist

- [ ] RunPod instance launched successfully
- [ ] Repository cloned with submodules
- [ ] Kohya submodule properly initialized (verify with `ls -la kohya/`)
- [ ] Dependencies installed (main + Kohya)
- [ ] Images uploaded and processed
- [ ] Training configuration set
- [ ] Training completed without errors
- [ ] Test images generated successfully
- [ ] Model downloaded to local machine
- [ ] Model integrated with Comic-Gen app
- [ ] Final testing completed

## 📚 Additional Resources

- [Kohya SS Documentation](https://github.com/kohya-ss/sd-scripts)
- [RunPod Documentation](https://docs.runpod.io/)
- [LoRA Training Guide](https://github.com/kohya-ss/sd-scripts/wiki/LoRA-training)
- [Stable Diffusion Guide](https://huggingface.co/docs/diffusers/training/lora)
- [Git Submodules Guide](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

---

**Happy Training! 🎨**

Your custom comic LoRA model will be ready to generate amazing panels in your unique style! 