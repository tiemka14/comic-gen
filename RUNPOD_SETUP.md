# RunPod Setup Guide for Comic-Gen LoRA Training

This guide will walk you through setting up and running LoRA training on RunPod for your comic style and characters.

## 🚀 Quick Start

1. **Launch RunPod Instance** (5 minutes)
2. **Clone Repository with Submodules** (5 minutes)
3. **Upload Your Comic Images** (10 minutes)
4. **Run Training Notebook** (2-8 hours depending on dataset)
5. **Download Your Trained Model** (5 minutes)

## 📋 Prerequisites

- RunPod account (sign up at [runpod.io](https://runpod.io))
- Credit card for GPU instance costs (~$0.50-2.00/hour)
- Your comic images (20-100 high-quality panels)
- Basic understanding of Jupyter notebooks

## 🖥️ Step 1: Launch RunPod Instance

### 1.1 Choose GPU Instance
- **Recommended**: RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **Budget Option**: RTX 3080 (10GB VRAM) - may need smaller batch size
- **High-End**: A100 80GB for large datasets

### 1.2 Select Template
Choose a template with:
- ✅ PyTorch pre-installed
- ✅ Jupyter Lab/Notebook
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
3. Click "Connect" → "HTTP Service" → "Jupyter Lab"
4. You'll see the Jupyter interface

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
   - Use Jupyter Lab file browser
   - Drag and drop your images to `input_images/`
   - Supported formats: PNG, JPG, JPEG, WebP

### 3.2 Image Requirements
- **Quantity**: 20-100 images (more = better quality)
- **Quality**: High resolution, clear artwork
- **Style**: Consistent art style and character designs
- **Content**: Various poses, expressions, scenes
- **Format**: PNG/JPG, any size (will be resized automatically)

### 3.3 Character Information
Prepare a list of your comic characters:
```python
CHARACTER_NAMES = [
    "Hero Name",
    "Villain Name", 
    "Sidekick Name",
    # Add all your characters
]
```

## 🎯 Step 4: Configure Training

### 4.1 Edit Training Parameters
Open `notebooks/train_lora.ipynb` and modify:

```python
# Your comic name (used for model naming)
COMIC_NAME = "your_comic_name"

# Character names from your comic
CHARACTER_NAMES = [
    "your_hero_name",
    "your_villain_name",
    # etc.
]

# Training parameters (adjust based on your needs)
TRAINING_CONFIG = {
    "training": {
        "num_train_epochs": 100,  # 50-200 depending on dataset size
        "learning_rate": 1e-4,    # 1e-4 to 1e-5
    },
    "lora": {
        "r": 16,  # 16-32 (higher = more capacity)
    }
}
```

### 4.2 Dataset Size Guidelines
| Images | Epochs | Training Time | Quality |
|--------|--------|---------------|---------|
| 20-30  | 150-200| 2-4 hours     | Good    |
| 30-50  | 100-150| 3-6 hours     | Better  |
| 50-100 | 80-120 | 4-8 hours     | Best    |

## 🚀 Step 5: Start Training

### 5.1 Run the Notebook
1. Open `notebooks/train_lora.ipynb`
2. Run cells 1-3 (setup and data preparation)
3. **Uncomment the training command** in cell 4
4. Run cell 4 to start training

**Note**: The training notebook now references scripts from the `kohya/` submodule directory.

### 5.2 Monitor Training
- **Loss**: Should decrease over time
- **Samples**: Generated every 500 steps
- **Checkpoints**: Saved every 10 epochs
- **Time**: 2-8 hours depending on dataset

### 5.3 Training Progress Indicators
```
✅ Good signs:
- Loss decreasing steadily
- Generated samples improving
- No out-of-memory errors

⚠️ Warning signs:
- Loss not decreasing
- Generated samples poor quality
- Memory errors
```

## 📊 Step 6: Monitor and Optimize

### 6.1 Check Training Logs
```bash
# View training progress
tail -f outputs/your_comic_name/trainer_state.json

# Check GPU usage
nvidia-smi
```

### 6.2 Common Adjustments
If training isn't going well:

```python
# Reduce learning rate
"learning_rate": 5e-5

# Increase epochs
"num_train_epochs": 150

# Reduce batch size (if out of memory)
"per_device_train_batch_size": 1
```

### 6.3 Early Stopping
If you see good results early:
1. Stop training (Ctrl+C in terminal)
2. Use the latest checkpoint
3. Test the model

## 🎨 Step 7: Test Your Model

### 7.1 Generate Test Images
Run the testing cell in the notebook:

```python
# Test prompts for your comic
test_prompts = [
    "your_comic_name style, hero character, determined expression, comic panel",
    "your_comic_name style, action scene, dynamic pose, comic panel",
    # Add more test prompts
]
```

### 7.2 Evaluate Results
Check generated images for:
- ✅ Style consistency with your comic
- ✅ Character likeness
- ✅ Overall quality
- ✅ Prompt adherence

## 💾 Step 8: Download Your Model

### 8.1 Locate Your Model
```bash
# Your trained model will be at:
outputs/your_comic_name/your_comic_name_lora.safetensors
```

### 8.2 Download Options
1. **Jupyter Lab**: Right-click → Download
2. **Terminal**: Use `scp` or `rsync`
3. **Cloud Storage**: Upload to Google Drive/Dropbox

### 8.3 Model Size
- **LoRA file**: ~50-200MB
- **Full model**: ~4GB (if you want the complete model)

## 🔧 Step 9: Integration with Comic-Gen

### 9.1 Update Configuration
Copy your model to the local comic-gen setup:

```bash
# Copy model to your local setup
cp your_comic_name_lora.safetensors /path/to/comic-gen/models/comic_lora.safetensors
```

### 9.2 Test in Gradio App
```bash
cd /path/to/comic-gen
python app/app.py
```

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

### 9.1 Instance Selection
| Instance | Cost/Hour | Best For |
|----------|-----------|----------|
| RTX 3080 | $0.30-0.50 | Small datasets, testing |
| RTX 4090 | $0.60-1.00 | Medium datasets, good balance |
| A100 40GB | $1.50-2.50 | Large datasets, fast training |

### 9.2 Time Estimates
| Dataset Size | Training Time | Cost (RTX 4090) |
|--------------|---------------|-----------------|
| 20 images    | 2-3 hours     | $1.20-1.80      |
| 50 images    | 4-6 hours     | $2.40-3.60      |
| 100 images   | 6-8 hours     | $3.60-4.80      |

### 9.3 Money-Saving Tips
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