# Comic-Gen: AI-Powered Comic Generation Pipeline

A complete pipeline for generating comic panels from text scripts using Stable Diffusion 1.5 and LoRA fine-tuning.

## 🚀 Features

- **LoRA Training**: Fine-tune Stable Diffusion for comic art styles
- **Script Processing**: Convert text scripts into structured panel prompts
- **Image Generation**: Generate comic panels with Gradio web interface
- **Prompt Templates**: Pre-built templates for various comic styles and panel types

## 📁 Repository Structure

```
comic-gen/
├── kohya/                        # Kohya SS LoRA training scripts (git submodule)
├── scripts/
│   ├── train_lora.py             # Script-based LoRA training (run from project root)
│   ├── script_to_panels.py       # Script-to-panel processor
│   └── prompt_templates.py       # Prompt templates and styles
├── app/
│   └── app.py                    # Gradio web interface
├── configs/
│   ├── training_config.yaml      # LoRA training configuration
│   └── generation_config.yaml    # Image generation settings
├── templates/
│   ├── comic_styles.json         # Comic art style definitions
│   └── panel_types.json          # Panel type templates
├── requirements.txt              # Python dependencies
├── .gitmodules                   # Git submodule configuration
└── README.md                     # This file
```

## 🛠️ Setup

### 1. Clone the Repository with Submodules

```bash
# Clone the repository and initialize submodules
git clone --recursive https://github.com/yourusername/comic-gen.git
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
git clone --recursive https://github.com/yourusername/comic-gen.git
cd comic-gen
```

### 2. Install Dependencies

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install Kohya dependencies
cd kohya
pip install -r requirements.txt
cd ..
```

### 3. Managing the Kohya Submodule

The Kohya SS training scripts are included as a git submodule. This allows you to:
- Keep the training scripts up-to-date independently
- Maintain a specific version of Kohya for compatibility
- Avoid nested git repository issues

**To update Kohya to the latest version:**
```bash
git submodule update --remote kohya
git add kohya
git commit -m "Update Kohya submodule to latest version"
```

**To use a specific version of Kohya:**
```bash
cd kohya
git checkout <commit-hash-or-tag>
cd ..
git add kohya
git commit -m "Pin Kohya to specific version"
```

## 📚 Usage

### 1. LoRA Training (Script-Based)

All training is now done via the script `scripts/train_lora.py`.

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

### 2. Script Processing

Convert your comic script into structured panel prompts:

```bash
python scripts/script_to_panels.py --input script.txt --output panels.json
```

This generates a JSON file with structured prompts for each panel.

### 3. Image Generation

Launch the Gradio web interface:

```bash
python app/app.py
```

Features:
- Upload panel prompt JSONs
- Generate individual panels
- Batch generation
- Style selection
- LoRA model loading

## 🎨 Comic Styles

Pre-configured styles include:
- **Manga**: Japanese comic style
- **Western**: American comic book style
- **European**: Franco-Belgian comic style
- **Webcomic**: Modern digital style
- **Vintage**: Classic comic strip style

## 📝 Prompt Templates

The system includes templates for:
- **Panel Types**: Close-up, wide shot, action, dialogue
- **Character Poses**: Standing, sitting, running, fighting
- **Emotions**: Happy, sad, angry, surprised
- **Backgrounds**: Urban, nature, interior, fantasy

## 🔧 Configuration

### Training Configuration (`configs/training_config.yaml`)
- Model settings
- LoRA parameters
- Training hyperparameters
- Dataset paths

### Generation Configuration (`configs/generation_config.yaml`)
- Stable Diffusion settings
- Sampling parameters
- Style presets
- Output formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

**Note for Contributors**: When contributing, make sure to:
- Test with the current Kohya submodule version
- Update the submodule if necessary for new features
- Document any changes that affect the training pipeline

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Kohya SS for LoRA training scripts
- Stability AI for Stable Diffusion
- Gradio for the web interface
- The open-source AI art community

---

**Happy Comic Creating! 🎭** 