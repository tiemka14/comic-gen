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
├── notebooks/
│   └── train_lora.ipynb          # LoRA training notebook (Kohya-compatible)
├── scripts/
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
└── README.md                     # This file
```

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Kohya Training Environment** (for LoRA training):
   ```bash
   # Follow Kohya installation guide or use RunPod
   git clone https://github.com/kohya-ss/sd-scripts
   cd sd-scripts
   pip install -r requirements.txt
   ```

## 📚 Usage

### 1. LoRA Training

Open `notebooks/train_lora.ipynb` in your preferred environment:
- **Local**: Jupyter notebook with Kohya setup
- **Cloud**: RunPod, Google Colab, or similar

The notebook includes:
- Dataset preparation
- Training configuration
- LoRA fine-tuning for comic styles
- Model evaluation and testing

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

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Kohya SS for LoRA training scripts
- Stability AI for Stable Diffusion
- Gradio for the web interface
- The open-source AI art community

---

**Happy Comic Creating! 🎭** 