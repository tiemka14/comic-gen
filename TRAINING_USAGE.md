# LoRA Training Usage Guide

You have two options for training your LoRA model: **Jupyter Notebook** or **Python Script**.

## 🎯 Option 1: Jupyter Notebook (Recommended for Beginners)

### RunPod Setup:
1. **Launch RunPod instance** (RTX 4090 recommended)
2. **Upload files** to your instance
3. **Open notebook**: `notebooks/train_lora.ipynb`
4. **Run cells sequentially**

### Usage:
```bash
# In RunPod terminal
cd comic-gen
jupyter lab
# Then open notebooks/train_lora.ipynb in browser
```

**Pros:**
- ✅ Visual interface
- ✅ Step-by-step execution
- ✅ Easy to debug
- ✅ Progress monitoring
- ✅ Interactive testing

## 🚀 Option 2: Python Script (Recommended for Automation)

### RunPod Setup:
1. **Launch RunPod instance** (RTX 4090 recommended)
2. **Upload files** to your instance
3. **Run script** with command line arguments

### Usage:

#### Full Training Pipeline:
```bash
# Complete training (dataset prep + training + testing + export)
python scripts/train_lora.py --comic_name "your_comic" --epochs 100 --learning_rate 1e-4
```

#### Dataset Preparation Only:
```bash
# Just prepare dataset, don't train
python scripts/train_lora.py --comic_name "your_comic" --prepare_only
```

#### Test Existing Model:
```bash
# Test a model that's already trained
python scripts/train_lora.py --comic_name "your_comic" --test_only
```

#### Custom Parameters:
```bash
# Full control over training parameters
python scripts/train_lora.py \
  --comic_name "my_awesome_comic" \
  --epochs 150 \
  --learning_rate 5e-5 \
  --lora_rank 32 \
  --batch_size 2 \
  --input_dir "my_comic_images"
```

### Command Line Options:
```bash
--comic_name, -n     Name of your comic (required)
--input_dir, -i      Input images directory (default: input_images)
--epochs, -e         Number of training epochs (default: 100)
--learning_rate, -lr Learning rate (default: 1e-4)
--lora_rank, -r      LoRA rank 16-32 (default: 16)
--batch_size, -b     Batch size per device (default: 1)
--prepare_only       Only prepare dataset, don't train
--test_only          Only test existing model
```

**Pros:**
- ✅ Faster execution
- ✅ Easy automation
- ✅ Command line control
- ✅ Better for batch processing
- ✅ Can be run in background

## 📊 Parameter Guidelines

### Dataset Size Recommendations:
| Images | Epochs | Training Time | LoRA Rank | Quality |
|--------|--------|---------------|-----------|---------|
| 20-30  | 150-200| 2-4 hours     | 16        | Good    |
| 30-50  | 100-150| 3-6 hours     | 16-24     | Better  |
| 50-100 | 80-120 | 4-8 hours     | 24-32     | Best    |

### Learning Rate Guidelines:
- **1e-4**: Standard rate, good for most cases
- **5e-5**: Lower rate, better for small datasets
- **2e-4**: Higher rate, faster training (may be less stable)

### LoRA Rank Guidelines:
- **16**: Good balance of quality and speed
- **24**: Better quality, more parameters
- **32**: Best quality, most parameters (requires more VRAM)

## 🔧 Quick Start Commands

### For Small Dataset (20-30 images):
```bash
python scripts/train_lora.py --comic_name "my_comic" --epochs 150 --learning_rate 5e-5
```

### For Medium Dataset (30-50 images):
```bash
python scripts/train_lora.py --comic_name "my_comic" --epochs 120 --lora_rank 24
```

### For Large Dataset (50+ images):
```bash
python scripts/train_lora.py --comic_name "my_comic" --epochs 100 --lora_rank 32 --batch_size 2
```

## 📁 File Structure After Training

```
comic-gen/
├── dataset/
│   └── your_comic/
│       ├── train/          # Training images + captions
│       └── validation/     # Validation images + captions
├── outputs/
│   └── your_comic/
│       ├── your_comic_lora.safetensors  # Your trained model
│       └── trainer_state.json           # Training logs
├── models/
│   └── comic_lora.safetensors          # Exported model for app
└── configs/
    └── your_comic_training_config.yaml  # Training configuration
```

## 🎨 Integration with Comic-Gen App

After training, your model is automatically integrated:

```bash
# Launch the Gradio app
python app/app.py

# Your comic style will be available in the dropdown
# Use prompts like: "your_comic_name style, hero character, comic panel"
```

## 🆘 Troubleshooting

### Common Issues:

#### Out of Memory:
```bash
# Reduce batch size
python scripts/train_lora.py --comic_name "my_comic" --batch_size 1
```

#### Poor Quality:
```bash
# Increase epochs and reduce learning rate
python scripts/train_lora.py --comic_name "my_comic" --epochs 200 --learning_rate 5e-5
```

#### Slow Training:
```bash
# Increase learning rate and reduce epochs
python scripts/train_lora.py --comic_name "my_comic" --epochs 80 --learning_rate 2e-4
```

## 💡 Tips

1. **Start with the notebook** if you're new to LoRA training
2. **Use the script** for repeated training or automation
3. **Monitor training progress** in the output logs
4. **Test your model** before downloading
5. **Save your best models** with descriptive names
6. **Use consistent image quality** for best results

---

**Happy Training! 🎨**

Both methods will give you the same high-quality LoRA model for your comic style! 