#!/usr/bin/env python3
"""
Comic-Gen: LoRA Training Script
Trains a LoRA model on your comic art style and characters using Kohya SS.

Usage:
    python train_lora.py --comic_name "your_comic" --epochs 100 --learning_rate 1e-4

Setup Instructions for RunPod:
1. Launch GPU instance (RTX 4090, A100, or similar)
2. Clone repository with submodules: git clone --recursive https://github.com/your-username/comic-gen.git
3. Install dependencies: pip install -r requirements.txt && cd kohya && pip install -r requirements.txt && cd ..
4. Upload your comic images to input_images/
5. Run this script with your parameters
6. Monitor training progress and download model when complete

Note: This script now uses the Kohya SS submodule located in the kohya/ directory.
"""

import os
import sys
import json
import yaml
import shutil
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch

# Add Kohya SS submodule to path
sys.path.append('./kohya')


class LoRATrainer:
    """LoRA Training class for comic generation"""
    
    def __init__(self, comic_name: str, input_dir: str = "input_images"):
        """
        Initialize the LoRA trainer
        
        Args:
            comic_name: Name of your comic (used for model naming)
            input_dir: Directory containing your comic images
        """
        self.comic_name = comic_name
        self.input_dir = input_dir
        self.dataset_path = f"./dataset/{comic_name}"
        self.output_path = f"./outputs/{comic_name}"
        self.target_size = 512
        self.train_ratio = 0.8
        
        # Create necessary directories
        self._create_directories()
        
        # Character names (customize for your comic)
        self.character_names = [
            "hero_name",  # Replace with your character names
            "villain_name",
            "sidekick_name"
        ]
        
        print(f"🎨 Initialized LoRA trainer for: {comic_name}")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Dataset path: {self.dataset_path}")
        print(f"📁 Output path: {self.output_path}")
        print(f"🔧 Using Kohya SS from submodule: ./kohya/")
    
    def _create_directories(self):
        """Create necessary directories for training"""
        directories = [
            f"{self.dataset_path}/train",
            f"{self.dataset_path}/validation",
            self.output_path,
            "./configs",
            "./models",
            "./test_outputs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def process_image(self, image_path: str, target_size: int = 512) -> Image.Image:
        """
        Process and resize image for training
        
        Args:
            image_path: Path to input image
            target_size: Target size for training
            
        Returns:
            Processed PIL Image
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Create square image with padding if necessary
            new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            new_img.paste(img, ((target_size - img.width) // 2, (target_size - img.height) // 2))
            
            return new_img
    
    def create_caption(self, image_path: str, comic_name: str, character_names: Optional[List[str]] = None) -> str:
        """
        Create caption for an image
        
        Args:
            image_path: Path to image file
            comic_name: Name of the comic
            character_names: List of character names to look for
            
        Returns:
            Generated caption string
        """
        filename = Path(image_path).stem
        
        # Base caption with comic style
        caption = f"{comic_name} style, comic panel"
        
        # Add character names if provided
        if character_names:
            for char in character_names:
                if char.lower() in filename.lower():
                    caption += f", {char}"
        
        # Add style-specific keywords
        caption += ", high quality, detailed artwork"
        
        return caption
    
    def prepare_dataset(self):
        """Prepare the dataset for training"""
        print(f"🔄 Preparing dataset for {self.comic_name}...")
        
        if not os.path.exists(self.input_dir):
            print(f"❌ Input directory not found: {self.input_dir}")
            print("   Please create the directory and upload your comic images")
            return False
        
        # Get image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ No images found in {self.input_dir}")
            print("   Please upload your comic images to this directory")
            return False
        
        print(f"📸 Found {len(image_files)} images to process")
        
        # Shuffle and split into train/validation
        random.shuffle(image_files)
        split_idx = int(len(image_files) * self.train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Kohya SS expects directories with repetition counts (e.g., "10_my_comic")
        # Create the correct directory structure
        train_repeat_dir = os.path.join(f"{self.dataset_path}/train", f"10_{self.comic_name}")
        val_repeat_dir = os.path.join(f"{self.dataset_path}/validation", f"10_{self.comic_name}")
        
        os.makedirs(train_repeat_dir, exist_ok=True)
        os.makedirs(val_repeat_dir, exist_ok=True)
        
        # Process training images
        print("🔄 Processing training images...")
        for i, filename in enumerate(train_files):
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(train_repeat_dir, filename)
            caption_path = os.path.join(train_repeat_dir, f"{Path(filename).stem}.txt")
            
            # Process image
            processed_img = self.process_image(input_path, self.target_size)
            processed_img.save(output_path)
            
            # Create caption
            caption = self.create_caption(filename, self.comic_name, self.character_names)
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(train_files)} training images")
        
        # Process validation images
        print("🔄 Processing validation images...")
        for i, filename in enumerate(val_files):
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(val_repeat_dir, filename)
            caption_path = os.path.join(val_repeat_dir, f"{Path(filename).stem}.txt")
            
            # Process image
            processed_img = self.process_image(input_path, self.target_size)
            processed_img.save(output_path)
            
            # Create caption
            caption = self.create_caption(filename, self.comic_name, self.character_names)
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(val_files)} validation images")
        
        print(f"✅ Dataset preparation completed!")
        print(f"📊 Training images: {len(train_files)}")
        print(f"📊 Validation images: {len(val_files)}")
        
        return True
    
    def verify_dataset_structure(self):
        """Verify that the dataset directory structure is correct for Kohya SS"""
        print(f"🔍 Verifying dataset structure...")
        
        # Check if dataset directory exists
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset directory not found: {self.dataset_path}")
            return False
        
        # Check if train and validation directories exist
        train_dir = os.path.join(self.dataset_path, "train")
        val_dir = os.path.join(self.dataset_path, "validation")
        
        if not os.path.exists(train_dir):
            print(f"❌ Training directory not found: {train_dir}")
            return False
        
        if not os.path.exists(val_dir):
            print(f"❌ Validation directory not found: {val_dir}")
            return False
        
        # Check for repetition directories (e.g., "10_my_comic")
        train_repeat_dir = os.path.join(train_dir, f"10_{self.comic_name}")
        val_repeat_dir = os.path.join(val_dir, f"10_{self.comic_name}")
        
        if not os.path.exists(train_repeat_dir):
            print(f"❌ Training repeat directory not found: {train_repeat_dir}")
            return False
        
        if not os.path.exists(val_repeat_dir):
            print(f"❌ Validation repeat directory not found: {val_repeat_dir}")
            return False
        
        # Check if directories contain images
        train_images = [f for f in os.listdir(train_repeat_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        val_images = [f for f in os.listdir(val_repeat_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not train_images:
            print(f"❌ No images found in training directory: {train_repeat_dir}")
            return False
        
        if not val_images:
            print(f"❌ No images found in validation directory: {val_repeat_dir}")
            return False
        
        print(f"✅ Dataset structure verified!")
        print(f"   Training images: {len(train_images)}")
        print(f"   Validation images: {len(val_images)}")
        print(f"   Total images: {len(train_images) + len(val_images)}")
        
        # Check for caption files
        train_captions = [f for f in os.listdir(train_repeat_dir) if f.endswith('.txt')]
        val_captions = [f for f in os.listdir(val_repeat_dir) if f.endswith('.txt')]
        
        print(f"   Training captions: {len(train_captions)}")
        print(f"   Validation captions: {len(val_captions)}")
        
        # Show the correct directory structure
        print(f"\n📂 Correct directory structure:")
        print(f"   {self.dataset_path}/")
        print(f"   ├── train/")
        print(f"   │   └── 10_{self.comic_name}/ ({len(train_images)} images)")
        print(f"   └── validation/")
        print(f"       └── 10_{self.comic_name}/ ({len(val_images)} images)")
        
        return True
    
    def create_training_config(self, epochs: int = 100, learning_rate: float = 1e-4, 
                              lora_rank: int = 16, batch_size: int = 1):
        """
        Create training configuration
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            lora_rank: LoRA rank (16-32 recommended)
            batch_size: Batch size per device
        """
        print(f"⚙️ Creating training configuration...")
        
        config = {
            "model": {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "model_name": f"{self.comic_name}_lora",
                "output_dir": self.output_path
            },
            "lora": {
                "r": lora_rank,
                "alpha": lora_rank * 2,
                "dropout": 0.1,
                "target_modules": [
                    "q_proj", "v_proj", "k_proj", "out_proj",
                    "to_q", "to_k", "to_v", "to_out.0"
                ]
            },
            "training": {
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": 4,
                "learning_rate": learning_rate,
                "lr_scheduler": "cosine",
                "lr_warmup_steps": 100,
                "weight_decay": 0.01,
                "mixed_precision": "fp16",
                "gradient_checkpointing": True,
                "logging_steps": 10,
                "save_steps": 500,
                "save_total_limit": 3,
                "evaluation_strategy": "steps",
                "eval_steps": 500
            },
            "dataset": {
                "train_data_dir": f"{self.dataset_path}/train",
                "validation_data_dir": f"{self.dataset_path}/validation",
                "resolution": self.target_size,
                "center_crop": True,
                "random_flip": True,
                "caption_extension": ".txt",
                "caption_column": "text"
            },
            "prompts": {
                "train_prompt": f"{self.comic_name} style, comic panel, {{description}}, high quality, detailed",
                "val_prompt": f"{self.comic_name} style, comic panel, {{description}}, high quality, detailed",
                "negative_prompt": "low quality, blurry, distorted, deformed, bad anatomy, bad proportions"
            }
        }
        
        # Save configuration
        config_path = f"./configs/{self.comic_name}_training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Training configuration saved: {config_path}")
        print(f"📊 Training for {epochs} epochs")
        print(f"🎯 LoRA rank: {lora_rank}")
        print(f"📚 Learning rate: {learning_rate}")
        
        return config_path
    
    def start_training(self, config_path: str):
        """
        Start LoRA training using Kohya SS
        
        Args:
            config_path: Path to training configuration file
        """
        print(f"🚀 Starting LoRA training for {self.comic_name}...")
        print(f"⏱️ This may take several hours depending on your dataset size")
        
        # Check if Kohya SS submodule is available
        if not os.path.exists('./kohya'):
            print("❌ Kohya SS submodule not found. Please initialize submodules:")
            print("   git submodule update --init --recursive")
            return False
        
        # Change to Kohya SS submodule directory
        os.chdir('./kohya')
        
        # Build training command
        # Note: train_data_dir should point to the parent directory containing train/validation folders
        training_cmd = [
            "accelerate", "launch", "--num_cpu_threads_per_process", "8",
            "train_network.py",
            "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
            f"--train_data_dir=../{self.dataset_path}",  # Point to parent directory, not train subdirectory
            f"--output_dir=../{self.output_path}",
            "--resolution=512",
            "--network_alpha=32",
            "--save_model_as=safetensors",
            "--network_module=networks.lora",
            f"--max_train_epochs={self.get_epochs_from_config(config_path)}",
            f"--learning_rate={self.get_lr_from_config(config_path)}",
            f"--unet_lr={self.get_lr_from_config(config_path)}",
            "--text_encoder_lr=1e-5",
            f"--network_dim={self.get_rank_from_config(config_path)}",
            f"--batch_size_per_device={self.get_batch_size_from_config(config_path)}",
            "--mixed_precision=fp16",
            "--save_every_n_epochs=10",
            "--save_precision=fp16",
            "--seed=42",
            "--caption_extension=.txt",
            "--cache_latents",
            "--optimizer_type=AdamW8bit",
            "--max_data_loader_n_workers=0",
            "--bucket_reso_steps=32",
            "--xformers",
            "--bucket_no_upscale",
            "--noise_offset=0.0",
            "--num_vectors_per_token=75",
            "--token_warmup_min=1",
            "--token_warmup_max=10",
            f"--output_name={self.comic_name}_lora"
        ]
        
        print("🔧 Training command:")
        print(" ".join(training_cmd))
        print(f"📁 Training data directory: ../{self.dataset_path}")
        print(f"📁 Expected structure:")
        print(f"   ../{self.dataset_path}/")
        print(f"   ├── train/ (training images)")
        print(f"   └── validation/ (validation images)")
        print(f"🔧 Using Kohya SS from: ./kohya/")
        print("\n🚀 Starting training...")
        
        try:
            # Run training
            result = subprocess.run(training_cmd, check=True)
            print("✅ Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with error: {e}")
            return False
        finally:
            # Change back to original directory
            os.chdir('..')
    
    def get_epochs_from_config(self, config_path: str) -> int:
        """Extract epochs from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['training']['num_train_epochs']
    
    def get_lr_from_config(self, config_path: str) -> float:
        """Extract learning rate from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['training']['learning_rate']
    
    def get_rank_from_config(self, config_path: str) -> int:
        """Extract LoRA rank from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['lora']['r']
    
    def get_batch_size_from_config(self, config_path: str) -> int:
        """Extract batch size from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['training']['per_device_train_batch_size']
    
    def test_model(self, test_prompts: Optional[List[str]] = None):
        """
        Test the trained model with sample prompts
        
        Args:
            test_prompts: List of test prompts (optional)
        """
        print(f"🧪 Testing trained LoRA model...")
        
        # Default test prompts
        if test_prompts is None:
            test_prompts = [
                f"{self.comic_name} style, hero character, determined expression, comic panel",
                f"{self.comic_name} style, action scene, dynamic pose, comic panel",
                f"{self.comic_name} style, dialogue scene, two characters talking, comic panel",
                f"{self.comic_name} style, establishing shot, detailed background, comic panel"
            ]
        
        # Check if model exists
        model_path = f"{self.output_path}/{self.comic_name}_lora.safetensors"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Please complete training first")
            return False
        
        print(f"✅ Found trained model: {model_path}")
        print("📝 Test prompts:")
        for i, prompt in enumerate(test_prompts):
            print(f"  {i+1}. {prompt}")
        
        # Note: Actual image generation would require loading the model
        # This is a placeholder for the testing functionality
        print("🎨 Model testing functionality would be implemented here")
        print("   (Requires loading the trained model with diffusers)")
        
        return True
    
    def export_model(self):
        """Export the trained model for use in the app"""
        print(f"📦 Exporting model for app integration...")
        
        # Source and target paths
        source_path = f"{self.output_path}/{self.comic_name}_lora.safetensors"
        target_path = f"./models/comic_lora.safetensors"
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"✅ Model exported to: {target_path}")
            
            # Update generation config
            config_path = "./configs/generation_config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['model']['lora_path'] = target_path
                config['model']['lora_scale'] = 0.8
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"✅ Updated generation config: {config_path}")
        else:
            print(f"⚠️ Model file not found: {source_path}")
            print("   Make sure training is complete")
            return False
        
        return True
    
    def run_full_pipeline(self, epochs: int = 100, learning_rate: float = 1e-4, 
                         lora_rank: int = 16, batch_size: int = 1):
        """
        Run the complete training pipeline
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            lora_rank: LoRA rank (16-32 recommended)
            batch_size: Batch size per device
        """
        print(f"🎯 Starting complete LoRA training pipeline for {self.comic_name}")
        print("=" * 60)
        
        # Step 1: Prepare dataset
        print("\n📁 Step 1: Preparing dataset...")
        if not self.prepare_dataset():
            return False
        
        # Step 2: Verify dataset structure
        print("\n🔍 Step 2: Verifying dataset structure...")
        if not self.verify_dataset_structure():
            return False
        
        # Step 3: Create training config
        print("\n⚙️ Step 3: Creating training configuration...")
        config_path = self.create_training_config(epochs, learning_rate, lora_rank, batch_size)
        
        # Step 4: Start training
        print("\n🚀 Step 4: Starting LoRA training...")
        if not self.start_training(config_path):
            return False
        
        # Step 5: Test model
        print("\n🧪 Step 5: Testing trained model...")
        self.test_model()
        
        # Step 6: Export model
        print("\n📦 Step 6: Exporting model...")
        self.export_model()
        
        print("\n" + "=" * 60)
        print(f"🎉 Training pipeline completed for {self.comic_name}!")
        print(f"📁 Your LoRA model: {self.output_path}/{self.comic_name}_lora.safetensors")
        print(f"🚀 Ready to use in the Gradio app: python app/app.py")
        
        return True


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Train LoRA model for comic generation")
    parser.add_argument("--comic_name", "-n", required=True, help="Name of your comic")
    parser.add_argument("--input_dir", "-i", default="input_images", help="Input images directory")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", "-r", type=int, default=16, help="LoRA rank (16-32)")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size per device")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare dataset, don't train")
    parser.add_argument("--test_only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LoRATrainer(args.comic_name, args.input_dir)
    
    if args.test_only:
        # Test existing model
        trainer.test_model()
    elif args.prepare_only:
        # Only prepare dataset
        trainer.prepare_dataset()
    else:
        # Run full pipeline
        trainer.run_full_pipeline(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main() 