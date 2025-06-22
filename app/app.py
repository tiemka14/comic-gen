import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import yaml
import os
import json
import sys

# Add the project root to Python path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prompt_templates import PromptTemplates

# Load generation config
def load_config(path="../configs/generation_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_styles(path="../templates/comic_styles.json"):
    with open(path, 'r') as f:
        return json.load(f)

# Load config and styles
gen_config = load_config()
styles = load_styles()
prompt_templates = PromptTemplates()

# Load SD 1.5 + LoRA pipeline
def load_pipeline():
    model_id = gen_config['model']['base_model']
    lora_path = gen_config['model']['lora_path']
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
    return pipe

pipe = load_pipeline()

# Image generation function
def generate_image(panel_prompt, style, panel_type, guidance_scale, steps, seed):
    # Compose prompt
    pos_prompt, neg_prompt = prompt_templates.generate_prompt(
        style=style,
        panel_type=panel_type,
        character="character",
        emotion="neutral",
        action="standing",
        environment="background",
        lighting="natural lighting",
        additional_details=panel_prompt
    )
    if seed == -1:
        generator = None
    else:
        generator = torch.manual_seed(seed)
    result = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    return result.images[0]

# Gradio UI
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Comic-Gen: AI Comic Panel Generator")
        with gr.Row():
            with gr.Column():
                panel_prompt = gr.Textbox(label="Panel Description / Prompt", lines=4)
                style = gr.Dropdown(list(styles.keys()), value="manga", label="Comic Style")
                panel_type = gr.Dropdown(prompt_templates.get_available_panel_types(), value="close_up", label="Panel Type")
                guidance_scale = gr.Slider(5, 15, value=gen_config['generation']['guidance_scale'], step=0.1, label="Guidance Scale")
                steps = gr.Slider(10, 100, value=gen_config['generation']['num_inference_steps'], step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")
                generate_btn = gr.Button("Generate Panel")
            with gr.Column():
                output_image = gr.Image(label="Generated Panel")
        generate_btn.click(
            generate_image, 
            inputs=[panel_prompt, style, panel_type, guidance_scale, steps, seed],
            outputs=output_image
        )
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch() 