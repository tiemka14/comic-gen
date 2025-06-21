"""
Prompt Templates and Style Management for Comic Generation
Handles comic styles, panel types, and prompt generation
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class ComicStyle:
    """Represents a comic art style with its characteristics"""
    name: str
    description: str
    base_prompt: str
    negative_prompt: str
    characteristics: List[str]
    color_palette: str
    line_style: str


@dataclass
class PanelType:
    """Represents a panel type with its composition rules"""
    name: str
    description: str
    prompt_template: str
    composition: str
    use_cases: List[str]
    camera_angle: str
    background: str


class PromptTemplates:
    """Manages prompt templates and comic styles"""
    
    def __init__(self, styles_path: str = "templates/comic_styles.json", 
                 panel_types_path: str = "templates/panel_types.json"):
        """
        Initialize prompt templates with style and panel type definitions
        
        Args:
            styles_path: Path to comic styles JSON file
            panel_types_path: Path to panel types JSON file
        """
        self.styles_path = styles_path
        self.panel_types_path = panel_types_path
        self.styles = {}
        self.panel_types = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load comic styles and panel types from JSON files"""
        try:
            # Load comic styles
            if os.path.exists(self.styles_path):
                with open(self.styles_path, 'r', encoding='utf-8') as f:
                    styles_data = json.load(f)
                    for style_name, style_data in styles_data.items():
                        self.styles[style_name] = ComicStyle(
                            name=style_name,
                            description=style_data.get('description', ''),
                            base_prompt=style_data.get('base_prompt', ''),
                            negative_prompt=style_data.get('negative_prompt', ''),
                            characteristics=style_data.get('characteristics', []),
                            color_palette=style_data.get('color_palette', ''),
                            line_style=style_data.get('line_style', '')
                        )
            
            # Load panel types
            if os.path.exists(self.panel_types_path):
                with open(self.panel_types_path, 'r', encoding='utf-8') as f:
                    panel_data = json.load(f)
                    for panel_name, panel_info in panel_data.items():
                        self.panel_types[panel_name] = PanelType(
                            name=panel_name,
                            description=panel_info.get('description', ''),
                            prompt_template=panel_info.get('prompt_template', ''),
                            composition=panel_info.get('composition', ''),
                            use_cases=panel_info.get('use_cases', []),
                            camera_angle=panel_info.get('camera_angle', ''),
                            background=panel_info.get('background', '')
                        )
                        
        except Exception as e:
            print(f"Error loading templates: {e}")
            # Load default templates if files don't exist
            self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default templates if JSON files are not available"""
        # Default comic styles
        self.styles = {
            "manga": ComicStyle(
                name="manga",
                description="Japanese manga style with distinctive features",
                base_prompt="manga style, anime style, cel shading, clean lines, expressive eyes, dynamic poses",
                negative_prompt="realistic, photorealistic, 3d render, western comic, european comic",
                characteristics=["large expressive eyes", "spiky or flowing hair", "clean line art"],
                color_palette="vibrant, saturated colors",
                line_style="clean, bold lines"
            ),
            "western": ComicStyle(
                name="western",
                description="American comic book style with bold colors and dramatic lighting",
                base_prompt="comic book style, american comic, bold colors, dramatic lighting, superhero style",
                negative_prompt="manga, anime, realistic, photorealistic, european comic",
                characteristics=["muscular, heroic proportions", "bold, dramatic shadows", "vibrant primary colors"],
                color_palette="bold primary colors with dramatic contrast",
                line_style="thick, bold lines with detailed crosshatching"
            )
        }
        
        # Default panel types
        self.panel_types = {
            "close_up": PanelType(
                name="close_up",
                description="Close-up shot focusing on character's face or important detail",
                prompt_template="close-up shot, {character} face, {emotion}, detailed facial features, {lighting}",
                composition="tight framing, emphasis on facial expression",
                use_cases=["emotional moments", "character reactions", "important details"],
                camera_angle="straight on or slight angle",
                background="blurred or minimal"
            ),
            "wide_shot": PanelType(
                name="wide_shot",
                description="Wide shot showing full scene and environment",
                prompt_template="wide shot, {character} in {environment}, full body, {background}, establishing shot",
                composition="full scene visible, character in context",
                use_cases=["establishing locations", "action scenes", "group shots"],
                camera_angle="eye level or slightly elevated",
                background="detailed environment"
            )
        }
    
    def get_available_styles(self) -> List[str]:
        """Get list of available comic styles"""
        return list(self.styles.keys())
    
    def get_available_panel_types(self) -> List[str]:
        """Get list of available panel types"""
        return list(self.panel_types.keys())
    
    def get_style(self, style_name: str) -> Optional[ComicStyle]:
        """Get a specific comic style by name"""
        return self.styles.get(style_name)
    
    def get_panel_type(self, panel_type_name: str) -> Optional[PanelType]:
        """Get a specific panel type by name"""
        return self.panel_types.get(panel_type_name)
    
    def generate_prompt(self, 
                       style: str, 
                       panel_type: str, 
                       character: str = "character",
                       emotion: str = "neutral",
                       action: str = "standing",
                       environment: str = "background",
                       lighting: str = "natural lighting",
                       additional_details: str = "",
                       **kwargs) -> Tuple[str, str]:
        """
        Generate positive and negative prompts for a comic panel
        
        Args:
            style: Comic style name
            panel_type: Panel type name
            character: Character description
            emotion: Character emotion
            action: Character action
            environment: Environment description
            lighting: Lighting description
            additional_details: Additional prompt details
            **kwargs: Additional template variables
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        # Get style and panel type
        comic_style = self.get_style(style)
        panel = self.get_panel_type(panel_type)
        
        if not comic_style or not panel:
            raise ValueError(f"Invalid style '{style}' or panel type '{panel_type}'")
        
        # Build positive prompt
        positive_parts = []
        
        # Add panel type template
        panel_prompt = panel.prompt_template.format(
            character=character,
            emotion=emotion,
            action=action,
            environment=environment,
            lighting=lighting,
            **kwargs
        )
        positive_parts.append(panel_prompt)
        
        # Add style base prompt
        positive_parts.append(comic_style.base_prompt)
        
        # Add additional details
        if additional_details:
            positive_parts.append(additional_details)
        
        # Add quality enhancers
        positive_parts.extend([
            "high quality",
            "detailed",
            "professional artwork",
            "comic panel"
        ])
        
        positive_prompt = ", ".join(positive_parts)
        
        # Build negative prompt
        negative_parts = [
            comic_style.negative_prompt,
            "low quality",
            "blurry",
            "distorted",
            "deformed",
            "bad anatomy",
            "bad proportions"
        ]
        
        negative_prompt = ", ".join(negative_parts)
        
        return positive_prompt, negative_prompt
    
    def mix_styles(self, styles: List[str], weights: Optional[List[float]] = None) -> Tuple[str, str]:
        """
        Mix multiple comic styles together
        
        Args:
            styles: List of style names to mix
            weights: Optional weights for each style (must sum to 1.0)
            
        Returns:
            Tuple of (mixed_positive_prompt, mixed_negative_prompt)
        """
        if not styles:
            raise ValueError("At least one style must be provided")
        
        if weights and len(weights) != len(styles):
            raise ValueError("Number of weights must match number of styles")
        
        if not weights:
            weights = [1.0 / len(styles)] * len(styles)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Collect style prompts
        positive_parts = []
        negative_parts = []
        
        for style_name, weight in zip(styles, weights):
            style = self.get_style(style_name)
            if style:
                # Add weighted style prompt
                if weight > 0.5:  # Primary style
                    positive_parts.append(style.base_prompt)
                else:  # Secondary style
                    positive_parts.append(f"({style.base_prompt}:{weight:.2f})")
                
                negative_parts.append(style.negative_prompt)
        
        # Combine prompts
        positive_prompt = ", ".join(positive_parts)
        negative_prompt = ", ".join(negative_parts)
        
        return positive_prompt, negative_prompt
    
    def enhance_prompt(self, prompt: str, enhancement_strength: float = 0.3) -> str:
        """
        Enhance a prompt with additional quality descriptors
        
        Args:
            prompt: Original prompt
            enhancement_strength: How much to enhance (0.0 to 1.0)
            
        Returns:
            Enhanced prompt
        """
        if enhancement_strength <= 0:
            return prompt
        
        # Quality enhancers based on strength
        enhancers = {
            0.1: ["high quality"],
            0.2: ["high quality", "detailed"],
            0.3: ["high quality", "detailed", "professional artwork"],
            0.4: ["high quality", "detailed", "professional artwork", "masterpiece"],
            0.5: ["high quality", "detailed", "professional artwork", "masterpiece", "best quality"]
        }
        
        # Get appropriate enhancers
        strength_key = min(enhancers.keys(), key=lambda x: abs(x - enhancement_strength))
        additional_enhancers = enhancers[strength_key]
        
        # Add enhancers to prompt
        enhanced_parts = [prompt] + additional_enhancers
        return ", ".join(enhanced_parts)
    
    def get_random_style(self) -> str:
        """Get a random comic style"""
        return random.choice(list(self.styles.keys()))
    
    def get_random_panel_type(self) -> str:
        """Get a random panel type"""
        return random.choice(list(self.panel_types.keys()))
    
    def get_style_characteristics(self, style: str) -> List[str]:
        """Get characteristics of a specific style"""
        comic_style = self.get_style(style)
        return comic_style.characteristics if comic_style else []
    
    def get_panel_use_cases(self, panel_type: str) -> List[str]:
        """Get use cases for a specific panel type"""
        panel = self.get_panel_type(panel_type)
        return panel.use_cases if panel else []
    
    def validate_style_combination(self, styles: List[str]) -> bool:
        """Validate if a combination of styles makes sense"""
        if not styles:
            return False
        
        # Check if all styles exist
        for style in styles:
            if style not in self.styles:
                return False
        
        # Check for conflicting styles (optional logic)
        conflicting_pairs = [
            ("manga", "western"),
            ("manga", "european"),
            ("western", "european")
        ]
        
        for style1, style2 in conflicting_pairs:
            if style1 in styles and style2 in styles:
                return False
        
        return True


# Utility functions for common prompt operations
def create_character_prompt(character_name: str, 
                           appearance: str, 
                           clothing: str = "",
                           accessories: str = "") -> str:
    """Create a character description prompt"""
    parts = [character_name, appearance]
    if clothing:
        parts.append(f"wearing {clothing}")
    if accessories:
        parts.append(f"with {accessories}")
    return ", ".join(parts)


def create_environment_prompt(location: str, 
                             time_of_day: str = "",
                             weather: str = "",
                             mood: str = "") -> str:
    """Create an environment description prompt"""
    parts = [location]
    if time_of_day:
        parts.append(time_of_day)
    if weather:
        parts.append(weather)
    if mood:
        parts.append(f"{mood} atmosphere")
    return ", ".join(parts)


def create_action_prompt(action: str, 
                        intensity: str = "normal",
                        direction: str = "",
                        target: str = "") -> str:
    """Create an action description prompt"""
    parts = [action]
    if intensity != "normal":
        parts.append(f"{intensity} intensity")
    if direction:
        parts.append(f"moving {direction}")
    if target:
        parts.append(f"towards {target}")
    return ", ".join(parts)


# Example usage and testing
if __name__ == "__main__":
    # Initialize prompt templates
    templates = PromptTemplates()
    
    # Example: Generate a manga close-up prompt
    positive, negative = templates.generate_prompt(
        style="manga",
        panel_type="close_up",
        character="young hero with spiky hair",
        emotion="determined",
        lighting="dramatic side lighting",
        additional_details="epic battle scene"
    )
    
    print("Generated Prompts:")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    
    # Example: Mix styles
    mixed_pos, mixed_neg = templates.mix_styles(
        ["manga", "western"], 
        weights=[0.7, 0.3]
    )
    
    print(f"\nMixed Style Prompt: {mixed_pos}")
    
    # Example: Get available options
    print(f"\nAvailable styles: {templates.get_available_styles()}")
    print(f"Available panel types: {templates.get_available_panel_types()}") 