#!/usr/bin/env python3
"""
Script to Panel Processor
Converts text comic scripts into structured panel prompts for AI generation
"""

import json
import re
import argparse
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from prompt_templates import PromptTemplates, create_character_prompt, create_environment_prompt, create_action_prompt


@dataclass
class Panel:
    """Represents a single comic panel"""
    panel_number: int
    panel_type: str
    style: str
    positive_prompt: str
    negative_prompt: str
    description: str
    characters: List[str]
    environment: str
    action: str
    emotion: str
    dialogue: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComicScript:
    """Represents a complete comic script"""
    title: str
    author: str
    panels: List[Panel]
    metadata: Dict[str, Any]


class ScriptParser:
    """Parses text scripts into structured panel data"""
    
    def __init__(self):
        """Initialize the script parser"""
        self.prompt_templates = PromptTemplates()
        self.logger = self._setup_logging()
        
        # Common patterns for script parsing
        self.panel_pattern = re.compile(r'PANEL\s+(\d+):?\s*(.*?)(?=PANEL\s+\d+:|$)', re.DOTALL | re.IGNORECASE)
        self.dialogue_pattern = re.compile(r'([A-Z][A-Z\s]+):\s*"([^"]*)"')
        self.action_pattern = re.compile(r'\[([^\]]+)\]')
        self.character_pattern = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def parse_script(self, script_text: str, default_style: str = "manga") -> ComicScript:
        """
        Parse a text script into a structured ComicScript object
        
        Args:
            script_text: Raw script text
            default_style: Default comic style to use
            
        Returns:
            ComicScript object with parsed panels
        """
        self.logger.info("Starting script parsing...")
        
        # Extract basic script information
        title = self._extract_title(script_text)
        author = self._extract_author(script_text)
        
        # Parse panels
        panels = []
        panel_matches = self.panel_pattern.findall(script_text)
        
        for panel_num, panel_content in panel_matches:
            panel = self._parse_panel(int(panel_num), panel_content, default_style)
            if panel:
                panels.append(panel)
        
        # Create metadata
        metadata = {
            "total_panels": len(panels),
            "styles_used": list(set(p.style for p in panels)),
            "panel_types_used": list(set(p.panel_type for p in panels)),
            "characters": list(set(char for p in panels for char in p.characters))
        }
        
        return ComicScript(
            title=title,
            author=author,
            panels=panels,
            metadata=metadata
        )
    
    def _extract_title(self, script_text: str) -> str:
        """Extract script title from the text"""
        # Look for title patterns
        title_patterns = [
            r'TITLE:\s*(.+)',
            r'Title:\s*(.+)',
            r'#\s*(.+)',
            r'^(.+?)\n'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, script_text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return "Untitled Comic"
    
    def _extract_author(self, script_text: str) -> str:
        """Extract script author from the text"""
        author_patterns = [
            r'BY:\s*(.+)',
            r'Author:\s*(.+)',
            r'Written by:\s*(.+)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, script_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Author"
    
    def _parse_panel(self, panel_number: int, panel_content: str, default_style: str) -> Optional[Panel]:
        """
        Parse a single panel's content
        
        Args:
            panel_number: Panel number
            panel_content: Raw panel content
            default_style: Default comic style
            
        Returns:
            Panel object or None if parsing fails
        """
        try:
            # Extract dialogue
            dialogue = self._extract_dialogue(panel_content)
            
            # Extract action descriptions
            actions = self._extract_actions(panel_content)
            
            # Extract characters
            characters = self._extract_characters(panel_content)
            
            # Determine panel type
            panel_type = self._determine_panel_type(panel_content, actions, characters)
            
            # Determine style
            style = self._determine_style(panel_content, default_style)
            
            # Generate prompts
            positive_prompt, negative_prompt = self._generate_panel_prompts(
                panel_content, panel_type, style, characters, actions
            )
            
            # Create description
            description = self._create_panel_description(panel_content, actions, characters)
            
            # Extract environment
            environment = self._extract_environment(panel_content)
            
            # Extract emotion
            emotion = self._extract_emotion(panel_content, dialogue)
            
            # Create metadata
            metadata = {
                "raw_content": panel_content.strip(),
                "parsed_actions": actions,
                "parsed_characters": characters,
                "dialogue_count": len(dialogue) if dialogue else 0
            }
            
            return Panel(
                panel_number=panel_number,
                panel_type=panel_type,
                style=style,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                description=description,
                characters=characters,
                environment=environment,
                action=", ".join(actions) if actions else "standing",
                emotion=emotion,
                dialogue=dialogue,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing panel {panel_number}: {e}")
            return None
    
    def _extract_dialogue(self, panel_content: str) -> Optional[str]:
        """Extract dialogue from panel content"""
        dialogue_matches = self.dialogue_pattern.findall(panel_content)
        if dialogue_matches:
            # Combine all dialogue
            dialogue_parts = []
            for speaker, text in dialogue_matches:
                dialogue_parts.append(f"{speaker.strip()}: \"{text}\"")
            return " | ".join(dialogue_parts)
        return None
    
    def _extract_actions(self, panel_content: str) -> List[str]:
        """Extract action descriptions from panel content"""
        actions = []
        
        # Look for bracketed actions
        action_matches = self.action_pattern.findall(panel_content)
        actions.extend(action_matches)
        
        # Look for action keywords
        action_keywords = [
            'running', 'walking', 'jumping', 'fighting', 'sitting', 'standing',
            'pointing', 'looking', 'smiling', 'frowning', 'crying', 'laughing',
            'attacking', 'defending', 'escaping', 'chasing', 'hiding'
        ]
        
        content_lower = panel_content.lower()
        for keyword in action_keywords:
            if keyword in content_lower:
                actions.append(keyword)
        
        return list(set(actions))  # Remove duplicates
    
    def _extract_characters(self, panel_content: str) -> List[str]:
        """Extract character names from panel content"""
        characters = []
        
        # Look for character patterns
        char_matches = self.character_pattern.findall(panel_content)
        for char in char_matches:
            # Filter out common non-character words
            if len(char.split()) <= 3 and char.lower() not in ['the', 'and', 'or', 'but', 'with', 'from']:
                characters.append(char)
        
        # Also look for dialogue speakers
        dialogue_matches = self.dialogue_pattern.findall(panel_content)
        for speaker, _ in dialogue_matches:
            characters.append(speaker.strip())
        
        return list(set(characters))  # Remove duplicates
    
    def _determine_panel_type(self, content: str, actions: List[str], characters: List[str]) -> str:
        """Determine the appropriate panel type based on content"""
        content_lower = content.lower()
        
        # Check for specific panel type indicators
        if any(word in content_lower for word in ['close-up', 'closeup', 'face', 'expression']):
            return 'close_up'
        elif any(word in content_lower for word in ['wide', 'establishing', 'landscape']):
            return 'establishing'
        elif any(word in content_lower for word in ['action', 'fighting', 'running', 'jumping']):
            return 'action'
        elif len(characters) > 1 and any('dialogue' in content_lower or 'speaking' in content_lower):
            return 'dialogue'
        elif any(word in content_lower for word in ['splash', 'epic', 'dramatic']):
            return 'splash'
        else:
            # Default based on content analysis
            if len(actions) > 2:
                return 'action'
            elif len(characters) > 1:
                return 'dialogue'
            else:
                return 'wide_shot'
    
    def _determine_style(self, content: str, default_style: str) -> str:
        """Determine comic style based on content"""
        content_lower = content.lower()
        
        # Check for style indicators
        if any(word in content_lower for word in ['manga', 'anime', 'japanese']):
            return 'manga'
        elif any(word in content_lower for word in ['western', 'american', 'superhero']):
            return 'western'
        elif any(word in content_lower for word in ['european', 'belgian', 'french']):
            return 'european'
        elif any(word in content_lower for word in ['webcomic', 'digital']):
            return 'webcomic'
        elif any(word in content_lower for word in ['vintage', 'retro', 'classic']):
            return 'vintage'
        elif any(word in content_lower for word in ['noir', 'dark', 'mystery']):
            return 'noir'
        else:
            return default_style
    
    def _generate_panel_prompts(self, content: str, panel_type: str, style: str, 
                               characters: List[str], actions: List[str]) -> tuple[str, str]:
        """Generate positive and negative prompts for the panel"""
        # Create character description
        character_desc = ", ".join(characters) if characters else "character"
        
        # Create action description
        action_desc = ", ".join(actions) if actions else "standing"
        
        # Create environment description
        environment = self._extract_environment(content)
        
        # Create emotion description
        emotion = self._extract_emotion(content, None)
        
        # Generate prompts using templates
        return self.prompt_templates.generate_prompt(
            style=style,
            panel_type=panel_type,
            character=character_desc,
            emotion=emotion,
            action=action_desc,
            environment=environment,
            lighting="natural lighting",
            additional_details=content.strip()
        )
    
    def _extract_environment(self, content: str) -> str:
        """Extract environment description from content"""
        # Look for environment keywords
        env_keywords = {
            'indoor': ['room', 'house', 'building', 'office', 'school', 'hospital'],
            'outdoor': ['street', 'park', 'forest', 'mountain', 'beach', 'city'],
            'urban': ['city', 'street', 'alley', 'building', 'skyscraper'],
            'nature': ['forest', 'mountain', 'river', 'ocean', 'desert', 'jungle']
        }
        
        content_lower = content.lower()
        for env_type, keywords in env_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return env_type
        
        return "background"
    
    def _extract_emotion(self, content: str, dialogue: Optional[str]) -> str:
        """Extract emotion from content and dialogue"""
        emotion_keywords = {
            'happy': ['smile', 'laugh', 'joy', 'happy', 'excited'],
            'sad': ['cry', 'sad', 'tears', 'depressed', 'melancholy'],
            'angry': ['angry', 'rage', 'furious', 'mad', 'frown'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'fearful': ['fear', 'scared', 'terrified', 'afraid', 'panic'],
            'determined': ['determined', 'focused', 'resolute', 'confident']
        }
        
        content_lower = content.lower()
        if dialogue:
            content_lower += " " + dialogue.lower()
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def _create_panel_description(self, content: str, actions: List[str], characters: List[str]) -> str:
        """Create a human-readable description of the panel"""
        parts = []
        
        if characters:
            parts.append(f"Characters: {', '.join(characters)}")
        
        if actions:
            parts.append(f"Actions: {', '.join(actions)}")
        
        # Add clean content description
        clean_content = re.sub(r'\[[^\]]*\]', '', content)  # Remove bracketed actions
        clean_content = re.sub(r'[A-Z][A-Z\s]+:\s*"[^"]*"', '', clean_content)  # Remove dialogue
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()  # Clean whitespace
        
        if clean_content:
            parts.append(f"Description: {clean_content}")
        
        return " | ".join(parts)


class ScriptProcessor:
    """Main processor for converting scripts to panel prompts"""
    
    def __init__(self):
        """Initialize the script processor"""
        self.parser = ScriptParser()
        self.logger = self.parser.logger
    
    def process_script_file(self, input_path: str, output_path: str, 
                           default_style: str = "manga") -> bool:
        """
        Process a script file and save the results
        
        Args:
            input_path: Path to input script file
            output_path: Path to output JSON file
            default_style: Default comic style
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                script_text = f.read()
            
            # Parse script
            comic_script = self.parser.parse_script(script_text, default_style)
            
            # Save results
            self._save_results(comic_script, output_path)
            
            self.logger.info(f"Successfully processed {len(comic_script.panels)} panels")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing script: {e}")
            return False
    
    def _save_results(self, comic_script: ComicScript, output_path: str):
        """Save the processed script results to JSON"""
        # Convert to dictionary format
        output_data = {
            "title": comic_script.title,
            "author": comic_script.author,
            "metadata": comic_script.metadata,
            "panels": []
        }
        
        for panel in comic_script.panels:
            panel_data = asdict(panel)
            output_data["panels"].append(panel_data)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def create_sample_script(self, output_path: str):
        """Create a sample script for testing"""
        sample_script = """TITLE: The Hero's Journey
BY: Comic Generator

PANEL 1: [Establishing shot of a small town at dawn]
A peaceful village nestled in the mountains, with smoke rising from chimneys.

PANEL 2: [Close-up of our hero, a young warrior]
YOUNG HERO: "Today is the day I begin my quest."

PANEL 3: [Wide shot of the hero walking down the village street]
The hero walks confidently through the village, villagers watching with curiosity.

PANEL 4: [Action shot of the hero drawing his sword]
YOUNG HERO: "I will protect this village from the darkness!"

PANEL 5: [Dramatic splash panel of the hero facing a dark forest]
The hero stands at the edge of a mysterious forest, sword drawn, ready for adventure.
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sample_script)
        
        self.logger.info(f"Sample script created at {output_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Convert comic scripts to panel prompts")
    parser.add_argument("--input", "-i", required=True, help="Input script file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--style", "-s", default="manga", 
                       choices=["manga", "western", "european", "webcomic", "vintage", "noir"],
                       help="Default comic style")
    parser.add_argument("--create-sample", action="store_true", 
                       help="Create a sample script file")
    
    args = parser.parse_args()
    
    processor = ScriptProcessor()
    
    if args.create_sample:
        processor.create_sample_script(args.input)
        print(f"Sample script created at {args.input}")
        return
    
    # Process the script
    success = processor.process_script_file(args.input, args.output, args.style)
    
    if success:
        print(f"Successfully processed script: {args.input}")
        print(f"Results saved to: {args.output}")
    else:
        print("Error processing script")
        exit(1)


if __name__ == "__main__":
    main() 