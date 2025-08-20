#!/usr/bin/env python3
"""
ROM Font Extractor - Extract character templates from Pokemon ROM memory

This module is responsible for extracting font templates directly from the game ROM,
which provides more accurate text recognition than generic OCR approaches.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FontTemplate:
    """Font template metadata"""
    char: str
    width: int
    height: int
    data: np.ndarray
    offset: int

class ROMFontExtractor:
    """Extract and manage Pokemon font templates from ROM data"""
    
    def __init__(self, rom_path: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize the font extractor
        
        Args:
            rom_path: Path to Pokemon Crystal ROM file
            output_dir: Directory to save extracted templates
        """
        self.rom_path = rom_path
        self.output_dir = output_dir or os.path.dirname(__file__)
        self.templates: Dict[str, FontTemplate] = {}
        
        # Font metadata
        self.font_table_address = 0x144000  # Default location in Crystal ROM
        self.char_width = 8  # Default width for most characters
        self.char_height = 8  # Default height for Pokemon font
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        if rom_path:
            self.load_rom()
    
    def load_rom(self) -> bool:
        """Load ROM file and verify it's Pokemon Crystal
        
        Returns:
            bool: True if ROM loaded successfully
        """
        if not self.rom_path or not os.path.exists(self.rom_path):
            self.logger.error(f"ROM file not found: {self.rom_path}")
            return False
        
        try:
            with open(self.rom_path, 'rb') as f:
                self.rom_data = f.read()
            
            # Verify ROM (Crystal specific header check)
            if len(self.rom_data) < 0x150 or self.rom_data[0x134:0x143] != b"POKEMON CRYSTAL":
                self.logger.error("Invalid Pokemon Crystal ROM")
                return False
            
            self.logger.info(f"Loaded ROM: {self.rom_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ROM: {e}")
            return False
    
    def extract_font_table(self) -> Dict[str, FontTemplate]:
        """Extract font table from ROM
        
        Returns:
            Dictionary mapping characters to their font templates
        """
        if not hasattr(self, 'rom_data'):
            self.logger.error("ROM not loaded")
            return {}
        
        templates = {}
        try:
            # Character mapping table
            char_map = {
                # Basic ASCII
                i: chr(i) for i in range(0x20, 0x7F)
            }
            char_map.update({
                # Pokemon-specific characters
                0x7F: '⯨',  # Pokeball
                0x80: '⬆️',  # Up arrow
                0x81: '⬇️',  # Down arrow
                0x82: '⬅️',  # Left arrow
                0x83: '➡️',  # Right arrow
                # Add more special characters as needed
            })
            
            # Extract each character's font data
            for char_index, char in char_map.items():
                offset = self.font_table_address + (char_index * self.char_height)
                
                if offset + self.char_height > len(self.rom_data):
                    break
                
                # Extract bit pattern for character
                char_data = []
                for y in range(self.char_height):
                    byte = self.rom_data[offset + y]
                    row = [(byte >> (7-x)) & 1 for x in range(8)]
                    char_data.append(row)
                
                # Create template
                template = FontTemplate(
                    char=char,
                    width=self.char_width,
                    height=self.char_height,
                    data=np.array(char_data, dtype=np.uint8),
                    offset=offset
                )
                
                templates[char] = template
            
            self.templates = templates
            self.logger.info(f"Extracted {len(templates)} font templates")
            return templates
            
        except Exception as e:
            self.logger.error(f"Error extracting font table: {e}")
            return {}
    
    def save_templates(self, output_path: Optional[str] = None) -> bool:
        """Save extracted templates to JSON file
        
        Args:
            output_path: Path to save templates JSON, defaults to output_dir/font_templates.json
            
        Returns:
            bool: True if save successful
        """
        if not self.templates:
            self.logger.error("No templates to save")
            return False
        
        try:
            if not output_path:
                output_path = os.path.join(self.output_dir, 'font_templates.json')
            
            # Convert templates to serializable format
            template_data = {}
            for char, template in self.templates.items():
                template_data[char] = {
                    'width': template.width,
                    'height': template.height,
                    'data': template.data.tolist(),
                    'offset': template.offset
                }
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.templates)} templates to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving templates: {e}")
            return False
    
    def load_templates(self, template_path: Optional[str] = None) -> bool:
        """Load templates from JSON file
        
        Args:
            template_path: Path to templates JSON file
            
        Returns:
            bool: True if load successful
        """
        if not template_path:
            template_path = os.path.join(self.output_dir, 'font_templates.json')
        
        try:
            if not os.path.exists(template_path):
                self.logger.error(f"Template file not found: {template_path}")
                return False
            
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Convert back to FontTemplate objects
            self.templates = {}
            for char, data in template_data.items():
                template = FontTemplate(
                    char=char,
                    width=data['width'],
                    height=data['height'],
                    data=np.array(data['data'], dtype=np.uint8),
                    offset=data['offset']
                )
                self.templates[char] = template
            
            self.logger.info(f"Loaded {len(self.templates)} templates from {template_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
            return False
    
    def get_template(self, char: str) -> Optional[np.ndarray]:
        """Get font template for a character
        
        Args:
            char: Character to get template for
            
        Returns:
            numpy array of template or None if not found
        """
        if char in self.templates:
            return self.templates[char].data
        return None
    
    def validate_templates(self) -> List[str]:
        """Validate loaded templates
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.templates:
            errors.append("No templates loaded")
            return errors
        
        # Check basic ASCII coverage
        ascii_range = set(chr(i) for i in range(0x20, 0x7F))
        missing_ascii = ascii_range - set(self.templates.keys())
        if missing_ascii:
            errors.append(f"Missing ASCII characters: {sorted(missing_ascii)}")
        
        # Validate template dimensions
        for char, template in self.templates.items():
            if template.data.shape != (self.char_height, self.char_width):
                errors.append(
                    f"Invalid dimensions for '{char}': "
                    f"expected ({self.char_height}, {self.char_width}), "
                    f"got {template.data.shape}"
                )
        
        return errors
