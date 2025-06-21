"""
Concise Multimodal Identity Extractor using LangChain + Mistral AI
================================================================

A streamlined version that analyzes logo images, PDF documents, and persona text
to extract brand identity insights using LangChain and Mistral AI.
"""

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import json
import base64
from typing import Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import colorsys
from collections import Counter

class ConciseIdentityExtractor:
    def __init__(self, mistral_api_key: str):
        """Initialize with LangChain Mistral integration."""
        self.llm = ChatMistralAI(
            api_key=mistral_api_key,
            model="mistral-large-latest",
            temperature=0.3
        )
        self.json_parser = JsonOutputParser()
        
    def extract_logo_features(self, image_path: str) -> Dict[str, Any]:
        """Extract basic visual features from logo."""
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        
        # Extract dominant colors
        colors = self._get_dominant_colors(image)
        
        # Basic visual properties
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        complexity = np.sum(edges > 0) / edges.size
        
        return {
            "dominant_colors": colors[:3],
            "complexity_score": float(complexity),
            "aspect_ratio": pil_image.width / pil_image.height,
            "size": {"width": pil_image.width, "height": pil_image.height}
        }
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and basic info from PDF."""
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page in doc:
            text_content.append(page.get_text())
        
        full_text = " ".join(text_content)
        return {
            "text": full_text[:2000],  # Limit for API
            "page_count": len(doc),
            "word_count": len(full_text.split())
        }
    
    def analyze_with_llm(self, logo_features: Dict, pdf_content: Dict, persona_text: str) -> Dict[str, Any]:
        """Use LangChain + Mistral to analyze all inputs and generate identity vector."""
        
        # Create analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a brand identity expert. Analyze the provided multimodal data and create a structured brand identity vector. Return ONLY valid JSON with no additional text."""),
            HumanMessage(content=f"""
            Analyze this brand data and create an identity vector:

            LOGO ANALYSIS:
            - Colors: {logo_features['dominant_colors']}
            - Complexity: {logo_features['complexity_score']:.2f}
            - Aspect ratio: {logo_features['aspect_ratio']:.2f}

            DOCUMENT CONTENT:
            - Text sample: {pdf_content['text'][:500]}...
            - Pages: {pdf_content['page_count']}
            - Words: {pdf_content['word_count']}

            PERSONA:
            {persona_text}

            Generate a JSON identity vector with this exact structure:
            {{
                "brand_personality": {{
                    "primary_traits": ["trait1", "trait2"],
                    "tone": "professional/friendly/bold/elegant"
                }},
                "visual_identity": {{
                    "color_palette": ["#hex1", "#hex2", "#hex3"],
                    "style": "minimalist/modern/classic/bold",
                    "complexity": "low/medium/high"
                }},
                "target_audience": {{
                    "demographics": "brief description",
                    "preferences": ["pref1", "pref2"]
                }},
                "recommendations": {{
                    "primary_colors": ["#hex1", "#hex2"],
                    "typography": "recommendation",
                    "layout": "recommendation"
                }},
                "confidence": {{
                    "overall": 0.85
                }}
            }}
            """)
        ])
        
        # Create chain
        chain = prompt | self.llm | self.json_parser
        
        try:
            result = chain.invoke({})
            return result
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._create_fallback_vector(logo_features, pdf_content, persona_text)
    
    def process_multimodal_input(self, logo_path: str, pdf_path: str, persona_text: str) -> Dict[str, Any]:
        """Main processing pipeline."""
        print("🔍 Extracting logo features...")
        logo_features = self.extract_logo_features(logo_path)
        
        print("📄 Extracting PDF content...")
        pdf_content = self.extract_pdf_content(pdf_path)
        
        print("🤖 Analyzing with Mistral AI...")
        identity_vector = self.analyze_with_llm(logo_features, pdf_content, persona_text)
        
        # Add metadata
        identity_vector["metadata"] = {
            "input_sources": ["logo", "pdf", "persona"],
            "model": "mistral-large-latest",
            "framework": "langchain"
        }
        
        return identity_vector
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> list:
        """Extract dominant colors using K-means."""
        data = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        colors = []
        for center in centers:
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(center[2]), int(center[1]), int(center[0])  # BGR to RGB
            )
            colors.append(hex_color)
        
        return colors
    
    def _create_fallback_vector(self, logo_features: Dict, pdf_content: Dict, persona_text: str) -> Dict:
        """Fallback identity vector if LLM fails."""
        return {
            "brand_personality": {
                "primary_traits": ["professional", "modern"],
                "tone": "balanced"
            },
            "visual_identity": {
                "color_palette": logo_features['dominant_colors'],
                "style": "modern" if logo_features['complexity_score'] < 0.3 else "detailed",
                "complexity": "low" if logo_features['complexity_score'] < 0.2 else "medium"
            },
            "target_audience": {
                "demographics": "Modern consumers",
                "preferences": ["clean design", "functionality"]
            },
            "recommendations": {
                "primary_colors": logo_features['dominant_colors'][:2],
                "typography": "Clean and readable",
                "layout": "Structured and organized"
            },
            "confidence": {
                "overall": 0.6
            }
        }

# Usage Example
def main():
    """Example usage of the concise extractor."""
    # Initialize with your Mistral API key
    extractor = ConciseIdentityExtractor(mistral_api_key="Os5Djkw6AcqVg9mSkltWsV4JiFbQ7lac")
    
    # File paths
    logo_path = "C:/Users/Pavan/Desktop/batman-laptop-ciluigv1xjrqmfgk.jpg"
    pdf_path = "C:/Users/Pavan/Desktop/ZCAIMLPL02.pdf"
    persona_text = """
    Dark, gothic superhero theme with high-tech elements. Embodies justice, fear, and urban vigilance in a dystopian city.
    """
    
    try:
        # Process inputs
        result = extractor.process_multimodal_input(logo_path, pdf_path, persona_text)
        
        # Display results
        print("\n" + "="*50)
        print("BRAND IDENTITY VECTOR")
        print("="*50)
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
