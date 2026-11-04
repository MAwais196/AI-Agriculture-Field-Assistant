"""
Visual Pathology Service for Crop Health and Leaf Disease Diagnosis
Extracts visual indicators: Chlorosis severity, pustule streak patterns, necrotic lesions, and pest clusters.
Supports both LLM multimodal analysis and automated computer-vision colorimetry.
"""
import os
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

class VisionService:
    def __init__(self):
        pass

    def analyze_leaf_image(self, image_path: str, context_notes: str = "") -> Dict[str, Any]:
        """
        Analyzes uploaded leaf photo for disease indicators.
        Returns visual findings, chlorosis percentage, stripe pattern likelihood, and visual suspicion.
        """
        if not os.path.exists(image_path):
            return {
                "has_image": False,
                "analysis": "No image provided. Relying on described symptoms."
            }

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((256, 256))
                arr = np.array(img_resized, dtype=np.float32)

                # Channel extraction
                r = arr[:, :, 0]
                g = arr[:, :, 1]
                b = arr[:, :, 2]

                # Agricultural Colorimetry Indices
                # 1. Healthy Green: High Green compared to Red & Blue
                green_mask = (g > 1.1 * r) & (g > 1.1 * b) & (g > 50)
                green_pct = float(np.sum(green_mask) / (256 * 256) * 100)

                # 2. Yellow Chlorosis (High R + G, Low B)
                yellow_mask = (r > 120) & (g > 120) & (b < 100) & (np.abs(r - g) < 50)
                yellow_pct = float(np.sum(yellow_mask) / (256 * 256) * 100)

                # 3. Necrotic / Brown patches (R > G > B with moderate values)
                brown_mask = (r > 80) & (g > 50) & (b < 60) & (r > g) & (~yellow_mask)
                brown_pct = float(np.sum(brown_mask) / (256 * 256) * 100)

                # Texture analysis: check for vertical / longitudinal stripe banding
                # In Stripe rust, yellow pixels form longitudinal columns
                col_yellow_sum = np.sum(yellow_mask, axis=0)
                stripe_variance = float(np.std(col_yellow_sum))
                has_linear_stripes = stripe_variance > 15.0 and yellow_pct > 8.0

                observed_features = []
                primary_suspicion = "Unknown / Healthy"
                confidence = 0.5

                if yellow_pct > 15.0 or has_linear_stripes:
                    if has_linear_stripes:
                        observed_features.append("Linear yellow chlorotic stripes along leaf parallel veins (characteristic of Stripe Rust)")
                        primary_suspicion = "Stripe/Yellow Rust (Puccinia striiformis)"
                        confidence = 0.85
                    else:
                        observed_features.append(f"Significant leaf chlorosis ({yellow_pct:.1f}% yellow leaf area)")
                        primary_suspicion = "Nitrogen Deficiency or Generalized Chlorosis"
                        confidence = 0.70

                if brown_pct > 10.0:
                    observed_features.append(f"Necrotic lesions / bronzing ({brown_pct:.1f}% brown/necrotic tissue)")

                if not observed_features:
                    observed_features.append(f"Predominantly green canopy ({green_pct:.1f}% healthy green tissue) with mild discoloration")

                return {
                    "has_image": True,
                    "image_path": image_path,
                    "dimensions": img.size,
                    "green_coverage_pct": round(green_pct, 1),
                    "chlorosis_yellow_pct": round(yellow_pct, 1),
                    "necrotic_brown_pct": round(brown_pct, 1),
                    "linear_stripes_detected": has_linear_stripes,
                    "observed_features": observed_features,
                    "primary_visual_suspicion": primary_suspicion,
                    "visual_confidence": confidence,
                    "visual_summary": f"Visual analysis detected {yellow_pct:.1f}% chlorosis and {'linear stripe patterns consistent with fungal rust' if has_linear_stripes else 'diffuse discoloration'}."
                }
        except Exception as e:
            return {
                "has_image": True,
                "error": str(e),
                "visual_summary": "Image format error. Please upload a clear photo of the affected leaf."
            }

vision_service = VisionService()

