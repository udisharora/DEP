# ALPR Pipeline Upgrades (Session Summary)

This document summarizes the architectural overhauls, modeling upgrades, and algorithmic logic integrated into the Automatic License Plate Recognition pipeline during this session.

## 1. Core Model Upgrades
### TrOCR Migration
*   **EasyOCR Deprecation:** Legacy OpenCV-based EasyOCR implementation was stripped from the execution pipeline entirely.
*   **Microsoft TrOCR Integration:** Integrated the heavy-weight `microsoft/trocr-base-printed` Transformer model natively into the pipeline via HuggingFace's `transformers` library, heavily improving text extraction on blurry/distant objects.
*   **Pipeline Simplification:** Removed the legacy CNN Condition Classifier and deprecated initial Grayscale preprocessing requirements to feed clean, vibrant crops to the Transformer.

## 2. Image Preprocessing & Accuracy Tuning
### AI Super-Resolution (Swin2SR)
*   Integrated Microsoft's **Swin2SR** (`caidas/swin2sr-classical-sr-x2-64`) algorithm via generating `modules/super_resolution.py`.
*   The tight bounding box crops from YOLOv8 are now physically upscaled dynamically by 2x using Deep Learning (hallucinating/sharpening edge lines algorithmically before hitting TrOCR for a reading).
*   **Box Context Expansion:** Modified bounding boxing mechanisms to purposefully expand the YOLOv8 coordinates by `5%` natively prior to snapping the image, specifically preventing tight border limitations from physically chopping edge-characters (like trailing terminal `8`s).

## 3. Strict Extraction Formatting (Indian Standards)
### Heuristic Forcing & Repair Algorithms
*   Built a strict alphanumeric heuristic sequence inside `modules/ocr_engine.py` exclusively tailored for modern Indian licensing standards.
*   It intercepts TrOCR strings and brute-forces mathematical replacements on individual characters (`Z` <-> `2`, `O` <-> `0`, `B` <-> `8`) using Dictionary configurations based solely on standard positional index expectations.
*   **Dynamic Structure Logic:** The algorithm natively handles dynamic length inputs (`8`, `9`, or `10`-character crops due to camera dropouts) and algorithmically pads them back into a strict visual render of `2-2-2-4` standard blocks (e.g. `MP 04 CC 2688`).

## 4. Test-Time Augmentation (TTA) Ensemble System
### Multi-Padding Majority Validator
*   Upgraded the standard OCR process in `app.py` from a fragile single point-of-failure read to a heavy ensemble iteration system.
*   **10-Scale Variant Creation:** The plate image generates 10 distinctly different padded sizes (ranging smoothly from a `0%` tight-crop to an `18%` padded margin context border).
*   **The Execution Pipeline:** Every single padded instance successfully gets shoved individually through the full `NAFNet` (Deblur) -> `Swin2SR` (Upscale) -> `TrOCR` (Text Output) block algorithm. 
*   **Majority Vote Protocol:** The overarching pipeline runs all finalized predictions through Python's `collections.Counter` to isolate and strictly select the "Mode" (the string layout explicitly extracted the most amount of times out of the 10 passes). This entirely masks out isolated optical illusions or single-pass inaccuracies organically.

## 5. UI Enhancements & Environment Tools
### Streamlit & Analytics Integrations
*   **RTO Location Parsing:** Established `modules/rto_metadata.py` containing District maps of 36+ regions parsing the dynamic state codes mapped directly to the Streamlit UI, immediately displaying the exact locality origin of the vehicle post-OCR.
*   **Ensemble Visualizer Grids:** Pushed an `st.expander` visual matrix actively into the bottom of the Streamlit output flow, detailing a 2-Row grid of all 10 cropped size-configurations arrayed exactly above their individualized text. This physically surfaces the visual evidence to the end user regarding the accuracy backing the "Majority Vote"!
