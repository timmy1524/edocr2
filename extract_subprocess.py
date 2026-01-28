#!/usr/bin/env python
"""
Standalone AI extraction script - runs in fresh Python process.
This ensures TensorFlow/Keras state is clean, matching test_drawing.py behavior.

Usage: python extract_subprocess.py <image_path> [output_json_path]
"""

import cv2
import time
import os
import sys
import json
import numpy as np

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python extract_subprocess.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"success": False, "error": f"Image not found: {image_path}"}))
        sys.exit(1)
    
    try:
        # Suppress TensorFlow warnings before import
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Import AI engine (fresh process = fresh imports)
        # Debug: show where modules are coming from
        print(f"DEBUG: CWD = {os.getcwd()}", file=sys.stderr)
        print(f"DEBUG: sys.path[0] = {sys.path[0]}", file=sys.stderr)
        
        from edocr2 import tools
        from edocr2.keras_ocr.recognition import Recognizer
        from edocr2.keras_ocr.detection import Detector
        import tensorflow as tf
        
        import edocr2
        import edocr2.keras_ocr.recognition as rec_module
        print(f"DEBUG: edocr2 from = {getattr(edocr2, '__file__', 'N/A')}", file=sys.stderr)
        print(f"DEBUG: recognition from = {rec_module.__file__}", file=sys.stderr)
        
        # Suppress additional TF logging
        tf.get_logger().setLevel('ERROR')
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load image - support both PDF and image files
        if image_path.lower().endswith('.pdf'):
            # PDF: Use pdf2image (Poppler) for rendering - same as test_drawing.py
            from pdf2image import convert_from_path
            print(f"Loading PDF with pdf2image (Poppler)...", file=sys.stderr)
            
            # Get page index from args if provided (default to first page)
            page_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            
            # Use 200 DPI (pdf2image default) to match test_drawing.py behavior
            images = convert_from_path(image_path, first_page=page_index+1, last_page=page_index+1, dpi=200)
            img = np.array(images[0])
            # pdf2image returns RGB, cv2 expects BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f"PDF rendered at 200 DPI: {img.shape[1]}x{img.shape[0]} pixels", file=sys.stderr)
        else:
            # Image file: Load directly
            img = cv2.imread(image_path)
            if img is None:
                print(json.dumps({"error": f"Failed to load image: {image_path}"}))
                sys.exit(1)
            print(f"Loaded image: {img.shape[1]}x{img.shape[0]} pixels", file=sys.stderr)
        
        # Apply same preprocessing as test_drawing.py for better OCR results
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        img = cv2.merge([img, img, img])
        print(f"Applied binary thresholding (200) for cleaner OCR", file=sys.stderr)
        
        height, width = img.shape[:2]
        
        # Step 1: Segmentation
        print("Running segmentation...", file=sys.stderr)
        img_boxes, frame, gdt_boxes, tables, dim_boxes = tools.layer_segm.segment_img(
            img, autoframe=True, frame_thres=0.7, GDT_thres=0.02, binary_thres=127
        )
        print(f"Segmentation results: frame={frame is not None}, gdt_boxes={len(gdt_boxes) if gdt_boxes else 0}, tables={len(tables) if tables else 0}, dim_boxes={len(dim_boxes) if dim_boxes else 0}", file=sys.stderr)
        
        # Step 2: Load models
        overall_start = time.time()
        print("Loading models...", file=sys.stderr)
        
        gdt_model = 'edocr2/models/recognizer_gdts.keras'
        dim_model = 'edocr2/models/recognizer_dimensions_2.keras'
        
        # Verify model files exist
        print(f"GDT model exists: {os.path.exists(gdt_model)}, size: {os.path.getsize(gdt_model) if os.path.exists(gdt_model) else 0}", file=sys.stderr)
        print(f"DIM model exists: {os.path.exists(dim_model)}, size: {os.path.getsize(dim_model) if os.path.exists(dim_model) else 0}", file=sys.stderr)
        
        # Suppress stdout during all keras-ocr operations (it prints to stdout, polluting JSON)
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Initialize variables for error handling
        dimensions = []
        other_info = []
        table_results = []
        gdt_results = []
        end_time = overall_start
        
        try:
            recognizer_gdt = None
            if gdt_boxes:
                print(f"Loading GDT model for {len(gdt_boxes)} detected GD&T regions...", file=sys.stderr)
                t1 = time.time()
                recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model))
                print(f"  GDT Recognizer init: {time.time()-t1:.3f}s", file=sys.stderr)
                t1 = time.time()
                recognizer_gdt.model.load_weights(gdt_model)
                print(f"  GDT load_weights: {time.time()-t1:.3f}s", file=sys.stderr)
            else:
                print("No GD&T boxes detected during segmentation, skipping GDT model", file=sys.stderr)
            
            t1 = time.time()
            alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_model)
            recognizer_dim = Recognizer(alphabet=alphabet_dim)
            print(f"  DIM Recognizer init: {time.time()-t1:.3f}s", file=sys.stderr)
            t1 = time.time()
            recognizer_dim.model.load_weights(dim_model)
            print(f"  DIM load_weights: {time.time()-t1:.3f}s", file=sys.stderr)
            t1 = time.time()
            detector = Detector()
            print(f"  Detector init: {time.time()-t1:.3f}s", file=sys.stderr)
            
            end_time = time.time()
            
            # Step 3: OCR Tables
            process_img = img.copy()
            table_results, updated_tables, process_img = tools.ocr_pipelines.ocr_tables(tables, process_img, 'eng')
            
            # Step 4: OCR GD&T
            gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(
                process_img, gdt_boxes, recognizer_gdt
            )
            
            # Step 5: OCR Dimensions
            if frame:
                process_img_cropped = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
            else:
                process_img_cropped = process_img
            
            dimensions, other_info, process_img_out, dim_tess = tools.ocr_pipelines.ocr_dimensions(
                process_img_cropped, detector, recognizer_dim, alphabet_dim, frame, dim_boxes,
                cluster_thres=20, max_img_size=1048, language='eng', backg_save=False
            )
        except Exception as inner_e:
            # Restore stdout first
            captured = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if captured.strip():
                for line in captured.strip().split('\n')[:5]:
                    print(f"[keras] {line}", file=sys.stderr)
            # Re-raise with context
            import traceback
            print(f"ERROR in OCR: {inner_e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            raise
        else:
            # Success - restore stdout
            captured = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if captured.strip():
                for line in captured.strip().split('\n')[:5]:
                    print(f"[keras] {line}", file=sys.stderr)
        
        print(f"Loading session took {end_time - overall_start:.6f} seconds", file=sys.stderr)
        print(f"Extraction complete: {len(dimensions)} dimensions, {len(other_info)} other_info, {len(gdt_results)} GD&T, {len(table_results)} tables", file=sys.stderr)
        
        # Convert results to JSON-serializable format
        detections = []
        
        # Convert dimensions
        for i, dim in enumerate(dimensions):
            try:
                # Handle both tuple and list
                if isinstance(dim, (tuple, list)) and len(dim) >= 2:
                    text, bbox_corners = dim[0], dim[1]
                    # Handle numpy array
                    if hasattr(bbox_corners, 'shape'):
                        x_coords = bbox_corners[:, 0]
                        y_coords = bbox_corners[:, 1]
                    else:
                        # Handle list of [x, y] pairs
                        x_coords = [p[0] for p in bbox_corners]
                        y_coords = [p[1] for p in bbox_corners]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    center_x = x_min + bbox_width / 2
                    center_y = y_min + bbox_height / 2
                    
                    # Adjust for frame offset
                    if frame:
                        center_x += frame.x
                        center_y += frame.y
                        x_min += frame.x
                        y_min += frame.y
                    
                    # Parse dimension designator (symbol like R, ⌀, M, etc.)
                    dim_text = str(text) if text else ""
                    designator = ""
                    if dim_text:
                        # Check for common dimension symbols at the start
                        # ⌀ = diameter, R = radius, M = metric thread, ∅ = diameter
                        first_char = dim_text[0]
                        if first_char in ['R', 'M', '⌀', '∅', 'Ø']:
                            designator = first_char
                        elif ' ' in dim_text:
                            # Fall back to first word if there's a space
                            designator = dim_text.split()[0]
                        else:
                            # No clear designator, use full text
                            designator = dim_text
                    
                    detections.append({
                        "type": "dimension",
                        "x": float(center_x),
                        "y": float(center_y),
                        "bbox": {"x": float(x_min), "y": float(y_min), "width": float(bbox_width), "height": float(bbox_height)},
                        "requirement": dim_text,
                        "designator": designator,
                        "ref_location": "",
                        "tolerance": "",
                        "confidence": 0.9
                    })
            except Exception as conv_e:
                print(f"Warning: Failed to convert dimension {i}: {conv_e}, dim={type(dim)}", file=sys.stderr)
        
        # Convert other_info
        for i, info in enumerate(other_info):
            try:
                # Handle both tuple and list
                if isinstance(info, (tuple, list)) and len(info) >= 2:
                    text, bbox_corners = info[0], info[1]
                    # Handle numpy array
                    if hasattr(bbox_corners, 'shape'):
                        x_coords = bbox_corners[:, 0]
                        y_coords = bbox_corners[:, 1]
                    else:
                        # Handle list of [x, y] pairs
                        x_coords = [p[0] for p in bbox_corners]
                        y_coords = [p[1] for p in bbox_corners]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    center_x = x_min + bbox_width / 2
                    center_y = y_min + bbox_height / 2
                    
                    if frame:
                        center_x += frame.x
                        center_y += frame.y
                        x_min += frame.x
                        y_min += frame.y
                    
                    detections.append({
                        "type": "info",
                        "x": float(center_x),
                        "y": float(center_y),
                        "bbox": {"x": float(x_min), "y": float(y_min), "width": float(bbox_width), "height": float(bbox_height)},
                        "requirement": str(text)[:200] if text else "",
                        "designator": "Info",
                        "ref_location": "",
                        "tolerance": "",
                        "confidence": 0.9
                    })
            except Exception as conv_e:
                print(f"Warning: Failed to convert info {i}: {conv_e}, info={type(info)}", file=sys.stderr)
        
        # Convert tables
        print(f"Converting {len(table_results)} table results...", file=sys.stderr)
        for i, table in enumerate(table_results):
            try:
                print(f"  Table {i}: type={type(table)}, len={len(table) if hasattr(table, '__len__') else 'N/A'}", file=sys.stderr)
                if isinstance(table, list) and len(table) > 0:
                    lefts = [word['left'] for word in table if 'left' in word]
                    tops = [word['top'] for word in table if 'top' in word]
                    rights = [word['left'] + word['width'] for word in table if 'left' in word and 'width' in word]
                    bottoms = [word['top'] + word['height'] for word in table if 'top' in word and 'height' in word]
                    
                    print(f"  Table {i}: lefts={len(lefts)}, tops={len(tops)}, rights={len(rights)}, bottoms={len(bottoms)}", file=sys.stderr)
                    
                    if lefts and tops and rights and bottoms:
                        x_min = min(lefts)
                        y_min = min(tops)
                        x_max = max(rights)
                        y_max = max(bottoms)
                        
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        center_x = x_min + bbox_width / 2
                        center_y = y_min + bbox_height / 2
                        
                        content = ' '.join(str(word.get('text', '')) for word in table)
                        
                        detections.append({
                            "type": "table",
                            "x": float(center_x),
                            "y": float(center_y),
                            "bbox": {"x": float(x_min), "y": float(y_min), "width": float(bbox_width), "height": float(bbox_height)},
                            "requirement": content[:100],
                            "designator": "Table",
                            "ref_location": "",
                            "tolerance": "",
                            "confidence": 0.9
                        })
                        print(f"  Table {i}: ADDED to detections", file=sys.stderr)
                    else:
                        print(f"  Table {i}: SKIPPED - missing coordinate data", file=sys.stderr)
                else:
                    print(f"  Table {i}: SKIPPED - not a non-empty list", file=sys.stderr)
            except Exception as conv_e:
                print(f"Warning: Failed to convert table {i}: {conv_e}", file=sys.stderr)
        
        # Convert GD&T - use updated_gdt_boxes for actual bounding boxes
        print(f"Converting {len(gdt_results)} GD&T results (with {len(updated_gdt_boxes)} boxes)...", file=sys.stderr)
        for i, gdt in enumerate(gdt_results):
            try:
                print(f"  GD&T {i}: type={type(gdt)}, len={len(gdt) if hasattr(gdt, '__len__') else 'N/A'}", file=sys.stderr)
                # Handle both tuple and list (edocr2 returns list)
                if (isinstance(gdt, (tuple, list))) and len(gdt) >= 2:
                    text, point_coords = gdt[0], gdt[1]
                    print(f"  GD&T {i}: text='{text}', point_coords={point_coords}", file=sys.stderr)
                    
                    # Try to get bounding box from updated_gdt_boxes (contains actual dimensions)
                    x_min, y_min, x_max, y_max = None, None, None, None
                    
                    if i < len(updated_gdt_boxes):
                        # updated_gdt_boxes[i] is a dict where values are lists of box objects with .x, .y, .w, .h
                        gdt_block = updated_gdt_boxes[i]
                        all_boxes = []
                        for key, box_list in gdt_block.items():
                            all_boxes.extend(box_list)
                        
                        if all_boxes:
                            # Compute bounding box that encloses all sub-boxes
                            x_min = min(b.x for b in all_boxes)
                            y_min = min(b.y for b in all_boxes)
                            x_max = max(b.x + b.w for b in all_boxes)
                            y_max = max(b.y + b.h for b in all_boxes)
                            print(f"  GD&T {i}: bbox from updated_gdt_boxes: ({x_min}, {y_min}) to ({x_max}, {y_max})", file=sys.stderr)
                    
                    # Fallback: create box from point if updated_gdt_boxes didn't work
                    if x_min is None:
                        if isinstance(point_coords, (tuple, list)) and len(point_coords) == 2:
                            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in point_coords):
                                # (x, y) is top-left of first sub-box, not center
                                px, py = int(point_coords[0]), int(point_coords[1])
                                # Create a box starting at this point (typical GD&T symbol size)
                                box_w, box_h = 150, 40
                                x_min, y_min = px, py
                                x_max, y_max = px + box_w, py + box_h
                                print(f"  GD&T {i}: Created fallback bbox from point ({px}, {py})", file=sys.stderr)
                    
                    if x_min is None:
                        print(f"  GD&T {i}: SKIPPED - couldn't determine bbox", file=sys.stderr)
                        continue
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    center_x = x_min + bbox_width / 2
                    center_y = y_min + bbox_height / 2
                    
                    print(f"  GD&T {i}: final bbox=({x_min}, {y_min}, {x_max}, {y_max})", file=sys.stderr)
                    
                    # Parse GD&T: format is typically "⌰|06|A" where first char is the symbol
                    gdt_text = str(text) if text else ""
                    # Extract the GD&T symbol (first character before |)
                    gdt_parts = gdt_text.split("|")
                    gdt_symbol = gdt_parts[0] if gdt_parts else gdt_text
                    
                    detections.append({
                        "type": "gdt",
                        "x": float(center_x),
                        "y": float(center_y),
                        "bbox": {"x": float(x_min), "y": float(y_min), "width": float(bbox_width), "height": float(bbox_height)},
                        "requirement": gdt_text,
                        "designator": gdt_symbol,  # Just the GD&T symbol (e.g., "⌰" for position)
                        "ref_location": "",
                        "tolerance": "",
                        "confidence": 0.9
                    })
                    print(f"  GD&T {i}: ADDED to detections", file=sys.stderr)
                else:
                    print(f"  GD&T {i}: SKIPPED - not tuple/list or len < 2", file=sys.stderr)
            except Exception as conv_e:
                import traceback
                print(f"Warning: Failed to convert gdt {i}: {conv_e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        
        # Output result as JSON (stdout)
        result = {
            "success": True,
            "detections": detections,
            "image_width": int(width),
            "image_height": int(height),
            "stats": {
                "dimensions": len(dimensions),
                "other_info": len(other_info),
                "tables": len(table_results),
                "gdt": len(gdt_results),
                "total_detections": len(detections)
            }
        }
        print(f"Outputting JSON with {len(detections)} detections...", file=sys.stderr)
        # Use flush=True to ensure output is captured by subprocess
        print(json.dumps(result), flush=True)
        print("JSON output complete", file=sys.stderr)
        
    except Exception as e:
        import traceback
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        # Catch-all to ensure we always output JSON
        error_result = {
            "success": False,
            "error": f"Unhandled exception: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)
