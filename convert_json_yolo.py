import json
import os

DATA_ROOT = os.getcwd() 

# Input/output directories based on project structure
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, 'annotations')
OUTPUT_LABELS_DIR = os.path.join(DATA_ROOT, 'labels')

# Dictionaries of files to process
JSON_FILES = {
    'train': 'wildlife_instance_train2017.json',
    'val': 'wildlife_instance_val2017.json',
    'test': 'wildlife_instance_test2017.json',
}

# --- GLOBAL INITIALIZATION (Classes and Mapping) ---
# We determine the classes only once from the TRAIN set to ensure ID consistency.
def setup_classes():
    """Create the class mapping and generate the data.yaml file."""
    train_json_path = os.path.join(ANNOTATIONS_DIR, JSON_FILES['train'])
    
    # Ensure the training file exists
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training file not found: {train_json_path}. Check DATA_ROOT.")

    with open(train_json_path, 'r', encoding='latin-1') as f:
        train_data = json.load(f)

    # Creation of class mapping (COCO ID -> YOLO ID)
    yolo_classes = []
    coco_id_to_yolo_id = {}
    
    for i, cat in enumerate(train_data['categories']):
        coco_id_to_yolo_id[cat['id']] = i
        yolo_classes.append(cat['name'])

    print(f"✅ Classes found: {len(yolo_classes)} : {yolo_classes}")
    
    # Creation of YOLOv8 configuration file (data.yaml)
    yaml_content = f"names: {yolo_classes}\n"
    yaml_content += f"nc: {len(yolo_classes)}\n"
    # Paths are relative to DATA_ROOT, matching the folder structure
    yaml_content += "train: images/train\n" 
    yaml_content += "val: images/val\n"
    yaml_content += "test: images/test\n"

    with open(os.path.join(DATA_ROOT, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("✅ data.yaml file created.")
    
    return coco_id_to_yolo_id, yolo_classes


# --- FUNCTION TO CONVERT A SINGLE SPLIT ---
def process_single_split(split_name, json_filename, coco_id_to_yolo_id):
    """Convert a single JSON file (train/val/test) to YOLO format."""
    json_path = os.path.join(ANNOTATIONS_DIR, json_filename)
    output_sub_dir = os.path.join(OUTPUT_LABELS_DIR, split_name)
    os.makedirs(output_sub_dir, exist_ok=True)

    print(f"\n--- Processing split: {split_name} ({json_filename}) ---")

    with open(json_path, 'r', encoding='latin-1') as f:
        data = json.load(f)

    # Create the mapping of images (ID -> Dimensions and Filename)
    image_info = {
        img['id']: {
            'width': img['width'], 
            'height': img['height'], 
            'file_name': img['file_name']
        } for img in data['images']
    }

    # Group annotations by image ID
    annotations_per_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []
        annotations_per_image[img_id].append(ann)

    print(f"Converting {len(annotations_per_image)} images for split {split_name}...")

    # Conversion loop and writing .txt files
    for img_id, anns in annotations_per_image.items():
        info = image_info[img_id]
        W_img, H_img = info['width'], info['height']
        file_name = info['file_name']
        
        yolo_lines = []
        
        for ann in anns:
            coco_bbox = ann['bbox'] # [x, y, w, h] in pixels
            coco_cat_id = ann['category_id']
            
            yolo_class_id = coco_id_to_yolo_id.get(coco_cat_id)
            if yolo_class_id is None:
                # This handles cases where a category ID exists in val/test but not in train
                continue 
            
            # Calculation of normalization (center and dimensions)
            x_min, y_min, w, h = coco_bbox
            x_center = (x_min + w / 2) / W_img
            y_center = (y_min + h / 2) / H_img
            w_norm = w / W_img
            h_norm = h / H_img
            
            # Format the YOLO line
            yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)
            
        # Writing the .txt file
        if yolo_lines:
            label_filename = os.path.splitext(file_name)[0] + '.txt' 
            label_path = os.path.join(output_sub_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

    print(f"Fichiers labels créés dans : {output_sub_dir}")

if __name__ == '__main__':
    try:
        # 1. Config of classes and mapping
        coco_map, _ = setup_classes()
        
        # 2. Processing the three splits
        for split, filename in JSON_FILES.items():
            process_single_split(split, filename, coco_map)
            
        print("\n--- GLOBAL CONVERSION COMPLETE! ---")
        print("Your structure is ready for YOLOv8 training.")
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please check that the DATA_ROOT path and folder structure are correct.")