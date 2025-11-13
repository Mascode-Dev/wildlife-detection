import json
import os

# --- 🎯 CONFIGURATION ---
# Mettez à jour ce chemin vers le répertoire 'WILDLIFE-DETECTION'.
# Si vide, le script utilisera le répertoire courant.
DATA_ROOT = os.getcwd() 

# Dossiers d'entrée/sortie basés sur la structure du projet
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, 'annotations')
OUTPUT_LABELS_DIR = os.path.join(DATA_ROOT, 'labels')
# --- FIN CONFIGURATION ---

# Dictionnaires des fichiers à traiter
JSON_FILES = {
    'train': 'wildlife_instance_train2017.json',
    'val': 'wildlife_instance_val2017.json',
    'test': 'wildlife_instance_test2017.json',
}

# --- INITIALISATION GLOBALE (Classes et Mappage) ---
# On détermine les classes une seule fois à partir du TRAIN set pour garantir la cohérence des IDs.
def setup_classes():
    """Crée le mappage des classes et génère le fichier data.yaml."""
    train_json_path = os.path.join(ANNOTATIONS_DIR, JSON_FILES['train'])
    
    # Assurez-vous que le fichier d'entraînement existe
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Fichier d'entraînement non trouvé : {train_json_path}. Vérifiez DATA_ROOT.")

    with open(train_json_path, 'r', encoding='latin-1') as f:
        train_data = json.load(f)

    # Création du mappage des classes (ID COCO -> ID YOLO)
    yolo_classes = []
    coco_id_to_yolo_id = {}
    
    for i, cat in enumerate(train_data['categories']):
        coco_id_to_yolo_id[cat['id']] = i
        yolo_classes.append(cat['name'])

    print(f"✅ Classes trouvées : {len(yolo_classes)} : {yolo_classes}")
    
    # Création du fichier de configuration YOLOv8 (data.yaml)
    yaml_content = f"names: {yolo_classes}\n"
    yaml_content += f"nc: {len(yolo_classes)}\n"
    # Les chemins sont relatifs au DATA_ROOT, correspondant à la structure de dossiers
    yaml_content += "train: images/train\n" 
    yaml_content += "val: images/val\n"
    yaml_content += "test: images/test\n"

    with open(os.path.join(DATA_ROOT, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("✅ Fichier data.yaml créé.")
    
    return coco_id_to_yolo_id, yolo_classes


# --- FONCTION DE CONVERSION POUR UN SEUL SPLIT ---
def process_single_split(split_name, json_filename, coco_id_to_yolo_id):
    """Convertit un seul fichier JSON (train/val/test) au format YOLO."""
    json_path = os.path.join(ANNOTATIONS_DIR, json_filename)
    output_sub_dir = os.path.join(OUTPUT_LABELS_DIR, split_name)
    os.makedirs(output_sub_dir, exist_ok=True)

    print(f"\n--- Traitement du split : {split_name} ({json_filename}) ---")

    with open(json_path, 'r', encoding='latin-1') as f:
        data = json.load(f)

    # Créer le mappage des images (ID -> Dimensions et Nom de Fichier)
    image_info = {
        img['id']: {
            'width': img['width'], 
            'height': img['height'], 
            'file_name': img['file_name']
        } for img in data['images']
    }

    # Grouper les annotations par ID d'image
    annotations_per_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []
        annotations_per_image[img_id].append(ann)

    print(f"Conversion de {len(annotations_per_image)} images pour le split {split_name}...")

    # Boucle de conversion et écriture des fichiers .txt
    for img_id, anns in annotations_per_image.items():
        info = image_info[img_id]
        W_img, H_img = info['width'], info['height']
        file_name = info['file_name']
        
        yolo_lines = []
        
        for ann in anns:
            coco_bbox = ann['bbox'] # [x, y, w, h] en pixels
            coco_cat_id = ann['category_id']
            
            yolo_class_id = coco_id_to_yolo_id.get(coco_cat_id)
            if yolo_class_id is None:
                # Cela gère les cas où un ID de catégorie existe dans val/test mais pas dans train
                # Bien que rare, c'est une bonne pratique de l'ignorer.
                continue 
            
            # Calcul de la normalisation (centre et dimensions)
            x_min, y_min, w, h = coco_bbox
            x_center = (x_min + w / 2) / W_img
            y_center = (y_min + h / 2) / H_img
            w_norm = w / W_img
            h_norm = h / H_img
            
            # Formater la ligne YOLO
            yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)
            
        # Écriture du fichier .txt
        if yolo_lines:
            label_filename = os.path.splitext(file_name)[0] + '.txt' 
            label_path = os.path.join(output_sub_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

    print(f"Fichiers labels créés dans : {output_sub_dir}")


# --- EXÉCUTION DU PROGRAMME PRINCIPAL ---
if __name__ == '__main__':
    try:
        # 1. Configuration des classes et création du data.yaml
        coco_map, _ = setup_classes()
        
        # 2. Traitement des trois splits
        for split, filename in JSON_FILES.items():
            process_single_split(split, filename, coco_map)
            
        print("\n--- CONVERSION GLOBALE COMPLÈTE ! ---")
        print("Votre structure est prête pour l'entraînement YOLOv8.")
        
    except FileNotFoundError as e:
        print(f"\nFATALE ERREUR : {e}")
        print("Veuillez vérifier que le chemin DATA_ROOT et la structure des dossiers sont corrects.")