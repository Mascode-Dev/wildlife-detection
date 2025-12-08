from ultralytics import YOLO

# 1. Charger le modèle pré-entraîné (Transfer Learning)
# C'est la base qui a déjà appris les formes générales.
model = YOLO('yolov8s.pt') 


if __name__ == "__main__":
    # 2. Lancer l'entraînement
    results = model.train(
        data='data.yaml',      # Fichier de configuration des données
        epochs=100,            # Nombre d'époques (ajustez si besoin)
        imgsz=640,             # Taille des images d'entrée (640x640)
        batch=-1,              # Batch size automatique
        name='wildlife_detection_v1', # Nom de la session
        patience=50            # Arrêter si pas d'amélioration sur 50 époques (évite l'overfitting)
    )

    print("Entraînement terminé. Résultats sauvegardés dans runs/detect/wildlife_detection_v1")