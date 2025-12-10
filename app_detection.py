import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import cv2
import numpy as np
from PIL import Image

# Import de la librairie YOLOv8
from ultralytics import YOLO

# --- 🎯 MODEL AND PATH CONFIG ---
MODEL_PATH = 'runs/detect/wildlife_detection_v1/weights/best.pt' 

# Load the YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLOv8 model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    # If the model fails to load, the app will not start correctly
    model = None 
# --- END CONFIGURATION ---


# --- DASH APPLICATION INITIALIZATION ---
app = dash.Dash(__name__)

# --- DASH APPLICATION LAYOUT ---
app.layout = html.Div(
    style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px', 'fontFamily': 'Arial'},
    children=[
        html.H1("🐾 Wildlife Detector (LoTE-Animal)"),
        html.P("Upload a camera trap image to detect and identify endangered species."),
        
        # Upload Component
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and drop or ',
                html.A('Select a File')
            ]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=False
        ),

        html.Div(id='output-image-upload', style={'margin-top': '20px'}),
        
        html.H3("Detection Results:", style={'margin-top': '30px'}),
        html.Div(id='output-resultats')
    ]
)

# --- PROCESSING FUNCTIONS ---

def parse_contents(contents, filename, date):
    """DDecode the image and prepare it for inference."""
    if contents is None:
        return None, None

    # DDecode the base64 string
    content_type, content_string = contents.split(',')
    decoded_bytes = base64.b64decode(content_string)
    
    try:
        # Open the image with PIL (Pillow)
        image = Image.open(io.BytesIO(decoded_bytes)).convert("RGB")
        
        # Convert PIL -> OpenCV (BGR/NumPy format) for plotting
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return html.Div(['An error occurred while processing the file.']), None

    return image, image_cv, decoded_bytes # Return the PIL image for YOLO and the CV image for plotting

@app.callback(
    [Output('output-image-upload', 'children'),
        Output('output-resultats', 'children')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename'),
        State('upload-image', 'last_modified')]
)
def update_output(contents, filename, date):
    """Main function for inference and display."""
    if contents is None or model is None:
        return html.Div(), html.Div() # Nothing to do if no content or model not loaded

    # 1. DDecode the image
    image_pil, image_cv, decoded_bytes = parse_contents(contents, filename, date)
    if image_pil is None:
        return html.Div(['Error loading image.']), html.Div()
    
    # 2. Run YOLO Inference
    try:
        # The model takes a PIL or NumPy image as input
        results = model(image_pil, imgsz=640, conf=0.25, verbose=False)
    except Exception as e:
        return html.Div([f"YOLO inference error: {e}"]), html.Div()


    # 3. Processing Results and Plotting
    detected_animals = []
    
    # Iterate over results (single result for one image)
    if results and results[0].boxes:
        
        im_plotted = results[0].plot() # Returns a NumPy/CV2 array with the boxes drawn
        
        # Convert the plotted image (CV2 BGR) to base64 format for Dash display
        im_plotted_rgb = cv2.cvtColor(im_plotted, cv2.COLOR_BGR2RGB)
        im_plotted_pil = Image.fromarray(im_plotted_rgb)
        
        # Temporary save in memory
        buff = io.BytesIO()
        im_plotted_pil.save(buff, format="PNG")
        encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")

        # Extract metrics for the list
        for box in results[0].boxes:
            class_id = int(box.cls)
            conf_score = float(box.conf)
            class_name = model.names.get(class_id, f"Unknown (ID {class_id})")
            
            detected_animals.append({
                'name': class_name,
                'conf': conf_score
            })
    
    else:
        # No detection
        encoded_image = content_string # Display original image
        detected_animals.append({'name': "Aucune espèce animale détectée.", 'conf': 0})
        
    
    # --- 4. Formatting for Display ---
    
    # Display the image (with or without detections)
    image_display = html.Div(
        [
            html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%', 'max-height': '600px', 'object-fit': 'contain'}),
            html.Hr(),
            html.P(f"Filename: {filename}")
        ]
    )
    
    # Display results
    results_list = [
        html.Li(f"**{item['name']}** - Confidence: {item['conf']:.2f}", 
                style={'color': 'green' if item['conf'] > 0.5 else 'orange', 'fontWeight': 'bold'})
        for item in detected_animals if item['conf'] > 0
    ]
    
    results_display = html.Ul(results_list)
    
    if not results_list and detected_animals[0]['name'] == "No animal species detected.":
        results_display = html.P("No animal species detected with sufficient confidence (conf > 25%).", style={'color': 'red'})


    return image_display, results_display


# --- SERVER LAUNCH ---
if __name__ == '__main__':
    # Set debug=True for development. Set to False for production.
    app.run(debug=True)