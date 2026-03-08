# PlantCare AI (Flask + Transfer Learning Inference)

PlantCare AI is a web application for plant leaf disease classification using a fine-tuned transfer learning model (MobileNetV2-based) saved as a pickle file.

## Features
- Upload a leaf image (`.png`, `.jpg`, `.jpeg`, `.webp`)
- Run classification from a pickled fine-tuned model
- Display predicted class, confidence, and status (`Healthy` or `Diseased`)

## Project Structure
- `app.py` - Flask backend and inference logic
- `templates/index.html` - Upload UI
- `static/style.css` - Styling
- `model/plantcare_model.pkl` - Place your trained pickle model here
- `uploads/` - Temporary uploaded images (auto-created)

## Pickle Model Expectations
The app supports either:
1. A direct model object with a `.predict()` method, or
2. A dictionary bundle with:
	- `model`: trained model object
	- `class_names`: list of class labels (recommended)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your model file at:

```text
model/plantcare_model.pkl
```

4. Run the app:

```bash
python app.py
```

5. Open in browser:

```text
http://127.0.0.1:5000
```

## Notes
- If the model file is missing, the app shows an error on the page.
- `Healthy`/`Diseased` status is inferred from whether predicted class label contains the word `healthy`.
