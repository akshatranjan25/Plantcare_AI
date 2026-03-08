import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, url_for
from flask import send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

try:
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
	preprocess_input = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "plantcare_model.pkl"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.txt"

# Standard PlantVillage 38-class order used by many transfer-learning projects.
DEFAULT_PLANTVILLAGE_38_CLASS_NAMES = [
	"Apple___Apple_scab",
	"Apple___Black_rot",
	"Apple___Cedar_apple_rust",
	"Apple___healthy",
	"Blueberry___healthy",
	"Cherry_(including_sour)___Powdery_mildew",
	"Cherry_(including_sour)___healthy",
	"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
	"Corn_(maize)___Common_rust_",
	"Corn_(maize)___Northern_Leaf_Blight",
	"Corn_(maize)___healthy",
	"Grape___Black_rot",
	"Grape___Esca_(Black_Measles)",
	"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
	"Grape___healthy",
	"Orange___Haunglongbing_(Citrus_greening)",
	"Peach___Bacterial_spot",
	"Peach___healthy",
	"Pepper,_bell___Bacterial_spot",
	"Pepper,_bell___healthy",
	"Potato___Early_blight",
	"Potato___Late_blight",
	"Potato___healthy",
	"Raspberry___healthy",
	"Soybean___healthy",
	"Squash___Powdery_mildew",
	"Strawberry___Leaf_scorch",
	"Strawberry___healthy",
	"Tomato___Bacterial_spot",
	"Tomato___Early_blight",
	"Tomato___Late_blight",
	"Tomato___Leaf_Mold",
	"Tomato___Septoria_leaf_spot",
	"Tomato___Spider_mites Two-spotted_spider_mite",
	"Tomato___Target_Spot",
	"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
	"Tomato___Tomato_mosaic_virus",
	"Tomato___healthy"
]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["SECRET_KEY"] = "plantcare-ai-secret"


MODEL = None
CLASS_NAMES = None
MODEL_ERROR = None


def allowed_file(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_bundle(path: Path):
	with open(path, "rb") as file:
		bundle = pickle.load(file)

	if isinstance(bundle, dict):
		model = bundle.get("model") or bundle.get("classifier") or bundle.get("estimator")
		class_names = bundle.get("class_names") or bundle.get("labels") or bundle.get("classes")
		if model is None:
			raise ValueError("Pickle dictionary must include a model object under 'model'.")
		return model, class_names

	return bundle, None


def load_class_names_sidecar(path: Path):
	if not path.exists():
		return None

	class_names = []
	with open(path, "r", encoding="utf-8") as file:
		for line in file:
			name = line.strip()
			if name:
				class_names.append(name)

	return class_names or None


def get_output_class_count(model) -> int | None:
	try:
		shape = model.output_shape
		if isinstance(shape, list):
			shape = shape[0]
		if isinstance(shape, tuple) and shape and shape[-1] is not None:
			return int(shape[-1])
	except Exception:
		return None

	return None


def prettify_class_name(raw_name: str) -> str:
	name = raw_name.replace("___", " - ").replace("_", " ")
	name = name.replace(",", ", ")
	name = " ".join(name.split())
	return name


def init_model():
	global MODEL, CLASS_NAMES, MODEL_ERROR

	UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

	if not MODEL_PATH.exists():
		MODEL_ERROR = f"Model file not found at: {MODEL_PATH}"
		return

	try:
		MODEL, CLASS_NAMES = load_model_bundle(MODEL_PATH)

		if not CLASS_NAMES:
			CLASS_NAMES = load_class_names_sidecar(CLASS_NAMES_PATH)

		if not CLASS_NAMES:
			class_count = get_output_class_count(MODEL)
			if class_count == len(DEFAULT_PLANTVILLAGE_38_CLASS_NAMES):
				CLASS_NAMES = DEFAULT_PLANTVILLAGE_38_CLASS_NAMES
	except Exception as exc:
		MODEL_ERROR = f"Failed to load model: {exc}"


def preprocess_image(image_path: Path) -> np.ndarray:
	image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
	image_array = np.array(image, dtype=np.float32)

	if preprocess_input is not None:
		image_array = preprocess_input(image_array)
	else:
		image_array = image_array / 127.5 - 1.0

	return np.expand_dims(image_array, axis=0)


def summarize_prediction(class_name: str) -> str:
	class_lower = class_name.lower()
	if "healthy" in class_lower:
		return "Healthy"
	return "Diseased"


def predict_image(image_path: Path):
	if MODEL is None:
		raise RuntimeError(MODEL_ERROR or "Model is not loaded.")

	input_tensor = preprocess_image(image_path)
	probabilities = MODEL.predict(input_tensor)

	if isinstance(probabilities, list):
		probabilities = np.array(probabilities)

	probabilities = np.array(probabilities)
	if probabilities.ndim == 1:
		probabilities = np.expand_dims(probabilities, axis=0)

	row = probabilities[0]

	if row.shape[0] == 1:
		score = float(row[0])
		predicted_index = 1 if score >= 0.5 else 0
		confidence = score if score >= 0.5 else 1.0 - score
	else:
		predicted_index = int(np.argmax(row))
		confidence = float(row[predicted_index])

	if CLASS_NAMES is not None and len(CLASS_NAMES) > predicted_index:
		predicted_class = prettify_class_name(str(CLASS_NAMES[predicted_index]))
	else:
		predicted_class = f"Class {predicted_index}"

	health_status = summarize_prediction(predicted_class)
	return predicted_class, health_status, confidence


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
	error_message = MODEL_ERROR
	prediction = None
	health_status = None
	confidence = None
	uploaded_image = None

	if request.method == "GET":
		return render_template("upload.html", error=error_message)

	if request.method == "POST":
		if MODEL is None:
			error_message = MODEL_ERROR or "Model is not loaded."
			return render_template("upload.html", error=error_message)

		if "image" not in request.files:
			error_message = "Please select an image file."
			return render_template("upload.html", error=error_message)

		file = request.files["image"]
		if file.filename == "":
			error_message = "No file selected."
			return render_template("upload.html", error=error_message)

		if not allowed_file(file.filename):
			error_message = "Unsupported file format. Use PNG, JPG, JPEG, or WEBP."
			return render_template("upload.html", error=error_message)

		filename = secure_filename(file.filename)
		save_path = UPLOAD_FOLDER / filename
		file.save(save_path)

		uploaded_image = url_for("uploaded_file", filename=filename)

		try:
			prediction, health_status, confidence = predict_image(save_path)
		except Exception as exc:
			error_message = f"Prediction failed: {exc}"
			return render_template("upload.html", error=error_message)

	return render_template(
		"result.html",
		prediction=prediction,
		health_status=health_status,
		confidence=confidence,
		uploaded_image=uploaded_image,
	)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
	return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


init_model()


if __name__ == "__main__":
	app.run(debug=True)
