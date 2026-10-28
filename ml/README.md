# CivicLens image-classification module

This is a reproducible **transfer-learning training pipeline**, not a claim that model weights have already been trained. It trains two separate binary classifiers using MobileNetV3-Small:

- **Pothole vs. non-pothole**
- **Garbage dump vs. non-garbage**

Streetlights and open drains are not covered by these classifiers; those categories remain manual choices. The two independent binary tasks are intentional because one photo may contain more than one kind of civic issue.

## Dataset

A relevant public dataset is [QR4Change: A Smart QR-Based Civic Grievance Reporting System](https://data.mendeley.com/datasets/zndzygc3p3/1). Its description reports 2,966 pothole/non-pothole images and 1,971 garbage/non-garbage images and lists a **CC BY 4.0** licence. Read the dataset page and its attribution terms before redistributing data. Do not commit the image dataset to this GitHub repository.

Arrange extracted images as:

```text
data/raw/
├── pothole/       # positive pothole images
├── not_pothole/   # plain/non-pothole images
├── garbage/       # positive garbage-dump images
└── not_garbage/   # clean/non-garbage images
```

If the dataset has nested folders, put images into the matching four class folders. Avoid putting duplicate or near-duplicate photos in both training and validation sets.

## Train in Google Colab (recommended)

1. Open the notebook [`CivicLens_Model_Training.ipynb`](../CivicLens_Model_Training.ipynb) in Google Colab.
2. Download the dataset from Mendeley and upload the extracted class folders to the Colab session, following the layout above.
3. Run the notebook cells. It prepares an 80/20 train/validation split, downloads ImageNet-pretrained MobileNetV3 weights, fine-tunes the classifier heads, and saves validation metrics.
4. Download the generated `civiclens_models.zip`, then extract its two `.pth` files into `ml/models/`.
5. Install dependencies and start the API:

```bash
pip install -r ml/requirements.txt
uvicorn ml.api:app --host 0.0.0.0 --port 8000
```

Check `http://127.0.0.1:8000/health`. The API returns model status and validation metrics are saved by the training script. Do not describe a metric as a test-set result; it is a validation result.

## API contract

`POST /predict` with multipart form field `file` returns scores for available models, a conservative suggestion (if a supported class exceeds the demo threshold), and a warning to verify manually. `GET /health` reports whether weights were loaded.

The browser UI has an **AI service URL** field. For local testing use `http://127.0.0.1:8000`. For a public demo, deploy this API to a trusted HTTPS host and enter that URL in the UI. GitHub Pages hosts only the static frontend; it does not run this Python API.

## Important limitations

- No trained weights are included until you run training; model weights are intentionally excluded from Git.
- Transfer learning reduces training cost but does not guarantee good generalization to local images.
- Report validation accuracy, macro-F1, confusion matrix, dataset source/licence, and known failure modes from your actual run. Never invent model performance.
- Model scores are not calibrated probabilities and should not automatically decide municipal action.
- CivicLens does not submit complaints to government portals. Authority mapping and the browser-only report tracker remain a prototype.
