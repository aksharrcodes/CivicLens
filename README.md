# CivicLens 🏙️

**See the problem. Find who owns it. Track what happens next.**

CivicLens is a prototype for Lumina'26 — **Hack the Unseen** at JIIT Noida. It explores a citizen-facing workflow for documenting visible civic issues, drafting a complaint, and making unresolved reports easier to track.

## Demo

- **Frontend:** GitHub Pages can host the static website after it is enabled in repository Settings → Pages.
- **ML API:** A separate Python API is provided under [`ml/`](ml/README.md). The static website does not run Python itself. The image-analysis feature works only when a trained model API is available and its URL is configured in the page.

## Features

- Upload and preview an issue photo.
- Manual issue categories: pothole/damaged road, broken streetlight, garbage dump, and open drain.
- Optional trained-model suggestions for **pothole vs non-pothole** and **garbage vs non-garbage** when the ML API is configured.
- Generate editable English or Hindi complaint drafts.
- Illustrative authority routing by demo zone.
- Interactive OpenStreetMap demo map, sample reports, status changes, and browser-local report persistence.

## Machine learning

The repository includes a reproducible MobileNetV3-Small transfer-learning pipeline and a [Google Colab training notebook](CivicLens_Model_Training.ipynb). It trains two independent binary classifiers using the public [QR4Change civic issue dataset](https://data.mendeley.com/datasets/zndzygc3p3/1), subject to the dataset's stated CC BY 4.0 terms and attribution requirements. The dataset is not bundled in this repository.

**Important:** trained weights are not included in this source package. You must run the notebook with the dataset to produce weights and validation metrics. Until that is done and the resulting `.pth` files are installed in `ml/models/`, the API cannot make predictions. Do not claim that the model is trained or report accuracy/F1 values until you have completed a run and inspected the generated metrics.

### Quick start for the ML API

```bash
pip install -r ml/requirements.txt
uvicorn ml.api:app --host 127.0.0.1 --port 8000
```

The API will report that model weights are missing until trained weights are copied into `ml/models/`. See [`ml/README.md`](ml/README.md) and the notebook for the full training workflow. For a public demo, deploy the API to an HTTPS host and set that URL in the website's **AI service URL** field. GitHub Pages only hosts the static frontend.

## Run the frontend locally

```bash
python -m http.server 8001
```

Open `http://localhost:8001`. The map needs internet access. To use the trained model locally, start the API separately and keep the API URL as `http://127.0.0.1:8000`.

## Demo flow

1. Upload a sample civic-issue image.
2. If the ML API has trained weights, request a prediction; otherwise select the category manually.
3. Select severity and a demo zone.
4. Generate and review the complaint draft.
5. Add the report to the local dashboard and show the map/status workflow.

## Limitations and responsible use

- This is a prototype, not a deployed municipal grievance system.
- No complaint is submitted to a government portal.
- Authority routing is illustrative and must be verified against official jurisdiction data.
- Reports and status updates are stored in the current browser's `localStorage`; they are not shared between visitors.
- The bundled classifier training pipeline supports pothole and garbage categories only. It does not classify streetlights or open drains.
- Model outputs can be wrong, scores are not calibrated probabilities, and a human must verify each suggestion.
- Sample markers, counts, and statuses are illustrative demo data.

## Team

Harshit Kumar · Aksharr Soni · Shreya Gupta

Built for **Lumina'26 — Hack the Unseen**.

## License

MIT for the original project code; third-party datasets, libraries, map tiles, and model weights retain their respective licences and terms. See [LICENSE](LICENSE).
