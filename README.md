# CivicLens 🏙️

**See the problem. Find who owns it. Track what happens next.**

CivicLens is a prototype for Lumina'26 — **Hack the Unseen** at JIIT Noida. It explores a citizen-facing workflow for documenting visible civic issues, drafting complaints, and making unresolved reports easier to track.

## Demo

- **Frontend:** The static website can be hosted on GitHub Pages after enabling it in repository Settings → Pages.
- **AI prototype:** Pothole image classification through a separate TensorFlow/Keras model served using FastAPI.
- **Important:** The AI service runs separately from the website. The Colab runtime and HTTPS tunnel must remain active for live AI predictions.

## Features

- Upload and preview a civic-issue photo.
- Select issue categories: pothole/damaged road, broken streetlight, garbage dump, and open drain.
- Request experimental AI suggestions for pothole vs. non-pothole classification when the Keras API is available.
- Generate editable complaint drafts in English or Hindi.
- Illustrative authority routing by demo zone.
- Interactive OpenStreetMap demo map.
- Sample reports, status changes, and browser-local report persistence.

## Machine Learning

The current working AI demonstration uses a **TensorFlow/Keras MobileNetV2 model** for binary pothole image classification.

The model was trained using a sample dataset and tested through a separate FastAPI service in Google Colab.

### Validation results

The reported validation run achieved:

- **Accuracy:** 97.5%
- **Precision:** 100%
- **Recall:** 95%
- **AUC:** 0.9972

These results are preliminary and are based on 120 validation images. They do not guarantee real-world performance, and further testing on an independent, diverse dataset is required.

**Current AI scope:** Pothole vs. non-pothole classification only. Garbage classification is not integrated into the current working AI demonstration.

## Running the AI Demo

The tested AI service uses a trained model named `civiclens_pothole_model.keras` and a FastAPI server running separately from the frontend.

1. Open the configured Google Colab notebook.
2. Load the trained Keras model.
3. Start the FastAPI service on port `8000`.
4. Expose the service using an HTTPS tunnel.
5. Copy the current public HTTPS URL into the website's **AI service URL** field.
6. Upload a pothole image and request a prediction.

Check the service health endpoint by opening:

`https://YOUR-CURRENT-TUNNEL-URL/health`

The expected response indicates that the API is running and the pothole model is loaded.

**Note:** Replace `YOUR-CURRENT-TUNNEL-URL` with the active tunnel URL. The URL is temporary and may change whenever the tunnel restarts. Never publish your ngrok authentication token.

### Repository API note

The repository also contains an older PyTorch/MobileNetV3 training and API pipeline that expects `.pth` model weights. This is separate from the Keras/MobileNetV2 service used for the tested pothole demonstration. The older API requires its own compatible trained weights before it can make predictions.

## Run the Frontend Locally

From the project root, run:

```bash
python -m http.server 8001
```

Open `http://localhost:8001` in your browser.

The map requires internet access. To use an API running on the same computer, configure the AI service URL as `http://127.0.0.1:8000`. For the Colab demonstration, use the active public HTTPS tunnel URL instead.

## Demo Flow

1. Upload a sample pothole image.
2. Request an AI prediction while the Keras API is running.
3. Select the issue category, severity, and demo zone.
4. Generate and review the complaint draft.
5. Add the report to the browser-local dashboard.
6. Demonstrate the sample map and report status workflow.

If the AI service is unavailable, use the manual category selection to demonstrate the remaining frontend features.

## Limitations and Responsible Use

- CivicLens is a prototype, not a deployed municipal grievance system.
- No complaint is submitted to a government portal.
- Authority routing is illustrative and must be verified against official jurisdiction data.
- Reports and status updates are stored in the current browser's `localStorage`; they are not shared between visitors.
- The current working AI demo supports pothole classification only.
- The separate training pipeline includes pothole and garbage classifier workflows, but this does not mean both classifiers are integrated into the live demo.
- Model outputs may be incorrect and must be reviewed by a person.
- The reported validation metrics are preliminary and should not be treated as proof of production-level accuracy.
- Sample map markers, counts, and statuses are illustrative demo data.

## Team

- Harshit Kumar
- Aksharr Soni
- Shreya Gupta

Built for **Lumina'26 — Hack the Unseen**, JIIT Noida.

## License

MIT for the original project code. Third-party datasets, libraries, and map tiles remain subject to their respective licenses and attribution requirements.
