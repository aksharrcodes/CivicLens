# Publish CivicLens on GitHub and GitHub Pages

## A. Create the repository

1. Sign in at https://github.com/ and click **+ → New repository**.
2. Repository name: **CivicLens** (exact capitalization recommended).
3. Description: **Civic issue reporting, authority-routing and complaint drafting prototype | Lumina’26**
4. Set **Public** so judges can see the code.
5. Do **not** select "Add a README", "Add .gitignore", or "Choose a license"; these files are already provided.
6. Click **Create repository**.

## B. Upload files without Git commands

1. On the new repository page, choose **uploading an existing file** (or **Add file → Upload files**).
2. **Extract** the ZIP you downloaded first.
3. Drag all the files **inside** `CivicLens_GitHub_Ready` into the GitHub upload area. Do not upload the ZIP itself or the enclosing folder.
4. GitHub's web upload interface may not show hidden files such as `.nojekyll` or `.gitignore` in some file pickers; these are optional and can be skipped.
5. Click **Commit changes**. Confirm `index.html` and `README.md` appear at the repository root.

Your source code link will be:
`https://github.com/YOUR-USERNAME/CivicLens`

## C. Turn on the live website

1. In your repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Choose branch **main**, folder **/(root)**, and click **Save**.
4. Wait a few minutes, then refresh **Settings → Pages** to find the published URL.
5. Expected format: `https://YOUR-USERNAME.github.io/CivicLens/`. The exact link appears on GitHub after deployment.
6. Open it in an incognito/private window and test upload, manual category, complaint, and map.

## D. Before submitting

- Edit README's **Live demo** section to replace `YOUR-USERNAME` with your actual username and commit the change.
- Copy both repository URL and live demo URL to your submission.
- If the live website is initially 404, wait a few minutes, check Pages settings, and verify `index.html` is at the root.
- The repository now includes a MobileNetV3 transfer-learning training pipeline and Colab notebook. The model weights are **not trained or bundled** yet. Run `CivicLens_Model_Training.ipynb` with the public dataset before claiming model performance.
- The ML API is separate from GitHub Pages. To use AI in a public demo, train weights, deploy `ml/api.py` to an HTTPS host with the weights installed, then enter its URL in the website. If you do not complete that deployment, use manual issue selection in the judge demo and explain that the ML pipeline is provided for reproducible training.

## E. Train the image models

1. After the repository is public, open `CivicLens_Model_Training.ipynb` in Google Colab.
2. Replace `YOUR-USERNAME` in the first code cell with the GitHub username.
3. Download the QR4Change dataset from https://data.mendeley.com/datasets/zndzygc3p3/1 and arrange the images in the four class folders described in `ml/README.md`.
4. Run the notebook, review the actual validation metrics, and download `civiclens_models.zip`.
5. Extract the two `.pth` files into `ml/models/` on the API host. Install dependencies with `pip install -r ml/requirements.txt` and start with `uvicorn ml.api:app --host 0.0.0.0 --port 8000`.
6. Check `/health` and test `/predict` before presenting the AI feature. GitHub Pages alone cannot host this Python API.
