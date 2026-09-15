# Freedom Score API

Flask API for scoring one campaign idea.

## Endpoint

`POST /score_campaign`

Request body:

```json
{"idea": "A beverage brand creates vending machines that trade cans for recycled bottles."}
```

The API only accepts the `idea` field. `baseline_full.json` and
`basicness_detector_model.pkl` are loaded by the service.

## Local Run

```bash
python3 app.py
```

## Google Cloud Run Deploy

Create a Secret Manager secret named `google-cloud-api-key` containing the
Gemini API key expected by `functions.py`:

```bash
printf '%s' "$GOOGLE_CLOUD_API_KEY" | gcloud secrets create google-cloud-api-key --data-file=-
```

Grant the Cloud Run runtime service account access to the secret if your
project does not already allow it.

Deploy with Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml
```

The default service name is `freedom-score-api` in `us-central1`. Override
Cloud Build substitutions if needed:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions _SERVICE=freedom-score-api,_REGION=us-central1,_REPOSITORY=cloud-run-source-deploy,_GEMINI_API_KEY_SECRET=google-cloud-api-key
```
