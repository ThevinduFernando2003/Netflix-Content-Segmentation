# Deploy

No public URL was published from this workspace. `HF_TOKEN` is not set, and there is no Hugging Face credential file. Do not invent a Space URL. The steps below are what to run once a token exists.

## What gets deployed

Two containers from the same image:

- API, port 8000, `uvicorn api.main:app`
- UI, port 8501, Streamlit, `API_BASE_URL` pointing at the API

The image does not install UMAP. `models/projection_2d.parquet` must already exist. Build after `make train`.

## Hugging Face Spaces (Docker)

1. Create a Space with the Docker SDK. The README in this repo is the project README; the Space needs `Dockerfile` at its root, or a copy of this repository.
2. Put `HF_TOKEN` in the Space secrets if the Space is private. The app itself does not need a token to serve.
3. The Docker Space runs one public port. A single Space cannot publish 8000 and 8501 as two public URLs. Practical layout:
   - One Space for the API. Set `PORT` to the Space port (Hugging Face sets `PORT` for you). The Dockerfile already honours `PORT`.
   - A second Space, or a second service, for Streamlit, with `API_BASE_URL` set to the API Space URL, for example `https://<user>-netflix-segmentation-api.hf.space`.
4. Confirm from outside the host:

```bash
curl https://<api-host>/health
curl https://<api-host>/meta
curl https://<api-host>/clusters
curl -X POST https://<api-host>/assign \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"The Paper Kite Club\",\"description\":\"Two children build kites and learn what friendship costs when the summer ends.\",\"type\":\"Movie\",\"genres\":[\"Children & Family Movies\"],\"release_year\":2019,\"country\":\"United States\",\"rating\":\"PG\",\"duration_minutes\":98}"
```

Paste those responses into the README only after they succeed. Until then the README says the service is not published.

## Environment

Copy `.env.example`. Set `CORS_ORIGINS` to the UI origin. Do not commit tokens.

## Other hosts

Render, Fly.io, and Railway can run `docker compose` or the two services separately. Render's free tier spins the service down when idle; the first request after idle is slow, and `/ready` stays 503 until the model loads. That cold start is a property of the free tier, not of the model.

## Local image check

```bash
make train
docker compose up --build
curl http://127.0.0.1:8000/health
```

`models/` is gitignored. A clean clone needs the CSV and `make train` before `docker compose up`.
