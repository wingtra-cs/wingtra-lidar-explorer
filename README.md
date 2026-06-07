# LiDAR Survey Explorer

A Streamlit web app for visualising WingtraRay LiDAR surveys. Displays classified or unclassified point clouds via PyDeck and terrain surfaces (DTM) from LiDAR360 via Plotly. Survey bundles are stored in Cloudflare R2 and loaded on demand.

---

## Repo structure

```
wingtra-lidar-explorer/
  lidar_app.py            Streamlit entry point
  lidar_r2.py             R2 read-only access layer
  requirements.txt        App runtime dependencies
  .gitignore
  assets/
    wingtra_logo.png      Wingtra logo (copy from multispectral repo)
  .streamlit/
    secrets.toml          Credentials — NOT committed
    config.toml           Optional theme overrides
  tools/
    lidar_preprocess.py   Local preprocessing pipeline (laptop → R2)
    lidar_diagnose.py     One-shot diagnostic on a new LAS file
    lidar_patch_meta.py   Patch meta.json after a manual rclone upload
```

---

## Setup

### 1. Secrets

Create `.streamlit/secrets.toml` (never committed):

```toml
[r2]
account_id = "..."       # Cloudflare R2 account ID
access_key = "..."       # READ-ONLY token  ← used by the app
secret_key = "..."
bucket     = "ptpn-bucket"

[r2_write]
access_key = "..."       # READ & WRITE token  ← used only by tools/
secret_key = "..."
```

### 2. Logo

Copy `wingtra_logo.png` from the multispectral repo into `assets/`.

### 3. Local install

```bash
pip install streamlit pydeck plotly numpy pandas pyarrow boto3

# Additional packages for the tools/ scripts:
pip install laspy pyproj rasterio scipy
# Python < 3.11 also needs: pip install tomli
```

---

## Running locally

```bash
streamlit run lidar_app.py
```

---

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (secrets.toml is gitignored).
2. Connect the repo in the Streamlit Community Cloud dashboard.
3. Set entry point to `lidar_app.py`.
4. Add secrets via the dashboard "Secrets" box (same TOML format as above).
5. Deploy.

After the first successful build, freeze exact dependency versions from the build log and replace `>=` bounds in `requirements.txt` with `~=` pinned versions.

---

## Preprocessing workflow

For each new survey:

**1. Inspect the LAS file**
```bash
python tools/lidar_diagnose.py /path/to/survey.las
```
Confirms CRS and classification codes before running the full pipeline.

**2. Edit tools/lidar_preprocess.py**
Set the config block at the top:
```python
LAS_PATH     = r"C:\...\survey.las"
DTM_TIF_PATH = r"C:\...\dtm_from_lidar360.tif"   # leave blank if unavailable
SLUG         = "client_block_date"                 # unique identifier
SURVEY_NAME  = "Client — Block X — DD Mon YYYY"   # display name in the app
```

**3. Run**
```bash
python tools/lidar_preprocess.py
```
This reads, reprojects (UTM → WGS84), reservoir-samples to ~300K display points per layer, packages the DTM, uploads everything to R2, and writes `meta.json` last as a readiness flag.

**4. Refresh the app**
Click "⟳ Refresh surveys" in the sidebar.

---

## Classification modes

The preprocessing script auto-detects which mode to use:

| LAS content | Mode | R2 files |
|---|---|---|
| All code 0 (unclassified) | `unclassified` | `points.parquet` |
| Ground (code 2) present | `classified` | `ground.parquet` + `nonground.parquet` |

No self-classification is performed. For classified output, run ground classification in LiDAR360 first, then re-run the preprocessing script.

---

## R2 bundle structure

```
lidar/<slug>/
  points.parquet          unclassified mode
  ground.parquet          classified mode
  nonground.parquet       classified mode
  dtm.npy                 decimated elevation grid (float32, H×W)
  dtm_meta.json           bounds, z range, grid shape, source resolution
  raw/<slug>.las          raw LAS (presigned download)
  raw/<slug>_dtm.tif      raw DTM GeoTIFF (presigned download)
  meta.json               readiness flag — written last by preprocess script
```

---

## Patching an existing bundle

If raw files were uploaded manually with rclone after preprocessing:

```bash
# Upload with standardised names
rclone copyto survey.las "r2:ptpn-bucket/lidar/<slug>/raw/<slug>.las" --progress --s3-no-check-bucket
rclone copyto DTM.tif    "r2:ptpn-bucket/lidar/<slug>/raw/<slug>_dtm.tif" --progress --s3-no-check-bucket

# Patch meta.json to add the download keys
python tools/lidar_patch_meta.py
```

---

## Notes

- The app uses `map_provider=None` in PyDeck — dark background, no Mapbox token required. Satellite background can be added later with a Mapbox token in secrets.
- DTM surface axes are in metres from the survey centre so horizontal and vertical dimensions are proportionally correct.
- Presigned download URLs expire after 15 minutes. Clicking the download button generates a fresh URL each time.
