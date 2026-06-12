# LIDAR Survey Explorer

Password-protected web app for visualising LIDAR survey data collected with WingtraOne. Built with Streamlit, Potree, and Plotly — deployed on Streamlit Community Cloud with survey data streamed from Cloudflare R2.

## Features

### Point Cloud Viewer (Potree)
- **Full-resolution streaming** — 131M+ points rendered progressively via Potree's octree LOD system. No downsampling, no pre-loading the entire dataset.
- **Elevation colouring** — rainbow gradient mapped to height (blue = low, red = high)
- **Adaptive point sizing** — points scale automatically based on zoom level
- **Adjustable point budget** — 500K to 5M points rendered simultaneously (sidebar slider)
- **Black background** — clean presentation for ellipsoidal height data without basemap alignment issues

#### Potree Navigation Controls

| Action | Mouse | 
|--------|-------|
| **Rotate** | Left-click + drag |
| **Pan** | Right-click + drag |
| **Zoom** | Scroll wheel |
| **Pan** (alt) | Middle-click + drag |

### Digital Terrain Model
- **3D surface** with hillshade lighting (directional light from northwest)
- **Adjustable vertical exaggeration** — 0.5× to 5× (1× = true scale)
- **Multiple colorscales** — Turbo, Viridis, Plasma, Inferno, Earth, RdYlBu
- **Contour lines** — selectable intervals (1m, 2m, 5m, 10m, 20m), filtered by survey relief
- **Contour export** — prepare and download contours as a shapefile (.shp/.shx/.dbf/.prj in a zip), georeferenced in EPSG:4326

### Downloads
- Raw LAS file (full resolution, via presigned URL from R2)
- Raw DTM GeoTIFF (via presigned URL from R2)
- Contour shapefile (generated on-demand at the selected interval)

## Architecture

```
Browser
  ├── Potree viewer (iframe, streams octree tiles from R2)
  ├── Plotly 3D surface (DTM tab)
  └── PyDeck fallback (surveys without Potree tiles)

Streamlit Community Cloud
  └── lidar_app.py (serves HTML, reads meta/parquet/npy from R2)

Cloudflare R2 (public bucket)
  └── lidar/<slug>/
        ├── potree/
        │     ├── metadata.json
        │     ├── hierarchy.bin
        │     └── octree.bin
        ├── ground.parquet        (or points.parquet if unclassified)
        ├── nonground.parquet
        ├── dtm.npy
        ├── dtm_meta.json
        ├── raw/<slug>.las
        ├── raw/<slug>_dtm.tif
        └── meta.json             ← readiness gate

  └── potree-assets/              (shared, one-time upload)
        ├── potree.js
        ├── potree.css
        ├── libs/other/BinaryHeap.js
        ├── lazylibs/
        ├── resources/
        └── workers/
```

### Potree Integration Notes

Potree runs inside a Streamlit `components.v1.html()` srcdoc iframe. This required several workarounds:

- **Worker monkey-patch** — srcdoc iframes have an opaque `null` origin, so `new Worker(crossOriginURL)` silently fails. A patch detects the null origin and creates all Workers from Blob URLs instead, rewriting relative `importScripts()` paths to absolute R2 URLs.
- **No `loadGUI()`** — Potree's `viewer.loadGUI()` tries to AJAX-load `sidebar.html` from R2, which hangs in the srcdoc context. Since we use the Streamlit sidebar for controls, `loadPointCloud()` is called directly (it's already on the namespace with all CDN dependencies loaded).
- **`_awaitSize()`** — the Viewer is created only after the iframe container has non-zero pixel dimensions, preventing WebGL framebuffer errors.
- **CDN dependencies** — THREE.js r124, jQuery, TWEEN, i18next, jstree, proj4, d3, spectrum are loaded from CDN before potree.js. BinaryHeap.js is loaded from R2.

## Repo Structure

```
lidar_app.py              ← Streamlit entry point
lidar_r2.py               ← R2 read-only layer (list surveys, load bundles, presigned URLs)
requirements.txt
.gitignore
README.md
assets/
  wingtra_logo.png
.streamlit/
  secrets.toml            ← NOT committed (local + Community Cloud Secrets)
tools/
  lidar_preprocess.py     ← full pipeline: LAS → parquet + DTM + Potree → R2
  lidar_dtm_only.py       ← standalone DTM processor (add DTM to existing bundle)
  lidar_patch_meta.py     ← patch meta.json fields on R2
  lidar_diagnose.py       ← diagnostic tool for R2 bundle inspection
```

## Preprocessing

### Full pipeline

Edit configuration at the top of `tools/lidar_preprocess.py`:

```python
LAS_PATH          = r"C:\path\to\survey.las"
DTM_TIF_PATH      = r"C:\path\to\dtm.tif"       # optional
SLUG              = "my_survey"
SURVEY_NAME       = "My Survey — Block 7"
POTREE_CONVERTER  = r"C:\tools\PotreeConverter\PotreeConverter.exe"
```

Run:
```
python tools/lidar_preprocess.py
```

Steps:
1. Inspect LAS (CRS, classification codes)
2. Reproject to WGS84 + reservoir-sample for PyDeck fallback
3. Build parquet layers (ground/nonground or all points)
4. Package DTM (decimate to 800px max, nearest resampling)
5. Upload raw LAS + DTM GeoTIFF to R2
6. Run PotreeConverter → upload octree tiles to R2
7. Upload processed bundle + meta.json (last — readiness gate)

### Add DTM to existing bundle

If the full pipeline was run without a DTM path:

```
python tools/lidar_dtm_only.py
```

### PotreeConverter

Download from [PotreeConverter releases](https://github.com/potree/PotreeConverter/releases). No install — standalone executable. Built into the preprocessing pipeline (Step 6), or run manually:

```
PotreeConverter.exe input.las -o output_folder
```

## Secrets

`.streamlit/secrets.toml` (local) and Streamlit Community Cloud Secrets:

```toml
app_password  = "your_password"
r2_public_url = "https://pub-<hash>.r2.dev"

[r2]
account_id = "..."
access_key = "..."    # read-only token
secret_key = "..."
bucket     = "ptpn-bucket"

[r2_write]            # local only — NOT on Community Cloud
access_key = "..."
secret_key = "..."
```

## R2 Bucket Setup

- **Public access** enabled via Cloudflare R2 development URL (`pub-<hash>.r2.dev`)
- **CORS policy** must allow all origins (srcdoc iframes send `Origin: null`):

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

## Requirements

```
streamlit
pydeck
plotly
pandas
numpy
boto3
pyarrow
pyshp
matplotlib
```

## Deployment

Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud). Push to `main` branch triggers automatic redeploy. Apps sleep after 12 hours of inactivity — use a GitHub Actions cron job to keep alive:

```yaml
name: Keep alive
on:
  schedule:
    - cron: '0 */8 * * *'
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -sL -o /dev/null https://your-app.streamlit.app/ || true
```
