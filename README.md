# HAPS coverage simulator

Simulates a region's ground network and a fleet of high-altitude platforms
(HAPS) that pick up traffic the ground can't carry. Regions live in
`regions/<name>/`; shared defaults are in `config.yaml`.

```bash
python -m pytest -q                                              # tests
python scripts/run_demo.py --region england                      # a few fixed ticks
python scripts/run_day.py --region england --controller greedy   # 24 h, greedy vs static
```

`run_day.py` saves every tick to `data/processed/<region>/runs/<run_id>.jsonl`.

## Replay viewer

`web/index.html` replays saved runs on a map. It is a single static file
(Leaflet from a CDN, no build step) that reads JSON exported into `web/data/`.

1. Simulate a day (writes one run per controller):

   ```bash
   python scripts/run_day.py --region england --controller greedy
   ```

2. Export the grid and the runs you want to view. Run ids are the log file
   names without `.jsonl`; with no `--runs`, every saved run is exported.

   ```bash
   python scripts/export_viewer_data.py --region england \
       --runs 20261022-192025_greedy 20261022-192025_static
   ```

   This writes `web/data/<region>/cells.geojson`, `static.json` and
   `runs/<run_id>.json` (about 8 MB per England day), plus `runs/index.json`.

3. Serve the repository root and open the viewer:

   ```bash
   python -m http.server 8000
   ```

   Then open http://localhost:8000/web/

URL options: `?region=test_island`, `?run=<run_id>`, `?frame=<n>` (start
frame), `?basemap=carto`. The default basemap is Esri's Dark Gray Canvas,
because CARTO's dark tiles now show an "API key required" image without a key.
