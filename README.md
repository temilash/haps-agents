# HAPS coverage simulator

Simulates a region's ground network and a fleet of high-altitude platforms
(HAPS) that pick up traffic the ground can't carry. Regions live in
`regions/<name>/`; shared defaults are in `config.yaml`.

```bash
python -m pytest -q                                   # tests
python scripts/run_demo.py                            # a few fixed ticks (default region: uk)
python scripts/run_day.py --controller greedy         # 24 h, greedy vs static
python scripts/run_day.py --region england --haps 10  # another region, 10 HAPS
```

Regions: `uk` (default; all four countries, WorldPop population), `england`,
and `test_island`. `--haps N` overrides `fleet.count` (0 works). `run_day.py`
saves every tick to `data/processed/<region>/runs/<run_id>.jsonl`; the run id
ends in the controller and HAPS count, e.g. `20261022-221742_greedy_h6`.

The uk population raster is gitignored; download it to
`regions/uk/population.tif` from
https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/GBR/gbr_ppp_2020_1km_Aggregated_UNadj.tif
(without it a synthetic population is used).

## Live mode

One command runs the simulation server and the viewer:

```bash
uvicorn api.server:app --reload
```

Then open http://localhost:8000. The page opens in **Live** mode when the
server is there (switch with the Replay | Live toggle, or `?mode=replay`).

- **Top bar:** region, controller and HAPS count, then Reset; Play / Pause /
  Step; speed in ticks per second; sim time and tick time.
- **Tools** (side panel; Esc returns to pan): *Add HAPS here* and *Add
  gateway* add one where you click; *Outage* uses the radius and tower-status
  sliders above it. *Clear outages* and *Cut subsea cables* are next to them.
- **HAPS:** + adds one (auto-placed), − removes the highest-numbered. Drag a
  HAPS to send it somewhere: it is pinned (📌) and controllers leave it alone
  until you Unpin it from its popup, which also has Remove.
- **Gateways:** drag to move; the popup has Remove. A gateway in an area cut
  off from the main network is drawn red and carries nothing.
- Cells outlined red are isolated from the largest hub's ground network.

Changes apply from the next tick. Errors from the server show as a toast.
Every live session is logged to `data/processed/<region>/runs/live_<timestamp>.jsonl`
and, when you Reset or stop the server, exported for the Replay viewer.

The server starts on the default region (`uk`); set `HAPS_REGION=england` to
start elsewhere. The REST API is documented at http://localhost:8000/docs.

## Replay viewer

`web/index.html` replays saved runs on a map. It is a single static file
(Leaflet from a CDN, no build step) that reads JSON exported into `web/data/`.

1. Simulate a day (writes one run per controller):

   ```bash
   python scripts/run_day.py --controller greedy
   ```

2. Export the grid and the runs you want to view. Run ids are the log file
   names without `.jsonl`; with no `--runs`, every saved run is exported.

   ```bash
   python scripts/export_viewer_data.py            # all saved uk runs
   python scripts/export_viewer_data.py --region england
   ```

   This writes `web/data/<region>/cells.geojson`, `static.json` and
   `runs/<run_id>.json` (about 6 MB per uk day), plus `runs/index.json`.

3. Serve the repository root and open the viewer:

   ```bash
   python -m http.server 8000
   ```

   Then open http://localhost:8000/web/

URL options: `?region=england` (default `uk`), `?run=<run_id>`, `?frame=<n>` (start
frame), `?basemap=carto`. The default basemap is Esri's Dark Gray Canvas,
because CARTO's dark tiles now show an "API key required" image without a key.

## Data and attribution

- Population: WorldPop (www.worldpop.org), *Global 2000-2020 1 km
  UN-adjusted population counts*, United Kingdom, 2020. Licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Boundaries: Office for National Statistics, *Countries (December 2023)
  Boundaries UK BUC*. Source: Office for National Statistics licensed under the
  [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
  Contains OS data © Crown copyright and database right 2023.
- Basemap tiles: Esri, HERE, Garmin, © OpenStreetMap contributors.

`regions/uk/backbone.yaml` and `regions/england/backbone.yaml` are
illustrative corridors, not any real operator's network.
