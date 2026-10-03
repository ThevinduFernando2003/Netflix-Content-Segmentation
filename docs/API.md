# API

Captured from `http://127.0.0.1:8000` after `scripts/train.py` and `scripts/evaluate.py` on representation R1, k=6. `took_ms` and Prometheus counters change on every call. The JSON bodies below are one real call each, not a schema sketch.

Interactive docs: `GET /docs` returned **200**.

`/assign` transforms the saved pipeline. It does not refit. Membership probabilities come from a diagonal Gaussian mixture fit on the same matrix. That mixture is not the k-means partition, so the highest probability can sit on a different segment from `cluster_id`. The Night Docket example below does that.

## GET /health

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

## GET /ready

Returns 200 after artifacts load, and 503 before that.

```bash
curl -s http://127.0.0.1:8000/ready
```

```json
{"status":"ready"}
```

## GET /meta

```bash
curl -s http://127.0.0.1:8000/meta
```

The live body includes `k_sweep` (k=2..20), `algorithm_comparison`, `figures`, and `cluster_colors`. Validation and external scores from that response:

```json
{
  "chosen_k": 6,
  "model_version": "r1-k6-42",
  "validation_scores": {
    "silhouette": 0.12823954345490216,
    "calinski_harabasz": 872.825442102727,
    "davies_bouldin": 2.517081910022174,
    "inertia": 4847.64212247034,
    "bootstrap_ari": 0.8202726543677922,
    "bootstrap_ari_std": 0.1211317682451567
  },
  "external_validation": {
    "type": {"ari": 0.26320706871732297, "nmi": 0.4270999864555926, "n": 8790},
    "rating": {"ari": 0.06645757630441528, "nmi": 0.11652511084543829, "n": 8790},
    "primary_genre": {"ari": 0.19625323699653616, "nmi": 0.3473715789731189, "n": 8790}
  },
  "projection_caveat": "t-SNE and UMAP do not preserve global distances or cluster sizes. Colour the projection by cluster; do not read gaps or areas as structure. Clustering was done in feature space."
}
```

## GET /clusters

```bash
curl -s http://127.0.0.1:8000/clusters
```

Each object also includes `genre_lift`, `top_terms`, `representative_titles`, and `color`. Sizes from the live response:

```json
[
  {"id": 0, "name": "US Family and Action Films", "size": 1484},
  {"id": 1, "name": "International Drama Films", "size": 2494},
  {"id": 2, "name": "International TV Series", "size": 1378},
  {"id": 3, "name": "US Independent Dramas", "size": 932},
  {"id": 4, "name": "US Kids and Docuseries", "size": 1278},
  {"id": 5, "name": "US Docs and Stand-Up", "size": 1224}
]
```

## GET /clusters/{id}

```bash
curl -s http://127.0.0.1:8000/clusters/3
```

Cluster 3 is **US Independent Dramas**, 932 titles. Representative titles in that response: Winter's Bone, Short Term 12, Frozen River, Hell or High Water, The Company Men.

```bash
curl -s http://127.0.0.1:8000/clusters/99
```

```json
{"detail":"Unknown cluster id 99"}
```

Status **404**.

## GET /clusters/{id}/titles

```bash
curl -s "http://127.0.0.1:8000/clusters/0/titles?limit=1&offset=0"
```

```json
{"cluster_id":0,"total":1484,"limit":1,"offset":0,"titles":[{"show_id":"s6819","title":"G-Force","type":"Movie","rating":"PG","country":"United States","release_year":2009,"genres":["Children & Family Movies","Comedies"],"cluster_id":0,"distance_to_centroid":0.6260173173389116,"silhouette":0.15144682593882436}]}
```

## GET /titles/{show_id}/cluster

```bash
curl -s http://127.0.0.1:8000/titles/s6819/cluster
```

```json
{"show_id":"s6819","title":"G-Force","cluster_id":0,"cluster_name":"US Family and Action Films","distance_to_centroid":0.6260173173389116,"silhouette":0.15144682593882436,"model_version":"r1-k6-42"}
```

## GET /similar/{show_id}

```bash
curl -s "http://127.0.0.1:8000/similar/s6819?n=2"
```

```json
{"show_id":"s6819","cluster_id":0,"neighbours":[{"show_id":"s8011","title":"Shorts","distance":0.010676356839507852},{"show_id":"s6434","title":"Cats & Dogs: The Revenge of Kitty Galore","distance":0.059453163222916765}]}
```

## GET /projection

`sample` must be at least 10. `sample=2` returns **422**.

```bash
curl -s "http://127.0.0.1:8000/projection?sample=10"
```

The live response had `sample_size` 10, `total` 8790, and this caveat:

```text
Projection distances are not interpretable. t-SNE and UMAP do not preserve global distances or cluster area. Clusters were fit in feature space.
```

## GET /figures/{name}

Precomputed PNG files copied into `models/figures` by `scripts/evaluate.py`.

```bash
curl -s -o elbow_product.png -w "%{http_code}" http://127.0.0.1:8000/figures/elbow_product.png
```

That call returned **200**. `GET /figures/not-real.png` returned **404**:

```json
{"detail":"Figure not-real.png is not in this build"}
```

## POST /assign

Family film. `is_outlier` is false. Unknown genres is empty.

```bash
curl -s -X POST http://127.0.0.1:8000/assign -H "Content-Type: application/json" -d "{\"title\":\"The Paper Kite Club\",\"description\":\"Two children build kites, chase a lost dog, and learn what friendship costs when the summer ends.\",\"type\":\"Movie\",\"genres\":[\"Children & Family Movies\",\"Comedies\"],\"release_year\":2019,\"country\":\"United States\",\"rating\":\"PG\",\"duration_minutes\":98}"
```

```json
{"cluster_id":0,"cluster_name":"US Family and Action Films","distance_to_centroid":0.7641562562320733,"membership_probabilities":[{"cluster_id":0,"cluster_name":"US Family and Action Films","probability":0.9864272850146756},{"cluster_id":1,"cluster_name":"International Drama Films","probability":0.0},{"cluster_id":2,"cluster_name":"International TV Series","probability":0.013572714985322986},{"cluster_id":3,"cluster_name":"US Independent Dramas","probability":0.0},{"cluster_id":4,"cluster_name":"US Kids and Docuseries","probability":0.0},{"cluster_id":5,"cluster_name":"US Docs and Stand-Up","probability":1.2179961277090424e-280}],"is_outlier":false,"nearest_titles":[{"show_id":"s7009","title":"Hotel Transylvania 3: Summer Vacation","distance":0.4411611165179428},{"show_id":"s6884","title":"Goosebumps 2: Haunted Halloween","distance":0.4534474889962424},{"show_id":"s7453","title":"Middle School: The Worst Years of My Life","distance":0.45346821181039},{"show_id":"s1495","title":"We Can Be Heroes","distance":0.45409026628534704},{"show_id":"s7733","title":"Peter Rabbit","distance":0.45495930812365615}],"why":["Shares distinctive genre: Children & Family Movies","Shares distinctive genre: Comedies"],"unknown_genres":[],"model_version":"r1-k6-42","took_ms":596.699699992314}
```

Crime drama. k-means assigns segment 3. The mixture puts probability 1.0 on segment 0. Both numbers are from the same response.

```bash
curl -s -X POST http://127.0.0.1:8000/assign -H "Content-Type: application/json" -d "{\"title\":\"Night Docket\",\"description\":\"A detective and a prosecutor unravel a murder that reaches into a city political machine and its oldest families.\",\"type\":\"Movie\",\"genres\":[\"Dramas\",\"Thrillers\"],\"release_year\":2018,\"country\":\"United States\",\"rating\":\"TV-MA\",\"duration_minutes\":124}"
```

```json
{"cluster_id":3,"cluster_name":"US Independent Dramas","distance_to_centroid":0.7576281827655437,"membership_probabilities":[{"cluster_id":0,"cluster_name":"US Family and Action Films","probability":1.0},{"cluster_id":1,"cluster_name":"International Drama Films","probability":0.0},{"cluster_id":2,"cluster_name":"International TV Series","probability":0.0},{"cluster_id":3,"cluster_name":"US Independent Dramas","probability":0.0},{"cluster_id":4,"cluster_name":"US Kids and Docuseries","probability":0.0},{"cluster_id":5,"cluster_name":"US Docs and Stand-Up","probability":6.376069736556484e-100}],"is_outlier":false,"nearest_titles":[{"show_id":"s3429","title":"El Camino: A Breaking Bad Movie","distance":0.43618231203143687},{"show_id":"s4994","title":"The Outsider","distance":0.44850080166516654},{"show_id":"s3010","title":"A Fall from Grace","distance":0.4491660124057034},{"show_id":"s5208","title":"1922","distance":0.5075931438183352},{"show_id":"s7075","title":"Indiscretion","distance":0.524351980807017}],"why":["Shares distinctive genre: Dramas","Shares distinctive genre: Thrillers","Overlaps c-TF-IDF term: night"],"unknown_genres":[],"model_version":"r1-k6-42","took_ms":263.5428998619318}
```

Deliberate outlier. `is_outlier` is true. Both genres are reported as unknown.

```bash
curl -s -X POST http://127.0.0.1:8000/assign -H "Content-Type: application/json" -d "{\"title\":\"Estonian Tax Law, Underwater\",\"description\":\"A silent experimental lecture on bracket creep, filmed inside a drained reservoir, with no characters, no story, and a glossary of statutes.\",\"type\":\"Movie\",\"genres\":[\"Not A Real Genre\",\"Underwater Basket Weaving\"],\"release_year\":1900,\"country\":\"Atlantis\",\"rating\":\"UNRATED-XYZ\",\"duration_minutes\":4}"
```

```json
{"cluster_id":0,"cluster_name":"US Family and Action Films","distance_to_centroid":1.0712019630989402,"membership_probabilities":[{"cluster_id":0,"cluster_name":"US Family and Action Films","probability":0.6301161080712271},{"cluster_id":1,"cluster_name":"International Drama Films","probability":0.0},{"cluster_id":2,"cluster_name":"International TV Series","probability":0.36988389192876014},{"cluster_id":3,"cluster_name":"US Independent Dramas","probability":6.971672785424534e-15},{"cluster_id":4,"cluster_name":"US Kids and Docuseries","probability":0.0},{"cluster_id":5,"cluster_name":"US Docs and Stand-Up","probability":1.542288696774431e-32}],"is_outlier":true,"nearest_titles":[{"show_id":"s8764","title":"WWII: Report from the Aleutians","distance":0.8443584195255466},{"show_id":"s4251","title":"Pioneers: First Women Filmmakers*","distance":0.855056994896235},{"show_id":"s8206","title":"The Battle of Midway","distance":0.8610729453887793},{"show_id":"s7791","title":"Prelude to War","distance":0.8620320826729586},{"show_id":"s8661","title":"Undercover: How to Operate Behind Enemy Lines","distance":0.8642895249173391}],"why":["No distinctive genre or title-term overlap; assigned by overall feature distance."],"unknown_genres":["Not A Real Genre","Underwater Basket Weaving"],"model_version":"r1-k6-42","took_ms":271.1595001164824}
```

A description shorter than 10 characters returns **422** and names the field:

```json
{"detail":[{"type":"string_too_short","loc":["body","description"],"msg":"String should have at least 10 characters","input":"short","ctx":{"min_length":10}}]}
```

## GET /metrics

Prometheus text. A prefix of one live scrape:

```text
# HELP http_requests_total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/health",status="200"} 1.0
http_requests_total{method="GET",path="/ready",status="200"} 1.0
http_requests_total{method="GET",path="/meta",status="200"} 1.0
```
