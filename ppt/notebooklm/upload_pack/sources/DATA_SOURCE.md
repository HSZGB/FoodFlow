# Data Source

## TRD

FoodFlow uses the Takeout Recommendation Dataset (TRD) as the primary public dataset.

- Title: Takeout Recommendation Dataset (TRD) from Meituan Takeout app
- DOI: `10.5281/zenodo.8025855`
- URL: <https://zenodo.org/records/8025855>
- License: CC-BY-4.0
- Period: 2021-03-01 to 2021-03-28
- Area: 11 commercial districts in Beijing

The implementation downloads the text files needed for recommendation experiments and skips `graph.bin` by default because this project does not depend on DGL.

The current committed experiment outputs are generated from the full TRD training text file. `docs/DATA_AUDIT.md` and `outputs/results/data_audit.json` record the evidence: all required raw files are present, `orders_train.txt` has 1,068,495 rows, and `orders_train.csv` also has 1,068,495 rows. For faster local iteration, `make preprocess` still supports a fixed-seed 50,000-order sample; `make preprocess-full` uses the full training set.

## Files Used

- `users.txt`: user attributes.
- `pois.txt`: restaurant attributes.
- `spus.txt`: food attributes.
- `orders_train.txt`: train order-to-restaurant interactions.
- `orders_test_poi.txt`: test order context.
- `orders_poi_test_label.txt`: restaurant labels for offline evaluation.

## Synthetic Rider Data

TRD does not include full rider state and dispatch records. FoodFlow therefore synthesizes rider position, online state, load, reliability, and income using a fixed random seed. The generated rider data is used only for delivery-feasibility simulation and is reported as synthetic proxy data.
