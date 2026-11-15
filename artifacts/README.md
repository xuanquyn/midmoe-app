# artifacts/vn

Model weights and lookups for the deployed VN model. **Do not commit `.pt` / `.npz` files.**

Produce them by pasting `notebooks/export_vn_weights.py` at the end of the Kaggle notebook
`midmoe-vn-transfer`, running it, downloading `midmoe_vn_artifacts.zip` and unzipping it here.

| File | Format |
|---|---|
| `weights/model_seed{0..9}.pt` | `state_dict` of `MarketMoE` (transfer_minimal, fraction 1.0, conv3d) |
| `model_config.json` | `{"F_dim", "dit_dim", "dit_heads", "dit_blocks", "num_middle", "num_bottom", "diffusion_K", ...}` |
| `vn_norm_params.json` | `{"feature_cols": [7], "clip_min": [7], "clip_max": [7], "mins": [7], "maxs": [7]}` in feature order |
| `vn_tickers_sorted.json` | list of 78 tickers, sorted by ICB sector (defines the N axis) |
| `vn_feature_order.json` | the 7 indicator names (defines the D axis) |
| `vn_sector_graph.npy` | (78, 78) 0/1 same-sector adjacency |
| `icb_industry.csv` | ticker metadata: `ticker, name, exchange, icb_sector` |
| `vn_test_cube.npz` | VN test cube (for the parity test) |
| `vn_parity_ref.npz` | `per_seed (10, T_test, 78)` reference scores computed on CPU with seeded noise |
| `vn_preds.npz` | (optional) paper predictions from the Phase 7 run, used as a ranking sanity check |

Window length `T` and news embedding size are inferred from the weights.
