# ruff: noqa
# =====================================================================================
# M1.2 — EXPORT CELL: paste at the END of notebook `midmoe-vn-transfer` (after Phase 7 defs,
# you can skip the long Phase 7 experiment loop) and run on Kaggle.
#
# Produces /kaggle/working/export/ :
#   weights/model_seed{0..9}.pt     state_dict of transfer_minimal, fraction=1.0 (decision D2: fit + early stop on val)
#   model_config.json               US config (F_dim, dit_dim, ...) + deploy info
#   vn_tickers_sorted.json, vn_feature_order.json, vn_sector_graph.npy   copied from VN_DATA_DIR
#   vn_norm_params.json             converted from the VN cube-build normalisation params (see below)
#   vn_parity_ref.npz               reference scores for tests/test_parity.py:
#                                   per_seed (10, T_test, 78) computed on CPU, batch=1, seeded noise
#   vn_test_cube.npz                copy of the VN test cube
# Download the folder and put it into artifacts/vn/ of the app repo.
# =====================================================================================
import shutil, glob

EXPORT_DIR = f'{OUT_DIR}/export'
os.makedirs(f'{EXPORT_DIR}/weights', exist_ok=True)
DEPLOY_CONFIG, DEPLOY_FRACTION, NOISE_SEED = 'transfer_minimal', 1.00, 0


def seeded_forward(model, x, news_emb, news_mask, noise_seed=NOISE_SEED):
    # MUST stay identical to midmoe/predict.py::seeded_forward
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(noise_seed)
        r_final, _, _ = model(x, news_emb, news_mask)
    return r_final


def cpu_reference_preds(model, raw):
    m = copy.deepcopy(model).cpu().eval()
    out = []
    for i in range(len(raw['x'])):
        x = torch.from_numpy(raw['x'][i:i+1]); ne = torch.from_numpy(raw['news_emb'][i:i+1])
        nm = torch.from_numpy(raw['news_mask'][i:i+1])
        out.append(seeded_forward(m, x, ne, nm)[0].numpy())
    return np.stack(out)


train_sub = subsample_train(vn_fit_raw, DEPLOY_FRACTION)
ref_per_seed, export_metrics = [], []
for seed in range(10):
    set_seed(seed)
    model, loss_kwargs = CONFIGS[DEPLOY_CONFIG](seed)
    model, hist = finetune_vn_model(model, train_sub, vn_val_raw, seed=seed, **loss_kwargs)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f'{EXPORT_DIR}/weights/model_seed{seed}.pt')
    preds = cpu_reference_preds(model, vn_test_raw)
    ref_per_seed.append(preds)
    m = compute_metrics(list(preds), list(vn_test_raw['y']), k=5)
    export_metrics.append({'seed': seed, 'epochs': len(hist), **m})
    print(f"seed={seed} epochs={len(hist)} SR={m['SR']:+.4f} nDCG@5={m['nDCG@5']:.4f}")

# Per-seed test SR printed by the original Phase 7 run (cell 20, transfer_minimal frac=1.00; mean 0.0410 = paper
# Table 10). Identical values => the retrained models reproduce the paper's models; small gaps are GPU
# non-determinism; large gaps mean something differs (data, config, seed handling) -> report before deploying.
PAPER_SR = [0.0326, 0.0402, -0.0080, -0.0130, 0.1232, 0.0420, 0.0238, 0.0773, 0.0008, 0.0916]
print("\nseed | SR paper | SR export | diff")
for row, ref in zip(export_metrics, PAPER_SR):
    print(f"{row['seed']:4d} | {ref:+.4f}  | {row['SR']:+.4f}   | {row['SR'] - ref:+.4f}")
print(f"mean | {np.mean(PAPER_SR):+.4f}  | {np.mean([r['SR'] for r in export_metrics]):+.4f}")

ref_per_seed = np.stack(ref_per_seed)  # (10, T_test, 78)
np.savez_compressed(f'{EXPORT_DIR}/vn_parity_ref.npz', per_seed=ref_per_seed,
                    window_end_date=vn_test_raw['window_end_date'], label_date=vn_test_raw['label_date'],
                    noise_seed=NOISE_SEED)
pd.DataFrame(export_metrics).to_csv(f'{EXPORT_DIR}/export_metrics.csv', index=False)

# Sanity check vs the predictions reported in the paper (vn_preds.npz, GPU, batch=8, unseeded noise):
# not bit-identical, but rankings should agree closely.
key = (DEPLOY_CONFIG, DEPLOY_FRACTION)
if 'PRED_VN' in globals() and key in PRED_VN:
    from scipy.stats import spearmanr
    old = np.stack(PRED_VN[key])
    rho = np.mean([spearmanr(old[s, t], ref_per_seed[s, t]).correlation
                   for s in range(10) for t in range(old.shape[1])])
    print(f"Mean Spearman(old paper preds, exported) = {rho:.4f} | max abs diff = {np.abs(old - ref_per_seed).max():.2e}")

# ---- configs / lookups ----
cfg = dict(us_model_config)
cfg.update({'deploy_config': DEPLOY_CONFIG, 'deploy_fraction': DEPLOY_FRACTION, 'market_conv_type': 'conv3d',
            'T_WINDOW': int(T_WINDOW), 'N': int(N_VN), 'D': int(D), 'news_emb_dim': int(EMB_DIM),
            'noise_seed': NOISE_SEED})
with open(f'{EXPORT_DIR}/model_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
for fn in ['vn_tickers_sorted.json', 'vn_feature_order.json', 'vn_sector_graph.npy', 'vn_test_cube.npz']:
    shutil.copy(f'{VN_DATA_DIR}/{fn}', f'{EXPORT_DIR}/{fn}')
for fn in glob.glob(f'{VN_DATA_DIR}/*icb*') + glob.glob(f'{VN_DATA_DIR}/*sector_map*'):
    shutil.copy(fn, EXPORT_DIR)

# ---- normalisation params -> JSON ----
# The VN cube-build notebook is expected to save a dict like the US one:
#   {'clip_min', 'clip_max', 'mins', 'maxs', 'feature_cols'}  (arrays shaped (1,1,D)), in its OWN feature order.
# We re-order to vn_feature_order.json. ADJUST THE FILE NAME if it differs.
cands = glob.glob(f'{VN_DATA_DIR}/*norm*') + glob.glob(f'/kaggle/input/**/vn*normalize_params*', recursive=True)
print("Normalisation param candidates:", cands)
npz = [c for c in cands if c.endswith('.npy')]
if npz:
    p = np.load(npz[0], allow_pickle=True).item()
    src_cols = list(p['feature_cols'])
    idx = [src_cols.index(c) for c in vn_feature_order]
    out = {'feature_cols': list(vn_feature_order)}
    for k in ['clip_min', 'clip_max', 'mins', 'maxs']:
        out[k] = np.asarray(p[k]).reshape(-1)[idx].astype(float).tolist()
    with open(f'{EXPORT_DIR}/vn_norm_params.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved vn_norm_params.json:", out)
elif any(c.endswith('.json') for c in cands):
    shutil.copy([c for c in cands if c.endswith('.json')][0], f'{EXPORT_DIR}/vn_norm_params.json')
else:
    print("!!! Không tìm thấy normalisation params VN — cần lấy từ notebook midmoe-vn-cube-build")

print(sorted(os.listdir(EXPORT_DIR)))
shutil.make_archive(f'{OUT_DIR}/midmoe_vn_artifacts', 'zip', EXPORT_DIR)
print("Download: /kaggle/working/midmoe_vn_artifacts.zip")
