import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# 0. Load raw 
# -------------------------------------------------------------------
df_0 = pd.read_csv("First 25000 Rows.csv")
print("rows read :", len(df_0))        # file advertises 25 000 but actually 5 000

# -------------------------------------------------------------------
# 1. Basic cleaning / type‑casting
# -------------------------------------------------------------------
df_0['ts_recv']  = pd.to_datetime(df_0['ts_recv']).dt.time
df_0['ts_event'] = pd.to_datetime(df_0['ts_event']).dt.time

levels = range(10)                                 # book depth 0 … 9

int_cols = (['rtype','publisher_id','instrument_id','depth','size',
             'flags','sequence']
            + [f'bid_ct_0{i}' for i in levels]
            + [f'ask_ct_0{i}' for i in levels])
float_cols = (['price','ts_in_delta']
              + [f'{side}_px_0{i}' for side in ('bid','ask') for i in levels]
              + [f'{side}_sz_0{i}' for side in ('bid','ask') for i in levels])
cat_cols   = ['action','side','symbol']

for c in int_cols   : df_0[c] = df_0[c].astype('Int64',  errors='ignore')
for c in float_cols : df_0[c] = pd.to_numeric(df_0[c], errors='coerce')
for c in cat_cols   : df_0[c] = df_0[c].astype('category', errors='ignore')

# -------------------------------------------------------------------
# 2. Per‑event order‑flow (bid/ask, each level)
# -------------------------------------------------------------------
ask_px_cols = [f"ask_px_0{i}" for i in levels]
ask_sz_cols = [f"ask_sz_0{i}" for i in levels]
bid_px_cols = [f"bid_px_0{i}" for i in levels]
bid_sz_cols = [f"bid_sz_0{i}" for i in levels]

ask_parts, bid_parts = [], []

for sym, grp in df_0.groupby('symbol', sort=False):
    grp = grp.sort_values('ts_event').reset_index(drop=True)

    # --- ask‑side OF ------------------------------------------------
    OF_ask = grp[['symbol','ts_recv','ts_event'] + ask_px_cols + ask_sz_cols].copy()
    for m in levels:
        px, sz = f'ask_px_0{m}', f'ask_sz_0{m}'
        px_prev, sz_prev = OF_ask[px].shift(1), OF_ask[sz].shift(1)
        OF_ask[f'OF_ask_0{m}'] = np.where(
            OF_ask[px] >  px_prev,             -OF_ask[sz],
            np.where(OF_ask[px] == px_prev,     sz_prev - OF_ask[sz], sz_prev)
        )
    ask_parts.append(OF_ask.drop(index=0).reset_index(drop=True))

    # --- bid‑side OF ------------------------------------------------
    OF_bid = grp[['symbol','ts_recv','ts_event'] + bid_px_cols + bid_sz_cols].copy()
    for m in levels:
        px, sz = f'bid_px_0{m}', f'bid_sz_0{m}'
        px_prev, sz_prev = OF_bid[px].shift(1), OF_bid[sz].shift(1)
        OF_bid[f'OF_bid_0{m}'] = np.where(
            OF_bid[px] >  px_prev,              OF_bid[sz],
            np.where(OF_bid[px] == px_prev,      OF_bid[sz] - sz_prev, -sz_prev)
        )
    bid_parts.append(OF_bid.drop(index=0).reset_index(drop=True))

OF_ask_all = pd.concat(ask_parts, ignore_index=True)
OF_bid_all = pd.concat(bid_parts, ignore_index=True)

# minute bucket
for df in (OF_ask_all, OF_bid_all):
    df['minute'] = df['ts_event'].astype(str).str[:5]

# -------------------------------------------------------------------
# 3. Multi‑level, depth‑scaled OFI per (symbol, minute)
# -------------------------------------------------------------------
events_parts = []
for sym in OF_bid_all['symbol'].unique():
    bid = OF_bid_all.query("symbol == @sym").sort_values('ts_event')
    ask = OF_ask_all.query("symbol == @sym").sort_values('ts_event')
    ev  = pd.concat([bid[['symbol','minute'] + [f'OF_bid_0{i}' for i in levels] +
                          [f'bid_sz_0{i}'   for i in levels]],
                     ask[[                f'OF_ask_0{i}' for i in levels] +
                          [f'ask_sz_0{i}'   for i in levels]]], axis=1)
    events_parts.append(ev)

events_all = pd.concat(events_parts, ignore_index=True)

ofi_unscaled = (
    events_all.groupby(['symbol','minute'], as_index=False)
              .apply(lambda df: pd.Series({f'OFI_lvl_{m}':
                                            (df[f'OF_bid_0{m}'] - df[f'OF_ask_0{m}']).sum()
                                            for m in levels}))
              .reset_index(drop=True)
)

def _compute_Q(df):
    return pd.Series({'Q_M_h': np.mean([(df[f'bid_sz_0{i}'] + df[f'ask_sz_0{i}'])/2
                                         for i in levels])})

Q_df = (events_all.groupby(['symbol','minute'], group_keys=False)
                 .apply(_compute_Q).reset_index())

deeper_ofi_df = ofi_unscaled.merge(Q_df, on=['symbol','minute'], how='left')
for m in levels:
    deeper_ofi_df[f'scaled_ofi_lvl_{m}'] = deeper_ofi_df[f'OFI_lvl_{m}'] / deeper_ofi_df['Q_M_h']

deep_ofi_scaled = deeper_ofi_df[['symbol','minute'] +
                                [f'scaled_ofi_lvl_{m}' for m in levels]]

# -------------------------------------------------------------------
# 4. “Integrated” OFI  (1st PC across levels)  – per‑symbol
# -------------------------------------------------------------------
integrated_parts = []
for sym, grp in deep_ofi_scaled.groupby('symbol', sort=False):
    X        = grp[[f'scaled_ofi_lvl_{m}' for m in levels]].values
    w1       = PCA(n_components=1).fit(X).components_[0]
    w1      /= np.sum(np.abs(w1))                      # L1 normalise
    grp['integrated_ofi'] = X.dot(w1)
    integrated_parts.append(grp[['symbol','minute','integrated_ofi']])

integrated_ofi_df = pd.concat(integrated_parts, ignore_index=True)

# -------------------------------------------------------------------
# 5. Cross‑section PCA  (market‑wide OFI factor)
# -------------------------------------------------------------------
cross_section_df = (integrated_ofi_df.pivot_table(index='minute',
                                                  columns='symbol',
                                                  values='integrated_ofi',
                                                  aggfunc='first')
                                    .fillna(0.0))                       # important!

pca_cross  = PCA(n_components=1).fit(cross_section_df.values)
w_l1       = pca_cross.components_[0] / np.sum(np.abs(pca_cross.components_[0]))
cross_section_df['cross_asset_ofi'] = cross_section_df.values.dot(w_l1)

# -------------------------------------------------------------------
# 6. Long format  +  idiosyncratic component
# -------------------------------------------------------------------
market_factor = cross_section_df['cross_asset_ofi'].rename_axis('minute').reset_index()
loadings      = pd.Series(w_l1, index=cross_section_df.columns[:-1], name='loading')

final_ofi_df = (
    integrated_ofi_df
      .merge(market_factor, on='minute', how='left')
      .merge(loadings, left_on='symbol', right_index=True, how='left')
)
final_ofi_df['idiosyncr_ofi'] = (final_ofi_df['integrated_ofi'] -
                                 final_ofi_df['loading'] * final_ofi_df['cross_asset_ofi'])

print("\n=== head of final dataset ===")
print(final_ofi_df.head())
