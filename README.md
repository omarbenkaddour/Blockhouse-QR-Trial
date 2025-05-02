# Multi-Level and Cross-Asset Order Flow Imbalance (OFI) Analysis

This script processes high-frequency limit order book data to construct **multi-level, depth-scaled order flow imbalance (OFI) measures** at both the **individual asset** and **cross-asset (market-wide)** levels.

> **Note:** Although the provided dataset contains only AAPL data, the analysis is designed and implemented *as if* a full cross-asset dataset were available. The code is fully general and scalable to multi-asset environments.

## Steps:

1. **Data Loading & Cleaning:**

   * Reads the raw LOB data, parses timestamps, and converts columns to appropriate types (int, float, categorical).

2. **Per-Event OFI Construction:**

   * For each event and each price level (depth 0–9), computes **order flow changes** on both the bid and ask sides, following standard Cont et al. (2014)-style logic.

3. **Aggregation:**

   * Aggregates per-event OFIs into **minute-level buckets** for each symbol:

     * *Best-level OFI:* simple difference between aggregated bid and ask flows at depth 0.
     * *Multi-level scaled OFI:* combines OFIs across all depths (0–9), normalized by average depth.

4. **Integrated OFI (Per Symbol):**

   * For each symbol, applies **PCA across levels 0–9** to extract the first principal component, which serves as the **integrated OFI**—capturing the dominant flow direction across levels.

5. **Cross-Asset Factor Construction:**

   * Pivots the integrated OFIs into a cross-sectional matrix (minutes × symbols).
   * Runs **PCA across symbols** to derive a **market-wide OFI factor** (the first principal component across symbols).
   * Records **per-symbol loadings** (the weight of each asset in the market factor).

6. **Final Dataset:**

   * Merges everything into a long-format DataFrame that contains:

     * `OFI_best_level`
     * `integrated_ofi`
     * `cross_asset_ofi` (market factor)
     * `loading` (per-symbol market factor loading)
     * `idiosyncr_ofi` (residual OFI after removing the market component)

