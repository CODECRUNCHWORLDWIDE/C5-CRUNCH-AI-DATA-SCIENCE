# Challenge 1 — Merge three tables

**Time estimate:** 2 hours.

## Problem statement

You have three small CSVs from three departments of the same fictional company. They almost agree on keys. Your job is to produce **one** clean, validated table with one row per order, every order's customer attached, and every order's product attached — without losing data and without silently inflating the row count.

You will write a single script `merge_three_tables.py` that:

1. Reads the three CSVs.
2. Cleans the keys (whitespace, casing).
3. Joins them in two `merge` calls.
4. Validates the result.
5. Writes the merged table to `merged.csv`.

This is the realistic version of every data-engineering task. Get it right and the rest of pandas feels easy.

## The data

Create three CSV files (also reproduced inline below — you can paste them into files in your working directory, or read them from a `StringIO` in the script). All three are deliberately small so you can eyeball the issues.

**`orders.csv`** — one row per order:

```csv
order_id,customer_id,product_sku,quantity,order_date
1001,C001,SKU-A,2,2025-08-01
1002,C002,SKU-B,1,2025-08-01
1003,C001,SKU-C,3,2025-08-02
1004,C003,SKU-A,1,2025-08-02
1005,C002,SKU-D,4,2025-08-03
1006,C004,SKU-B,2,2025-08-04
1007,C999,SKU-A,1,2025-08-05
1008,C001,SKU-E,1,2025-08-05
```

**`customers.csv`** — one row per customer (note the leading whitespace and the casing mismatch on `C002`):

```csv
customer_id,name,country,segment
C001,Ada Lovelace,United Kingdom,enterprise
 c002 ,Bo Liu,United States,SMB
C003,Cy Reeves,Canada,enterprise
C004,Dee Singh,India,SMB
C005,Edu Garcia,Spain,enterprise
```

**`products.csv`** — one row per SKU (note the duplicate row for `SKU-A`):

```csv
sku,name,category,unit_price
SKU-A,Widget Pro,widget,49.99
SKU-A,Widget Pro,widget,49.99
SKU-B,Gadget Lite,gadget,19.99
SKU-C,Thingamajig,thingamajig,8.50
SKU-D,Doohickey,doohickey,12.00
SKU-E,Whatsit,whatsit,35.00
```

## What is wrong with this data (read before coding)

- `customers.csv` row 2 has `customer_id` of `" c002 "` — leading space, trailing space, lowercase. The `orders.csv` references `"C002"`. They will not match without cleaning.
- `customers.csv` has no row for `"C999"`. Order `1007` is orphaned.
- `products.csv` has a duplicate row for `SKU-A`. If you `merge` straight, every order for `SKU-A` will appear twice. This is a "many-to-many" bug.
- `orders.csv` is the source of truth for "which orders happened." Every row of `orders.csv` must appear exactly once in the output.

## Acceptance criteria

- [ ] A single script `merge_three_tables.py` runs end-to-end with `python merge_three_tables.py` and writes `merged.csv` to the current directory.
- [ ] The output has exactly **8 rows** (one per order in `orders.csv`).
- [ ] The output has columns `order_id, customer_id, product_sku, quantity, order_date, customer_name, customer_country, customer_segment, product_name, product_category, unit_price, line_total`.
- [ ] `line_total = quantity * unit_price`.
- [ ] The orphaned order (`1007`, `customer_id = C999`) is **kept** in the output, with the customer columns set to NaN/empty. The script prints a warning about this row.
- [ ] No `_x` / `_y` suffix columns leak into the output. Choose `suffixes=` and `.drop(columns=...)` deliberately.
- [ ] You pass `validate=` on **both** merges. The product merge needs the duplicate cleaned first or it will raise — that is the point.
- [ ] `python -m py_compile merge_three_tables.py` succeeds.

## Suggested layout

```python
"""Join three CSVs into one clean orders table."""
from __future__ import annotations

import pandas as pd


def load_orders(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["order_date"])


def load_customers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize the key: strip whitespace, uppercase.
    df["customer_id"] = df["customer_id"].str.strip().str.upper()
    # Verify uniqueness AFTER cleaning, not before.
    assert df["customer_id"].is_unique, "customer_id should be unique"
    return df


def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sku"] = df["sku"].str.strip()
    before = len(df)
    df = df.drop_duplicates(subset="sku", keep="first")
    after = len(df)
    if after < before:
        print(f"products: dropped {before - after} duplicate SKU rows")
    assert df["sku"].is_unique, "sku should be unique after dedup"
    return df


def join_all(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    # Customer merge (left, many-to-one)
    merged = orders.merge(
        customers.rename(columns={
            "name":    "customer_name",
            "country": "customer_country",
            "segment": "customer_segment",
        }),
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator="_customer_merge",
    )
    orphans = merged[merged["_customer_merge"] == "left_only"]
    if len(orphans):
        print(
            f"WARNING: {len(orphans)} order(s) have no matching customer: "
            f"{orphans['order_id'].tolist()}"
        )
    merged = merged.drop(columns="_customer_merge")

    # Product merge (left, many-to-one after dedup)
    merged = merged.merge(
        products.rename(columns={
            "sku":      "product_sku",     # rename to match orders' key
            "name":     "product_name",
            "category": "product_category",
        }),
        on="product_sku",
        how="left",
        validate="many_to_one",
    )

    merged["line_total"] = (merged["quantity"] * merged["unit_price"]).round(2)
    return merged


def main() -> None:
    orders    = load_orders("orders.csv")
    customers = load_customers("customers.csv")
    products  = load_products("products.csv")
    merged    = join_all(orders, customers, products)
    assert len(merged) == len(orders), (
        f"row count exploded: {len(merged)} vs {len(orders)} orders. "
        "Check for many-to-many bug."
    )
    merged.to_csv("merged.csv", index=False)
    print(f"wrote merged.csv: {len(merged)} rows")


if __name__ == "__main__":
    main()
```

## Hints

<details>
<summary>If your merge inflates the row count from 8 to 9 or more</summary>

You forgot to dedupe `products.csv`. The duplicate row for `SKU-A` means every order for `SKU-A` joins twice. `validate="many_to_one"` will catch this — it raises a `MergeError` if the right side has duplicate keys. Drop the duplicates first; the assertion in `load_products` is what should fail (loudly) if you skip the dedup.

</details>

<details>
<summary>If your customer_id matches do not work</summary>

Look at the raw value of `customers.csv` row 2: `" c002 "`. Leading space, trailing space, lowercase. The orders file has `"C002"`. Apply `.str.strip().str.upper()` to *both* sides if you want to be paranoid, or just to the side that is dirty (the customer table). Always normalize keys before merging.

</details>

<details>
<summary>If the orphan order (1007) silently disappears</summary>

You used `how="inner"` instead of `how="left"`. The whole point of `left` is to keep every row of the left table even when there is no match on the right. The orphan should appear with NaN customer columns and your script should print the warning.

</details>

## Stretch goals

- Use `indicator=True` on both merges and write a small report (`merge_report.txt`) of how many rows came from each side.
- Compute `customer_total = sum(line_total) per customer` as a final groupby+merge and add it as a column to every order row. `transform` is the cleanest spelling.
- Convert the merged table to a Parquet file (`merged.parquet`) with `pyarrow` and time the difference between writing CSV and Parquet. Parquet is roughly 5–10× smaller and 10× faster to read; this is why every data-engineering team uses it for intermediate storage.
- Try the same join in DuckDB (`duckdb.sql("SELECT ... FROM orders LEFT JOIN customers ... LEFT JOIN products ...")`) and confirm the row counts match. This is the comparison Lecture 3 promises.

## Why this matters

Every real ML pipeline is a chain of joins. The reason analyses go wrong silently is almost always a bad join: a many-to-many that doubled rows, a left-vs-inner that dropped them, a stale lookup table that left half the rows orphaned. The discipline of (a) cleaning keys, (b) passing `validate=`, (c) using `indicator=` to count the slices, (d) asserting on the row count after — that is the difference between a script that ships and a script that is haunted.

## Submission

Commit `merge_three_tables.py` and the three input CSVs and the output `merged.csv` to your Week 2 repo. Paste the script's printed output (the warning about the orphan, the row count) into the commit message.
