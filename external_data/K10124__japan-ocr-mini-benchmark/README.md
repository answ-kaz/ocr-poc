---
license: cc-by-4.0
task_categories:
- image-to-text
- visual-question-answering
language:
- ja
tags:
- ocr
- japanese
- receipt
- synthetic-data
- document-ai
- vision-language-model
pretty_name: Japan OCR Mini Benchmark
size_categories:
- n<1K
---

# Japan OCR Mini Benchmark

## Dataset Summary

Japan OCR Mini Benchmark is a compact benchmark for evaluating whether OCR/VLM models can read Japanese receipts and return structured JSON. It covers receipt-level fields, item rows, tax summaries, totals, payment methods, and clean/noisy image variants.

The current repository entry point includes the frozen `v0.2.0` dataset payload, the `v0.4.0` Clean/Noisy LM Studio leaderboard, the `v0.4.2` receipt-generation design-candidate library, and the new `alpha10` public alpha payload.

## Latest Status

- Latest public alpha payload: `alpha10`
  - 10 approved synthetic Japanese receipt samples
  - clean images + public source JSON + public metadata
  - invoice_profile / phone_profile included
  - public safety scan: pass
  - license: CC BY 4.0
  - CASE-000048 excluded
- Frozen core dataset payload: `v0.2.0`
- Current model benchmark: `v0.4.0` Clean/Noisy LM Studio leaderboard
- Current receipt-generation library: `v0.4.2`
- Accepted receipt design candidates: `83`
- Distinct semantic structures: audit in progress

## Alpha10 Approved Payload

Alpha10 is the first small public-approved payload from the receipt library workflow. It contains:

- 10 clean synthetic Japanese receipt images
- 10 public source JSON files
- 10 public metadata JSON files
- invoice_profile and phone_profile fields
- public safety scan pass
- CC BY 4.0 license confirmation for the Alpha10 approved payload
- human approval record in the local release workflow
- CASE-000048 excluded

The CC BY 4.0 confirmation applies to the Alpha10 approved payload. Other project materials and future releases should be checked separately before reuse.

## Dataset Structure

```text
alpha10/
  README.md
  LICENSE_NOTICE.md
  ALPHA10_LICENSE_FINAL_CONFIRMATION.md
  alpha10_manifest.json
  alpha10_manifest.csv
  alpha10_public_safety_scan.md
  images/
  source_json/
  metadata/
```

## Data Fields

Alpha10 source JSON records include receipt identity, item rows, totals, tax summary, payment fields, and public safety metadata. Alpha10 metadata records include image path, source JSON path, SHA-256 hash, generation identifiers, invoice_profile, phone_profile, and public safety status.

## Intended Uses

- OCR/VLM evaluation on Japanese receipt images
- JSON extraction testing
- receipt-specific field, item, tax, and payment evaluation
- small local benchmark checks before larger experiments

## Out-of-Scope Uses

- large-scale model training as a primary dataset
- identity, store, phone, or registration-number verification
- financial, tax, legal, or accounting advice
- claims about real merchants or real transactions

## Synthetic Data Notice

The public data is synthetic. It does not include real receipt photos, real customer data, real transactions, copied real logos, or real brand assets.

## Public Safety and Privacy

- phone numbers and invoice registration numbers are synthetic or unverified OCR benchmark fields
- no live lookup was performed for phone numbers or registration numbers
- invoice_profile and phone_profile are included for structured evaluation
- real reference images are not included
- CASE-000048 is excluded

## Limitations

Alpha10 is a small alpha payload with 10 samples across cinema, coin_laundry, discount_store, onsen, restaurant, shared_workspace, supermarket_food, sweets_shop, taxi_transport, washoku. It is useful for inspection and early evaluation, but it is not a final taxonomy of Japanese receipt structures. The `v0.4.2` library contains 83 accepted design candidates, but the audited distinct semantic-structure count is still in progress.

## Benchmark Releases

| Area | Version | What it contains |
| --- | --- | --- |
| Public alpha payload | `alpha10` | 10 human-approved synthetic Japanese receipt samples with clean images, public source JSON, metadata, invoice/phone profiles, and public safety scan pass |
| Dataset payload | `v0.2.0` | 20 synthetic Japanese receipts with clean/noisy images and ground-truth JSON |
| LM Studio baseline | `v0.3.0` | first local multi-model noisy-image benchmark |
| Operational snapshots | `v0.3.1`, `v0.3.2` | additional LM Studio runs and combined rankings |
| Clean/Noisy leaderboard | `v0.4.0` | paired clean and noisy benchmark tables |
| Generator QA | `v0.4.1` | audited 23-template clean receipt generator snapshot |
| Design-candidate library | `v0.4.2` | 83 accepted receipt designs for future taxonomy and 100-type generation |

## Repository Map

```text
alpha10/
  README.md
  LICENSE_NOTICE.md
  ALPHA10_LICENSE_FINAL_CONFIRMATION.md
  alpha10_manifest.json
  alpha10_manifest.csv
  alpha10_public_safety_scan.md
  images/
  source_json/
  metadata/

v0.4.2 reference-generation report folder/
  README.md
  index.html
  contact_sheet.png
  manifest.jsonl
  summary.json

reports/v0.4.0/
  clean_leaderboard.*
  noisy_leaderboard.*
  clean_noisy_paired_leaderboard.*
```

## License

The Alpha10 approved payload is released under CC BY 4.0. This license confirmation applies to the Alpha10 approved payload. See `alpha10/ALPHA10_LICENSE_FINAL_CONFIRMATION.md` and `alpha10/LICENSE_NOTICE.md` after the payload is mirrored.
