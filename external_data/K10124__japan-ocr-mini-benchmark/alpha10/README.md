# Japan OCR Mini Benchmark - Alpha10

This Alpha10 approved payload is published as a small synthetic Japanese receipt OCR/VLM evaluation sample.

## What Is Included

- 10 approved synthetic Japanese receipt samples.
- Clean receipt images for OCR/VLM evaluation.
- Matching public source JSON files.
- Matching public metadata JSON files.
- Alpha10 manifest and public safety scan.

## Synthetic Data Notice

All records in this payload are synthetic. They are not copies of real store receipts, real transactions, real people, or real brands.

No real reference images are included. Reference material was used only as private design guidance during dataset construction.

## Public Safety

Public safety scan: pass.

CASE-000048 is excluded from this Alpha10 payload.

The payload includes invoice_profile and phone_profile fields so reviewers can understand how registration-number-like and phone-number-like text is handled. These fields are synthetic or unverified OCR benchmark fields and were not externally looked up.

## License

The Alpha10 approved payload is released under CC BY 4.0.

## Files

- `images/`: synthetic receipt images.
- `source_json/`: public source records for each receipt.
- `metadata/`: public metadata for each receipt.
- `alpha10_manifest.json`: machine-readable manifest.
- `alpha10_manifest.csv`: table summary of the included records.
- `alpha10_public_safety_scan.md`: public safety scan summary.
- `LICENSE_NOTICE.md`: license and use notice.
- `ALPHA10_LICENSE_FINAL_CONFIRMATION.md`: final license confirmation for this payload.

## Limitations

This is a small alpha payload, not a large training dataset. It is intended for OCR/VLM evaluation and workflow checks.

The samples are designed to be useful benchmark examples, but they do not represent every Japanese receipt format.
