# Input datasets

CSV inputs use Sequence and label. Original rows and labels are preserved.

- training/tada_round1_train_historical.csv: 20 records (10 positive, 10 contrasts).
- training/tada_round2_train.csv: 111-record historical snapshot (44 label 1,
  67 label 0); not an exact reconstruction of the manuscript's 20+60 update.
- training/tada_positive_10.csv and ccda_positive_22.csv: positive seed records
  transcribed from the supplied final SI Tables S1 and S5. These are positive-only
  inputs, not binary training sets. No new contrasts were generated for release.
- tadA_lora_set.csv: legacy public 40-record example, not the 20-record input.
- tadA_seed.csv: historical 100-record starting-pool example, not activity labels.
- reference_seeds/: archived reference records; not substituted for training data.

Label-0 synthetic contrasts do not establish experimental inactivity.
Source/size/checksum details are in PACKAGE_MANIFEST.json and training/manifest.json.
An exact CcdA binary training snapshot and a verified cumulative Round II snapshot
remain unrecovered; the positive seed records must not be mislabeled as either.
