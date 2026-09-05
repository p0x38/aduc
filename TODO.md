# TODO.md

## Completed

* [x] JSONL loading
* [x] Dataset validation
* [x] Dataset statistics
* [x] Dataset splitting
* [x] Dataset export
* [x] Dataset normalization
* [x] Dataset conversion
* [x] Exact record deduplication
* [x] Duplicate prompt detection
* [x] CLI commands
* [x] CLI tests
* [x] Deduplication CLI output
* [x] Python 3.12/3.13 compatibility

## Work in progress

* [ ] Review dataset format

  * [ ] Define chord progression representation
  * [ ] Define subdivision/rhythm representation
  * [ ] Define optional metadata
  * [ ] Define canonical JSONL schema

## Planned

* [ ] Dataset quality checks

  * [ ] Invalid chord detection
  * [ ] Empty/malformed progression detection
  * [ ] Duplicate/near-duplicate analysis

* [ ] Dataset preprocessing

  * [ ] Canonicalize chord notation
  * [ ] Normalize progression structure
  * [ ] Generate training-ready JSONL

* [ ] Dataset inspection

  * [ ] Rich CLI statistics
  * [ ] Distribution analysis
  * [ ] Dataset summary report

* [ ] Training preparation

  * [ ] Train/validation/test split
  * [ ] Deterministic preprocessing
  * [ ] Training sample generation

* [ ] Model

  * [ ] Define model input/output format
  * [ ] Choose tokenizer/representation
  * [ ] Build baseline model
  * [ ] Training pipeline
  * [ ] Evaluation
