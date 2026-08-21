# Retained evidence

This directory intentionally contains only:

- the current fresh information-acquisition demo and its source frames;
- the matching initial and post-action uncertainty/interaction maps;
- the frozen six-frame RGB critic and its clean/sealed reports;
- the current learned object sidecar and its first clean NOT-GO report;
- machine-readable method/product gate summaries.

The retained checkpoints are runnable evidence. New scenarios use the generic
collection, training, evaluation, and rendering entry points under `scripts/`;
scenario-specific constants belong in `configs/scenarios/`, not Python files.

Earlier variants, pose sweeps, failed scenario trials, duplicate videos, and
intermediate checkpoints were removed from the working tree. Tracked history
remains recoverable from Git.
