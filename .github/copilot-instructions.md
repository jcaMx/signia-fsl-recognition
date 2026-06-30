# Copilot Instructions

This repository is for Filipino Sign Language Recognition.

## Project context

The project supports two recognition pipelines:

1. Static signs
   - Single-frame landmark classification
   - Used for alphabet, numbers, or signs that do not require motion

2. Dynamic signs
   - Video-based sign/gesture recognition
   - Pipeline:
     video -> MediaPipe landmarks per frame -> normalized fixed-length sequence -> sequence model

## Labels

The master labels file uses:

```csv
id,label,display_label,category,modality,enabled