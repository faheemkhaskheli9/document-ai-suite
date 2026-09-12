# Evaluation Notes: Document AI Suite

## Metrics

- **Layout + OCR**: layout-detection mAP/IoU (YOLO), OCR character/word error
  rate on detected regions, field-extraction precision/recall against
  ground-truth JSON
- **Classify + Review**: classification accuracy (document type/scanned vs
  digital), confidence-score calibration, % routed to human review vs.
  auto-accepted, reviewer-agreement rate on routed items

## Reproducing Results

```bash
python -m src.evaluate --config configs/eval.yaml
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
|      |        |        |       |       |
