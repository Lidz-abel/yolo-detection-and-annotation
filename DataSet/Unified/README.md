# Unified Detection Dataset

This directory stores the normalized detection manifests used to compare loading strategies across VOC2012 and COCO2017.

## Schema

Each line in `manifests/*.jsonl` is one image sample with a unified schema:

- `sample_id`
- `dataset_source`
- `split`
- `image_id`
- `image_path`
- `image_rel_path`
- `width`
- `height`
- `boxes` in `xyxy`
- `labels` using dataset-local contiguous ids
- `label_names`
- `original_category_ids`
- `iscrowd`
- `difficult`
- `truncated`
- `occluded`
- `areas`
- `annotation_count`

## Outputs

- `manifests/voc2012_train.jsonl`
- `manifests/voc2012_val.jsonl`
- `manifests/coco2017_train.jsonl`
- `manifests/coco2017_val.jsonl`
- `manifests/all_train.jsonl`
- `manifests/all_val.jsonl`
- `manifests/all.jsonl`
- `metadata/schema.json`
- `metadata/class_maps.json`
- `metadata/summary.json`

## Build

Run:

```bash
python3 /home/lidz/YOLO/DataSet/Unified/build_unified_detection_manifest.py
```
