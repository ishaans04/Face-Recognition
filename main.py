"""Pipeline controller.

Orchestrates: input -> preprocess -> embedding -> aggregation -> matching -> evaluation.
"""
import argparse
import csv
import sys

import config
from src import aggregator, evaluator, matcher
from src import embedding as embedding_module
from src import input as input_module
from src import preprocess


def build_reference_embedding(reference_dir):
    images, paths = input_module.load_images(reference_dir)

    embeddings = []
    for image, path in zip(images, paths):
        preprocess.ensure_min_resolution(image)
        try:
            embeddings.append(embedding_module.get_embedding(image))
        except ValueError as exc:
            h, w = image.shape[:2]
            print(f"Skipping {path} ({w}x{h}): {exc}", file=sys.stderr)

    if not embeddings:
        raise ValueError(
            f"No usable faces found in reference set: {reference_dir} "
            "(see 'Skipping ...' lines above for the reason per image)"
        )

    return aggregator.aggregate_embeddings(embeddings)


def run_verify(reference_dir, test_image_path, threshold=None):
    reference_embedding = build_reference_embedding(reference_dir)

    test_image = input_module.load_image(test_image_path)
    preprocess.ensure_min_resolution(test_image)
    test_embedding = embedding_module.get_embedding(test_image)

    similarity, match = matcher.verify(reference_embedding, test_embedding, threshold)
    print(f"Similarity: {similarity:.4f}")
    print(f"Match: {match}")
    return similarity, match


def run_evaluate(pairs_csv, threshold=None):
    """pairs_csv columns: reference_dir, test_image, label (1 = same identity, 0 = different)."""
    y_true = []
    y_pred = []

    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _, match = run_verify(row["reference_dir"], row["test_image"], threshold)
            y_true.append(int(row["label"]))
            y_pred.append(int(match))

    metrics = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Face verification system (ArcFace-based)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify a test image against a reference set")
    verify_parser.add_argument("--reference-dir", default=config.REFERENCE_DIR)
    verify_parser.add_argument("--test-image", required=True)
    verify_parser.add_argument("--threshold", type=float, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Batch-evaluate verification pairs from a CSV")
    evaluate_parser.add_argument("--pairs-csv", required=True)
    evaluate_parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    if args.command == "verify":
        run_verify(args.reference_dir, args.test_image, args.threshold)
    elif args.command == "evaluate":
        run_evaluate(args.pairs_csv, args.threshold)


if __name__ == "__main__":
    main()
