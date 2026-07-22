"""Pipeline controller.

Orchestrates: input -> preprocess -> embedding -> aggregation -> matching -> evaluation.
"""
import argparse
import csv
import os
import sys

import config
from src import aggregator, evaluator, gallery, matcher
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


def build_gallery(dataset_dir):
    """dataset_dir/<identity_name>/*.jpg for each of N people -> {identity_name: embedding}."""
    identities = {}
    for name in sorted(os.listdir(dataset_dir)):
        identity_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(identity_dir):
            continue
        try:
            identities[name] = build_reference_embedding(identity_dir)
        except ValueError as exc:
            print(f"Skipping identity '{name}': {exc}", file=sys.stderr)

    if not identities:
        raise ValueError(f"No usable identities found in dataset: {dataset_dir}")

    return identities


def run_build_gallery(dataset_dir, output_path):
    identities = build_gallery(dataset_dir)
    gallery.save_gallery(list(identities.keys()), list(identities.values()), output_path)
    print(f"Embedded {len(identities)} identities into gallery: {output_path}")
    return identities


def run_identify(gallery_path, test_image_path, threshold=None, top_k=None):
    gallery_labels, gallery_embeddings = gallery.load_gallery(gallery_path)

    test_image = input_module.load_image(test_image_path)
    preprocess.ensure_min_resolution(test_image)
    test_embedding = embedding_module.get_embedding(test_image)

    identity, best_score, ranked = matcher.identify(
        gallery_embeddings, gallery_labels, test_embedding, threshold, top_k
    )

    print(f"Best match: {identity} ({best_score * 100:.2f}% confidence)")
    print(f"Top {len(ranked)} matches:")
    for rank, (label, score) in enumerate(ranked, start=1):
        print(f"  {rank}. {label:<30} {score:.4f}")
    return identity, best_score, ranked


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

    build_gallery_parser = subparsers.add_parser(
        "build-gallery", help="Embed a dataset of N identities into a cached gallery"
    )
    build_gallery_parser.add_argument("--dataset-dir", default=config.DATASET_DIR)
    build_gallery_parser.add_argument("--output", default=config.GALLERY_PATH)

    identify_parser = subparsers.add_parser(
        "identify", help="Identify a test image against a prebuilt gallery (1:N search)"
    )
    identify_parser.add_argument("--gallery", default=config.GALLERY_PATH)
    identify_parser.add_argument("--test-image", required=True)
    identify_parser.add_argument("--threshold", type=float, default=None)
    identify_parser.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()

    if args.command == "verify":
        run_verify(args.reference_dir, args.test_image, args.threshold)
    elif args.command == "evaluate":
        run_evaluate(args.pairs_csv, args.threshold)
    elif args.command == "build-gallery":
        run_build_gallery(args.dataset_dir, args.output)
    elif args.command == "identify":
        run_identify(args.gallery, args.test_image, args.threshold, args.top_k)


if __name__ == "__main__":
    main()
