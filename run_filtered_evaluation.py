#!/usr/bin/env python3
"""Convenience script to run filtered medical coding evaluation and/or dataset augmentation.

Usage examples (after setting up any required RAG system wiring):
  python run_filtered_evaluation.py --dataset data/enhanced_mimic_iv_structured_coding.json \
      --primary-icd-version 9 --sample-size 5 --evaluate --save-json results_icd9_sample5.json

  python run_filtered_evaluation.py --dataset data/enhanced_mimic_iv_structured_coding.json \
      --primary-icd-version 10 --augment augmented_icd10_sample20.json --sample-size 20

If you don't have a RAG system instantiated yet, predictions will be empty and warnings logged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from medical_coding_evaluator import MedicalCodingEvaluator


def build_arg_parser():
    p = argparse.ArgumentParser(description="Filtered medical coding evaluation & augmentation")
    p.add_argument('--dataset', required=True, help='Path to structured dataset JSON')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--primary-icd-version', type=int, help='Filter single primary_icd_version (e.g. 9 or 10)')
    group.add_argument('--icd-version-allowlist', nargs='+', type=int, help='Allowlist of primary_icd_version values')
    p.add_argument('--sample-size', type=int, help='Sample size after filtering')
    p.add_argument('--evaluate', action='store_true', help='Run evaluation on filtered set')
    p.add_argument('--save-json', help='Where to save evaluation JSON (if --evaluate)')
    p.add_argument('--save-csv', help='Where to export evaluation CSV (if --evaluate)')
    p.add_argument('--augment', help='Path to write augmented dataset with predictions')
    p.add_argument('--no-metrics', action='store_true', help='Skip per-record metrics in augmentation')
    return p


def main():
    args = build_arg_parser().parse_args()

    evaluator = MedicalCodingEvaluator(args.dataset)
    # Note: plug in RAG system here if available. Example:
    # from agents.agno_rag_agent import YourRAGBuilder
    # rag_system = YourRAGBuilder(...)
    # evaluator.rag_system = rag_system

    if args.evaluate:
        results = evaluator.run_filtered_evaluation(
            primary_icd_version=args.primary_icd_version,
            icd_version_allowlist=args.icd_version_allowlist,
            sample_size=args.sample_size
        )
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved evaluation JSON -> {args.save_json}")
        if args.save_csv:
            evaluator.export_to_csv(args.save_csv)

    if args.augment:
        evaluator.augment_dataset_with_predictions(
            output_path=args.augment,
            primary_icd_version=args.primary_icd_version,
            icd_version_allowlist=args.icd_version_allowlist,
            sample_size=args.sample_size,
            include_metrics=not args.no_metrics
        )

    if not args.evaluate and not args.augment:
        print("Nothing to do: specify --evaluate and/or --augment")


if __name__ == '__main__':
    main()
