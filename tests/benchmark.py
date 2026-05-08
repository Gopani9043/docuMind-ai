import sys
import json
import os
from pathlib import Path

# Add backend to path so we can import from it
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.ocr import extract_text_from_file
from services.extractor import extract_structured_data

# ── Paths ─────────────────────────────────────────
SAMPLES_DIR     = Path(__file__).parent.parent / "sample_documents"
GROUND_TRUTH_DIR = SAMPLES_DIR / "ground_truth"

# ── Helpers ───────────────────────────────────────

def load_ground_truth(filename: str) -> dict:
    """Load the manually verified correct answer for a document."""
    stem = Path(filename).stem
    gt_path = GROUND_TRUTH_DIR / f"{stem}.json"
    if not gt_path.exists():
        return None
    with open(gt_path) as f:
        return json.load(f)


def compare_field(predicted, expected) -> bool:
    """
    Compare a predicted value against expected.
    Flexible matching — handles strings, numbers, None.
    """
    if expected is None:
        return True
    if predicted is None:
        return False
    # Number comparison — allow 1% tolerance
    if isinstance(expected, (int, float)):
        try:
            return abs(float(predicted) - float(expected)) / float(expected) < 0.01
        except (ValueError, ZeroDivisionError):
            return False
    # String comparison — case insensitive, strip whitespace
    return str(predicted).strip().lower() == str(expected).strip().lower()


def score_extraction(predicted: dict, ground_truth: dict) -> dict:
    """
    Compare predicted extraction against ground truth.
    Returns per-field scores and overall accuracy.
    """
    expected_data  = ground_truth.get("extracted_data", {})
    predicted_data = predicted.get("extracted_data", {})

    field_scores = {}
    correct = 0
    total   = 0

    for field, expected_val in expected_data.items():
        predicted_val = predicted_data.get(field)
        is_correct    = compare_field(predicted_val, expected_val)
        field_scores[field] = {
            "expected":  expected_val,
            "predicted": predicted_val,
            "correct":   is_correct
        }
        total   += 1
        correct += 1 if is_correct else 0

    # Check document type
    type_correct = predicted.get("document_type") == ground_truth.get("document_type")
    field_scores["document_type"] = {
        "expected":  ground_truth.get("document_type"),
        "predicted": predicted.get("document_type"),
        "correct":   type_correct
    }
    total   += 1
    correct += 1 if type_correct else 0

    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    return {
        "accuracy":     accuracy,
        "correct":      correct,
        "total":        total,
        "field_scores": field_scores
    }


def run_benchmark():
    """
    Run the full pipeline on every document that has a ground truth file.
    Print results and save to benchmarks.json.
    """
    print("\n" + "="*60)
    print("  DocParse Benchmark")
    print("="*60 + "\n")

    results   = []
    gt_files  = list(GROUND_TRUTH_DIR.glob("*.json"))

    if not gt_files:
        print("No ground truth files found.")
        return

    for gt_file in gt_files:
        filename = gt_file.stem
        # Find matching PDF
        pdf_path = SAMPLES_DIR / f"{filename}.pdf"
        if not pdf_path.exists():
            print(f"  ⚠ Skipping {filename} — PDF not found")
            continue

        print(f"  Processing: {filename}.pdf")

        try:
            # Run pipeline
            with open(pdf_path, "rb") as f:
                file_bytes = f.read()

            raw_text  = extract_text_from_file(file_bytes, f"{filename}.pdf")
            predicted = extract_structured_data(raw_text)
            gt        = load_ground_truth(f"{filename}.pdf")
            score     = score_extraction(predicted, gt)

            results.append({
                "filename":    f"{filename}.pdf",
                "accuracy":    score["accuracy"],
                "correct":     score["correct"],
                "total":       score["total"],
                "field_scores": score["field_scores"],
                "document_type": predicted.get("document_type", "unknown")
            })

            # Print result
            status = "✓" if score["accuracy"] >= 80 else "✗"
            print(f"  {status} {filename}: {score['accuracy']}% ({score['correct']}/{score['total']} fields)\n")

            # Print field details
            for field, fs in score["field_scores"].items():
                icon = "  ✓" if fs["correct"] else "  ✗"
                print(f"    {icon} {field}")
                if not fs["correct"]:
                    print(f"        expected : {fs['expected']}")
                    print(f"        predicted: {fs['predicted']}")
            print()

        except Exception as e:
            print(f"  ✗ {filename} FAILED: {e}\n")
            results.append({
                "filename": f"{filename}.pdf",
                "accuracy": 0,
                "error":    str(e)
            })

    # ── Summary ───────────────────────────────────
    if results:
        accuracies   = [r["accuracy"] for r in results if "error" not in r]
        overall      = round(sum(accuracies) / len(accuracies), 1) if accuracies else 0

        print("="*60)
        print(f"  Overall accuracy : {overall}%")
        print(f"  Documents tested : {len(results)}")
        print(f"  Passed (>=80%)   : {sum(1 for a in accuracies if a >= 80)}")
        print("="*60 + "\n")

        # Save results
        output = {
            "overall_accuracy": overall,
            "total_documents":  len(results),
            "results":          results
        }
        out_path = Path(__file__).parent.parent / "backend" / "benchmarks.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to backend/benchmarks.json\n")


if __name__ == "__main__":
    run_benchmark()