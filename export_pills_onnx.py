"""
Step 8: Export the fine-tuned DEIMv2-Atto pill checkpoint to ONNX
and apply int8 dynamic quantization.

Usage (run from /workspaces/deimv2):
    python export_pills_onnx.py --checkpoint outputs/deimv2_atto_pills/best.pth

Output:
    deimv2_atto_pills.onnx       (~2 MB fp32)
    deimv2_atto_pills_int8.onnx  (~0.5 MB int8)
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/deimv2_atto_pills/best.pth",
        help="Path to the fine-tuned .pth checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/deimv2/deimv2_hgnetv2_atto_pills.yml",
    )
    parser.add_argument(
        "--out", default="deimv2_atto_pills.onnx",
        help="Output ONNX filename",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        print("Run training first (see train_pills_colab.ipynb)")
        sys.exit(1)

    # ── Step 8a: Export fp32 ONNX ────────────────────────────────────────────
    print("Exporting to ONNX (fp32)...")
    result = subprocess.run(
        [
            sys.executable,
            "tools/deployment/export_onnx.py",
            "--check",
            "-c", args.config,
            "-r", args.checkpoint,
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ONNX export failed.")
        sys.exit(1)

    # Find the exported file (export_onnx.py places it in the working dir)
    import glob
    candidates = glob.glob("*.onnx") + glob.glob("outputs/**/*.onnx", recursive=True)
    if not candidates:
        print("ERROR: no .onnx file found after export.")
        sys.exit(1)

    fp32_path = candidates[0]
    fp32_kb = os.path.getsize(fp32_path) // 1024
    print(f"fp32 model: {fp32_path}  ({fp32_kb} KB)")

    # Rename to our standard name
    if fp32_path != args.out:
        os.rename(fp32_path, args.out)
        fp32_path = args.out
        print(f"Renamed to: {fp32_path}")

    # ── Step 8b: int8 dynamic quantization ────────────────────────────────────
    int8_path = fp32_path.replace(".onnx", "_int8.onnx")
    print(f"\nQuantizing to int8: {int8_path} ...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    except Exception as e:
        print(f"WARNING: quantization failed ({e}). Keeping fp32 only.")
        int8_path = None

    if int8_path and os.path.isfile(int8_path):
        int8_kb = os.path.getsize(int8_path) // 1024
        print(f"int8 model: {int8_path}  ({int8_kb} KB)")
        print(f"Size reduction: {fp32_kb / int8_kb:.1f}x")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Done ────────────────────────────────────────────────────────────")
    print(f"fp32 : {fp32_path}")
    if int8_path:
        print(f"int8 : {int8_path}  ← upload this to your CDN for the app")
    print()
    print("Next: host the int8 file and paste the URL into")
    print("  /workspaces/Aipaa/src/lib/detection/scannerModelManager.ts")
    print('  under the "pill_deim" model entry.')


if __name__ == "__main__":
    main()
