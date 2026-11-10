#!/usr/bin/env python3
"""
Bulk Background Remover
-----------------------
Removes the background from every image in a folder and saves the result in
the format you choose (default: lossless PNG with transparency).

Install (CPU only, no GPU needed; use Python 3.11 / 3.12 / 3.13):
    pip install "rembg[cpu]" pillow tqdm

Usage:
    python bulk_bg_remover.py input_folder output_folder
    python bulk_bg_remover.py input_folder output_folder -f same
    python bulk_bg_remover.py input_folder output_folder -f jpg
    python bulk_bg_remover.py input_folder output_folder -f jpg --bg-color "#ffffff"
    python bulk_bg_remover.py input_folder output_folder -f webp -r
    python bulk_bg_remover.py input_folder output_folder -f jpg --compress=80
    python bulk_bg_remover.py input_folder output_folder -f webp --compress=60

Optional lossy compression (--compress=N, N = 1-100, like a quality %):
    Not set / 100  -> no quality loss (lossless where the format allows)
    80 / 60 / 40 / 20 -> smaller files, lower quality as N goes down
    jpg  : JPEG quality N
    webp : lossy WebP quality N (keeps transparency)
    png  : colours reduced to about N% of 256 (keeps transparency)
    tiff / bmp : not affected (always lossless)

Output formats (-f / --format):
    png   (default) lossless, keeps transparency
    same  keep each image's original extension
    jpg   NO transparency, background is filled with --bg-color (default white), lossy
    webp  lossless, keeps transparency
    tiff  lossless (LZW), keeps transparency
    bmp   NO transparency, background is filled with --bg-color

Notes:
    * Formats without transparency (jpg, bmp) get the removed background filled
      with a solid colour, because they cannot store a see-through background.
    * With "-f same", GIF files are saved as PNG (GIF transparency is poor quality).
    * The first run downloads the AI model (~170 MB) once.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageOps
from rembg import new_session, remove

try:
    import oxipng  # optional: pip install pyoxipng
except ImportError:
    oxipng = None

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif",
}

# extension -> (Pillow format name, supports transparency)
FORMAT_INFO = {
    ".png": ("PNG", True),
    ".jpg": ("JPEG", False),
    ".jpeg": ("JPEG", False),
    ".jfif": ("JPEG", False),
    ".webp": ("WEBP", True),
    ".tif": ("TIFF", True),
    ".tiff": ("TIFF", True),
    ".bmp": ("BMP", False),
}

# what -f value maps to which output extension
CHOICE_TO_EXT = {
    "png": ".png",
    "jpg": ".jpg",
    "webp": ".webp",
    "tiff": ".tiff",
    "bmp": ".bmp",
}


def find_images(input_dir: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def output_extension(src: Path, fmt: str) -> str:
    """Decide the output extension for a given source file."""
    if fmt == "same":
        ext = src.suffix.lower()
        # GIF can't hold smooth transparency, so fall back to PNG
        return ext if ext in FORMAT_INFO else ".png"
    return CHOICE_TO_EXT[fmt]


def flatten(img: Image.Image, bg_rgb) -> Image.Image:
    """Paste an RGBA image onto a solid background (for JPG/BMP)."""
    base = Image.new("RGB", img.size, bg_rgb)
    base.paste(img, mask=img.getchannel("A"))
    return base


def clean_transparent(img: Image.Image) -> Image.Image:
    """Set the hidden colour of fully transparent pixels to black.
    Visually identical (those pixels are invisible) but compresses far better."""
    arr = np.array(img)
    arr[arr[..., 3] == 0] = 0
    return Image.fromarray(arr, "RGBA")


def save_image(result: Image.Image, dst: Path, args, bg_rgb):
    pil_format, has_alpha = FORMAT_INFO[dst.suffix.lower()]
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not has_alpha:
        result = flatten(result, bg_rgb)
    elif pil_format in ("PNG", "TIFF"):
        result = clean_transparent(result)

    lossy = args.compress is not None and args.compress < 100

    if pil_format == "PNG":
        if lossy:
            # Lossy PNG: reduce the colour palette (transparency is kept)
            colors = max(2, round(256 * args.compress / 100))
            result = result.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
        # Otherwise PNG is lossless; compress_level/optimize only affect size/speed
        result.save(dst, format="PNG", compress_level=args.compress_level, optimize=True)
        if args.oxipng and oxipng is not None:
            oxipng.optimize(str(dst), level=4)  # extra lossless squeeze
    elif pil_format == "JPEG":
        quality = args.compress if args.compress is not None else 100
        result.save(dst, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2,
                    optimize=True)
    elif pil_format == "WEBP":
        if lossy:
            result.save(dst, format="WEBP", lossless=False, quality=args.compress, method=6)
        else:
            result.save(dst, format="WEBP", lossless=True, quality=100, method=6)
    elif pil_format == "TIFF":
        result.save(dst, format="TIFF", compression="tiff_lzw")
    else:  # BMP
        result.save(dst, format=pil_format)


def process_image(src: Path, dst: Path, session, args, bg_rgb):
    with Image.open(src) as img:
        # Respect camera rotation info, then convert to a mode rembg handles well
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")

        result = remove(
            img,
            session=session,
            alpha_matting=args.alpha_matting,
        )

    save_image(result, dst, args, bg_rgb)


def main():
    parser = argparse.ArgumentParser(description="Bulk background remover")
    parser.add_argument("input", type=Path, help="Folder containing input images")
    parser.add_argument("output", type=Path, help="Folder to save background-removed images")
    parser.add_argument("-f", "--format", default="png",
                        choices=["png", "same", "jpg", "webp", "tiff", "bmp"],
                        help="Output format. 'same' keeps each image's original extension. "
                             "Default: png (lossless, transparent)")
    parser.add_argument("--bg-color", default="white",
                        help="Background colour for formats without transparency "
                             "(jpg, bmp). Name or hex, e.g. white, black, '#f5f5f5'. "
                             "Default: white")
    parser.add_argument("--compress", type=int, default=None, choices=range(1, 101),
                        metavar="N",
                        help="Optional lossy compression, N = 1-100 (e.g. 80, 60, 40, 20). "
                             "Lower = smaller file, lower quality. Not set = no quality loss. "
                             "Works for jpg, webp and png; tiff/bmp are unaffected")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Also process images in sub-folders (keeps folder structure)")
    parser.add_argument("-m", "--model", default="u2net",
                        help="rembg model: u2net (default), isnet-general-use, "
                             "u2netp (fast), silueta, birefnet-general, ...")
    parser.add_argument("-c", "--png-level", dest="compress_level", type=int, default=9,
                        choices=range(0, 10), metavar="0-9",
                        help="PNG compression level (still lossless). 0 = fastest/biggest, "
                             "9 = slowest/smallest. Default 9")
    parser.add_argument("--oxipng", action="store_true",
                        help="Extra lossless PNG shrinking (needs: pip install pyoxipng). "
                             "Slower, smaller files")
    parser.add_argument("-a", "--alpha-matting", action="store_true",
                        help="Smoother edges on hair/fur (slower)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process images even if the output file already exists")
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"Input folder not found: {args.input}")

    try:
        bg_rgb = ImageColor.getrgb(args.bg_color)[:3]
    except ValueError:
        sys.exit(f"Invalid --bg-color: {args.bg_color}")

    images = find_images(args.input, args.recursive)
    if not images:
        sys.exit("No supported images found.")

    if args.format in ("jpg", "bmp"):
        print(f"Note: {args.format.upper()} has no transparency; background will be filled "
              f"with '{args.bg_color}'.")

    if args.compress is not None and args.compress < 100:
        print(f"Note: --compress={args.compress} is lossy for jpg/webp/png "
              f"(tiff/bmp are not affected).")

    args.output.mkdir(parents=True, exist_ok=True)
    session = new_session(args.model)  # load the model once, reuse for all images

    iterator = tqdm(images, unit="img") if tqdm else images
    done = skipped = failed = 0
    used_outputs = set()  # avoid a.jpg and a.png overwriting each other

    for src in images if tqdm is None else iterator:
        rel = src.relative_to(args.input)
        dst = (args.output / rel).with_suffix(output_extension(src, args.format))

        # Same output name already produced in this run (e.g. photo.jpg + photo.png)
        if dst in used_outputs:
            dst = dst.with_name(f"{dst.stem}_{src.suffix.lstrip('.').lower()}{dst.suffix}")
        used_outputs.add(dst)

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            process_image(src, dst, session, args, bg_rgb)
            done += 1
        except Exception as exc:  # keep going if one file is broken
            failed += 1
            print(f"\n[FAILED] {src}: {exc}", file=sys.stderr)

    print(f"\nFinished. Processed: {done} | Skipped (already exist): {skipped} | Failed: {failed}")
    print(f"Output folder: {args.output.resolve()}")


if __name__ == "__main__":
    main()