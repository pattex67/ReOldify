import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="reoldify",
        description="ReOldify — Colorize and restore old images and video using deep learning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # colorize command
    colorize_parser = subparsers.add_parser("colorize", help="Colorize an image or video")
    colorize_parser.add_argument("input", type=str, help="Path to input image or video file")
    colorize_parser.add_argument("-o", "--output", type=str, default=None, help="Output file path (default: input_colorized.ext)")
    colorize_parser.add_argument(
        "--model",
        type=str,
        choices=["artistic", "stable", "video"],
        default=None,
        help="Model to use (default: artistic for images, video for videos)",
    )
    colorize_parser.add_argument(
        "--render-factor",
        type=int,
        default=35,
        help="Render factor — higher means better quality but slower (default: 35)",
    )
    colorize_parser.add_argument("--cpu", action="store_true", help="Force CPU inference (default: use GPU if available)")
    colorize_parser.add_argument("--no-watermark", action="store_true", help="Disable watermark on output")

    # version
    parser.add_argument("--version", action="version", version="ReOldify 1.0.0")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "colorize":
        _run_colorize(args)


def _run_colorize(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Detect if input is video
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    is_video = input_path.suffix.lower() in video_extensions

    # Set default model based on input type
    model = args.model
    if model is None:
        model = "video" if is_video else "artistic"

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = input_path.suffix
        output_path = input_path.with_name(f"{input_path.stem}_colorized{suffix}")

    # Configure device
    from deoldify import device as device_settings
    from deoldify.device_id import DeviceId

    if args.cpu:
        device_settings.set(DeviceId.CPU)
        print("Using CPU for inference")
    else:
        device_settings.set(DeviceId.GPU0)
        print("Using GPU for inference")

    watermarked = not args.no_watermark
    render_factor = args.render_factor

    if is_video:
        _colorize_video(input_path, output_path, model, render_factor, watermarked)
    else:
        _colorize_image(input_path, output_path, model, render_factor, watermarked)


def _colorize_image(input_path, output_path, model, render_factor, watermarked):
    from deoldify.visualize import get_image_colorizer

    print(f"Colorizing image: {input_path}")
    print(f"Model: {model} | Render factor: {render_factor}")

    artistic = model == "artistic"
    colorizer = get_image_colorizer(artistic=artistic, render_factor=render_factor)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = colorizer.get_transformed_image(
        path=input_path,
        render_factor=render_factor,
        watermarked=watermarked,
    )
    result.save(str(output_path))
    print(f"Done! Saved to: {output_path}")


def _colorize_video(input_path, output_path, model, render_factor, watermarked):
    from deoldify.visualize import get_video_colorizer, get_artistic_video_colorizer

    print(f"Colorizing video: {input_path}")
    print(f"Model: {model} | Render factor: {render_factor}")

    if model == "artistic":
        colorizer = get_artistic_video_colorizer(render_factor=render_factor)
    else:
        colorizer = get_video_colorizer(render_factor=render_factor)

    import shutil
    # Copy source to the expected location
    colorizer.source_folder.mkdir(parents=True, exist_ok=True)
    source_in_folder = colorizer.source_folder / input_path.name
    if not source_in_folder.exists() or source_in_folder != input_path:
        shutil.copy2(str(input_path), str(source_in_folder))

    result_path = colorizer.colorize_from_file_name(
        file_name=input_path.name,
        render_factor=render_factor,
        watermarked=watermarked,
    )

    if result_path != output_path:
        shutil.copy2(str(result_path), str(output_path))

    print(f"Done! Saved to: {output_path}")


if __name__ == "__main__":
    main()
