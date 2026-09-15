"""
capture_photos.py — Interactive webcam snapshot tool for enrollment.

Usage:
    python -m scripts.capture_photos --name your_name

Keys in the window:
    SPACE / s    take a photo (saved to data/faces/<name>/)
    q / ESC     quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture enrollment photos from the webcam."
    )
    parser.add_argument(
        "--name", required=True,
        help="Person's name — photos are saved to data/faces/<name>/",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--res", default="640x480",
        help="Camera resolution WxH (default: 640x480)",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Auto-quit after this many photos are taken (default: 5, 0 = never)",
    )
    args = parser.parse_args()

    # Parse resolution
    try:
        w, h = (int(v) for v in args.res.lower().split("x"))
    except ValueError:
        print(f"[ERROR] Invalid --res '{args.res}'. Use format WxH e.g. 640x480")
        sys.exit(1)

    # Prepare output directory
    out_dir = Path(__file__).resolve().parents[1] / "data" / "faces" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Photos will be saved to: {out_dir}")

    # Open camera
    # use: cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) for windows
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}. Try --camera 1")
        sys.exit(1)

    photo_count = 0
    last_flash = 0.0
    FLASH_DURATION = 0.15          # seconds the white flash stays on screen
    window_name = f"Capture — {args.name}  |  SPACE/S = snap  |  Q/ESC = quit"

    print("\n  +------------------------------------------+")
    print("  |  Press SPACE or S to take a photo        |")
    print("  |  Press Q or ESC to quit                  |")
    print("  +------------------------------------------+\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — retrying…")
            time.sleep(0.05)
            continue

        display = frame.copy()

        # ── Flash overlay ──────────────────────────────────────────────────
        if time.time() - last_flash < FLASH_DURATION:
            alpha = 1.0 - (time.time() - last_flash) / FLASH_DURATION
            white = display.copy()
            white[:] = (255, 255, 255)
            cv2.addWeighted(white, alpha, display, 1 - alpha, 0, display)

        # ── HUD overlay ────────────────────────────────────────────────────
        h_frame, w_frame = display.shape[:2]
        bar_h = 50
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h_frame - bar_h), (w_frame, h_frame), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        count_text = f"Photos taken: {photo_count}"
        if args.count > 0:
            count_text += f" / {args.count}"
        cv2.putText(display, count_text,
                    (12, h_frame - bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "SPACE/S: snap    Q/ESC: quit",
                    (12, h_frame - bar_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Person label ───────────────────────────────────────────────────
        cv2.putText(display, f"Enrolling: {args.name}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

        # ── Guide rectangle (face positioning guide) ───────────────────────
        cx, cy = w_frame // 2, h_frame // 2
        rw, rh = w_frame // 4, int(h_frame * 0.45)
        cv2.rectangle(display,
                      (cx - rw, cy - rh),
                      (cx + rw, cy + rh),
                      (0, 200, 255), 2)
        cv2.putText(display, "position face here",
                    (cx - rw, cy - rh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        # ── Quit ───────────────────────────────────────────────────────────
        if key in (ord("q"), 27):          # q or ESC
            print("[INFO] Quit by user.")
            break

        # ── Snap ───────────────────────────────────────────────────────────
        if key in (ord(" "), ord("s")):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = out_dir / f"{args.name}_{timestamp}_{photo_count + 1:02d}.jpg"
            cv2.imwrite(str(filename), frame)   # save original (no overlay)
            photo_count += 1
            last_flash = time.time()
            print(f"  📷 Saved: {filename.name}  ({photo_count} total)")

            # Auto-quit once target count reached
            if args.count > 0 and photo_count >= args.count:
                print(f"\n[INFO] Reached {args.count} photos — done!")
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'-'*50}")
    print(f"  Saved {photo_count} photo(s) to: {out_dir}")
    if photo_count > 0:
        print(f"  Run enroll next:")
        print(f"    .venv\\Scripts\\python.exe -m scripts.enroll")
    print(f"{'-'*50}\n")


if __name__ == "__main__":
    main()