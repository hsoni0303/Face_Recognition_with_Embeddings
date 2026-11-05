"""Face recognition CLI.

  python main.py register "Alice" photos/alice.jpg   # or --camera
  python main.py identify group.jpg                  # or --camera
  python main.py live                                # webcam with labels (q to quit)
"""

import argparse

from PIL import Image

from face_id import FaceEncoder, Gallery


def grab_frame(camera=0):
    import cv2

    cap = cv2.VideoCapture(camera)
    for _ in range(10):  # let auto-exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read from camera.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_image(args):
    return grab_frame() if args.camera else Image.open(args.image).convert("RGB")


def register(args, encoder, gallery):
    _, emb = encoder.encode(load_image(args))
    if len(emb) != 1:
        raise SystemExit(f"Expected exactly one face, found {len(emb)}.")
    gallery.add(args.name, emb[0])
    gallery.save(args.gallery)
    print(f"Registered {args.name} ({len(gallery.names)} people in gallery).")


def identify(args, encoder, gallery):
    boxes, embs = encoder.encode(load_image(args))
    if not len(embs):
        print("No face found.")
    for box, emb in zip(boxes, embs):
        name, score = gallery.match(emb, args.threshold)
        print(f"{name:<15} similarity={score:.2f}  box={box.round().astype(int).tolist()}")


def live(args, encoder, gallery):
    import cv2

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, embs = encoder.encode(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for (x1, y1, x2, y2), emb in zip(boxes.astype(int), embs):
            name, score = gallery.match(emb, args.threshold)
            color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Face recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gallery", default="gallery.npz", help="where embeddings are stored")
    p.add_argument("--threshold", type=float, default=0.7, help="min cosine similarity to accept a match")
    sub = p.add_subparsers(dest="cmd", required=True)

    for cmd in ("register", "identify"):
        s = sub.add_parser(cmd)
        if cmd == "register":
            s.add_argument("name")
        src = s.add_mutually_exclusive_group(required=True)
        src.add_argument("image", nargs="?")
        src.add_argument("--camera", action="store_true")
    sub.add_parser("live")

    args = p.parse_args()
    commands = {"register": register, "identify": identify, "live": live}
    commands[args.cmd](args, FaceEncoder(), Gallery.load(args.gallery))


if __name__ == "__main__":
    main()
