from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

N_ERRORS = [
    "N1694.jpg", "N2459.jpg", "N209.jpg", "N1699.jpg", "N71.jpg",
    "N2419.jpg", "N2052.jpg", "N423.jpg", "N185.jpg", "N223.jpg",
    "N2066.jpg", "N30.jpg", "N2420.jpg", "N1680.jpg", "N2334.jpg",
    "N1737.jpg", "N2340.jpg", "N2351.jpg", "N2276.jpg", "N1853.jpg"
]

M_ERRORS = [
    "M2306.jpg", "M2149.jpg", "M1901.jpg", "M737.jpg", "M1106.jpg",
    "M1346.jpg", "M1488.jpg", "M1109.jpg", "M1206.jpg"
]

BASE = Path(
    "dataset/ASL/asl_alphabet_train/asl_alphabet_train"
)

OUTPUT = Path("backend/modelA/nm_error_images.jpg")

THUMB_W = 180
THUMB_H = 180
LABEL_H = 40
COLS = 5

items = []

for filename in N_ERRORS:
    items.append(("N -> M", BASE / "N" / filename))

for filename in M_ERRORS:
    items.append(("M -> N", BASE / "M" / filename))

rows = (len(items) + COLS - 1) // COLS

sheet = Image.new(
    "RGB",
    (COLS * THUMB_W, rows * (THUMB_H + LABEL_H)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, (label, path) in enumerate(items):

    image = Image.open(path).convert("RGB")
    image.thumbnail((THUMB_W - 10, THUMB_H - 10))

    x = (index % COLS) * THUMB_W
    y = (index // COLS) * (THUMB_H + LABEL_H)

    image_x = x + (THUMB_W - image.width) // 2
    image_y = y + (THUMB_H - image.height) // 2

    sheet.paste(image, (image_x, image_y))

    draw.text(
        (x + 5, y + THUMB_H + 8),
        label,
        fill="black"
    )

sheet.save(OUTPUT, quality=95)

print()
print("N / M ERROR CONTACT SHEET CREATED")
print("=" * 60)
print(f"Images: {len(items)}")
print(f"Output: {OUTPUT.resolve()}")
