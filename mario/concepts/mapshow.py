"""Print the RAM concept map as text at a few points of a level (sanity check). python mapshow.py 1-1 60 3"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cplay as C  # noqa: E402

lv = tuple(int(v) for v in sys.argv[1].split("-")); n = int(sys.argv[2]); act = int(sys.argv[3])
b = C.CBody(0, "ram", "lean", lv)
for k in range(n):
    b.step(act)
    if b.g.done:
        break
ram = b.g.env.ram
m = C.concept_map(ram)
print("x", b.g.x, "scr x", ram[0x3AD], "y", ram[0xCE], "B5", ram[0xB5], "power", ram[0x754], "done", b.g.done)
for i in range(5):
    print("enemy", i, ram[0xF + i], hex(ram[0x16 + i]), ram[0x6E + i] * 256 + ram[0x87 + i], ram[0xCF + i], ram[0xB6 + i])
for r in range(13):
    print("".join("E" if m[1, r, j] else "M" if m[4, r, j] else "P" if m[2, r, j] else "#" if m[0, r, j] else "?" if m[3, r, j] else "." for j in range(16)))
print(b.obs().shape)
b.g.screen
from PIL import Image
Image.fromarray(b.g.screen).save(Path(__file__).resolve().parent / "runs" / "mapshow.png")
