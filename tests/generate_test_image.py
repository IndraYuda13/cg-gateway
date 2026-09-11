from PIL import Image
import os

os.makedirs("data", exist_ok=True)
img = Image.new("RGB", (100, 100), color="red")
img.save("data/test_red_square.png", format="PNG")
print("Saved data/test_red_square.png")
