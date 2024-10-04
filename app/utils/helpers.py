import json
from PIL import ImageDraw

def get_cure_recommendations(disease_name):
    # Load cure data from a JSON file
    with open('app/data/cure_data.json', 'r') as f:
        cures = json.load(f)
    return cures.get(disease_name, ["No cure information available."])

def draw_red_square(img):
    # Simulate infected area detection by drawing a red square
    draw = ImageDraw.Draw(img)
    width, height = img.size
    draw.rectangle(
        [(width * 0.3, height * 0.3), (width * 0.7, height * 0.7)],
        outline="red",
        width=5
    )
    return img
