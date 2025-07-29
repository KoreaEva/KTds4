import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

endpoint = os.getenv("AI_SERVICE_ENDPOINT")
key = os.getenv("AI_SERVICE_KEY")

# input: Image file path
file_path = input("Enter the path to the image file: ")

with open(file_path, "rb") as image_file:
    image_data = image_file.read()

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)

def sample_analyze_all_image_file():
    result = client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS
        ],
        model_version="latest",
    )

    print("Image Analysis Result:")

    # Print caption
    if result.caption is not None:
        print(" Caption:")
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

    # Print dense captions
    if result.dense_captions is not None:
        print(" Dense Captions:")
        for caption in result.dense_captions.list:
            print(f"   '{caption.text}', {caption.bounding_box}, Confidence: {caption.confidence:.4f}")

    # Print tags
    if result.tags is not None:
        print(" Tags:")
        for tag in result.tags.list:
            print(f"   '{tag.name}', Confidence: {tag.confidence:.4f}")
    
    # Print objects
    object_boxes = []
    if result.objects is not None:
        print(" Objects:")
        for object in result.objects.list:
            print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")
            object_boxes.append(object.bounding_box)

    draw_bounding_boxes(file_path, object_boxes)

def draw_bounding_boxes(image_path, bounding_boxes):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for box in bounding_boxes:
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x, y), "Object", fill="red")
    
    image.show()

if __name__ == "__main__":
    sample_analyze_all_image_file()