from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

# retrieve environment variables
ENDPOINT = ""
training_key = ""
prediction_key = ""
prediction_resource_id = ""

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# target_project_name = "Greatwall00"
# project_id = None

# # 프로젝트 리스트에서 이름이 일치하는 프로젝트의 ID만 출력
# for project in trainer.get_projects():
#     if project.name == target_project_name:
#         print(f"Project '{target_project_name}'의 ID: {project.id}")
#         project_id = project.id
#         break
# else:
#     print(f"이름이 '{target_project_name}'인 프로젝트를 찾을 수 없습니다.")
project_id = "28984b37-77e3-41aa-9261-f3a213099234"  # Replace with your actual project ID

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
ENDPOINT = "https://winkeycustomvision001-prediction.cognitiveservices.azure.com/"
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

results = predictor.classify_image_url(project_id,  # 프로젝트 ID
    "greatwall_model",
    "https://png.pngtree.com/png-clipart/20231019/original/pngtree-jajangmyeon-korean-food-png-image_13358305.png"
)

# Display the results.
for prediction in results.predictions:
    print("\t" + prediction.tag_name +
            ": {0:.2f}%".format(prediction.probability * 100))