import gradio as gr
import os
import torch

from timeit import default_timer as timer
from typing import Tuple, Dict
from utils import get_model_and_test_transforms
import requests
from dotenv import load_dotenv

load_dotenv()

# calories_food101.py

calories_per_100g = {
    "apple_pie": 237,
    "baby_back_ribs": 320,
    "baklava": 430,
    "beef_carpaccio": 180,
    "beef_tartare": 195,
    "beet_salad": 75,
    "beignets": 289,
    "bibimbap": 560,
    "bread_pudding": 245,
    "breakfast_burrito": 305,
    "bruschetta": 150,
    "caesar_salad": 190,
    "cannoli": 340,
    "caprese_salad": 110,
    "carrot_cake": 326,
    "ceviche": 114,
    "cheese_plate": 350,
    "cheesecake": 321,
    "chicken_curry": 190,
    "chicken_quesadilla": 330,
    "chicken_wings": 290,
    "chocolate_cake": 371,
    "chocolate_mousse": 360,
    "churros": 444,
    "clam_chowder": 90,
    "club_sandwich": 330,
    "crab_cakes": 250,
    "creme_brulee": 330,
    "croque_madame": 450,
    "cup_cakes": 399,
    "deviled_eggs": 143,
    "donuts": 452,
    "dumplings": 180,
    "edamame": 121,
    "eggs_benedict": 284,
    "escargots": 90,
    "falafel": 333,
    "filet_mignon": 271,
    "fish_and_chips": 290,
    "foie_gras": 462,
    "french_fries": 312,
    "french_onion_soup": 58,
    "french_toast": 240,
    "fried_calamari": 175,
    "fried_rice": 163,
    "frozen_yogurt": 145,
    "garlic_bread": 345,
    "gnocchi": 149,
    "greek_salad": 120,
    "grilled_cheese_sandwich": 330,
    "grilled_salmon": 206,
    "guacamole": 160,
    "gyoza": 175,
    "hamburger": 295,
    "hot_and_sour_soup": 70,
    "hot_dog": 290,
    "huevos_rancheros": 220,
    "hummus": 166,
    "ice_cream": 207,
    "lasagna": 150,
    "lobster_bisque": 175,
    "lobster_roll_sandwich": 345,
    "macaroni_and_cheese": 164,
    "macarons": 497,
    "miso_soup": 40,
    "mussels": 172,
    "nachos": 300,
    "omelette": 154,
    "onion_rings": 411,
    "oysters": 81,
    "pad_thai": 139,
    "paella": 130,
    "pancakes": 227,
    "panna_cotta": 170,
    "peking_duck": 350,
    "pho": 109,
    "pizza": 266,
    "pork_chop": 231,
    "poutine": 428,
    "prime_rib": 280,
    "pulled_pork_sandwich": 298,
    "ramen": 436,
    "ravioli": 143,
    "red_velvet_cake": 350,
    "risotto": 130,
    "samosa": 308,
    "sashimi": 130,
    "scallops": 111,
    "seaweed_salad": 45,
    "shrimp_and_grits": 210,
    "spaghetti_bolognese": 160,
    "spaghetti_carbonara": 176,
    "spring_rolls": 146,
    "steak": 271,
    "strawberry_shortcake": 256,
    "sushi": 130,
    "tacos": 226,
    "takoyaki": 250,
    "tiramisu": 240,
    "tuna_tartare": 184,
    "waffles": 291
}


API_KEY=os.getenv("USDA_API_KEY")

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

model, test_transforms = get_model_and_test_transforms(num_classes=101)

model.load_state_dict(
    torch.load(
        "models/food101_20_percent_trained.pth",
        map_location=torch.device("cpu"),
    )
)

def get_calories(food_name: str) -> str:
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key":API_KEY, "query":food_name, "pagesize":1}
    resp = requests.get(url, params=params)
    data = resp.json()

    if resp.status_code == 200 and data.get("foods"):
        nutrients = data["foods"][0].get("foodnutrients",[])
        for nut in nutrients:
            if nut.get("nutrientName","").lower() in ("energy","energy-kcal"):
                return f"{int(nut['value'])} kcal"
    
    url = "https://world.openfoodfacts.org/api/v2/search"
    params = {"search_terms": food_name, "fields": "product_name,nutriments"}
    resp = requests.get(url, params=params)
    data = resp.json()

    if resp.status_code == 200 and data.get("products"):
        prod = data["products"][0]
        nutr = prod.get("nutriments", {})
        kcal = nutr.get("energy-kcal_100g") or nutr.get("energy-kcal")
        if kcal:
            return f"{int(kcal)} kcal"
    
    calories = calories_per_100g.get(food_name.lower().replace(" ", "_"))
    return f"{calories} kcal per 100g" if calories else "N/A"


def predict(img, history) -> Tuple[Dict, float, list]:
    start_timer = timer()
    img = test_transforms(img).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)

    probs = pred_probs[0]
    top5 = torch.topk(probs, k=5)

    predictions_with_calories = {}
    top_class = class_names[top5.indices[0]]
    top_cal = get_calories(top_class)
    calories_val = int(top_cal.split()[0]) if top_cal and top_cal[0].isdigit() else 0

    for idx, prob in zip(top5.indices, top5.values):
        class_name = class_names[idx]
        confidence = float(prob)
        calories = get_calories(class_name)
        predictions_with_calories[f"{class_name} ({calories})"] = confidence

    pred_time = round(timer() - start_timer, 5)

    # Append to history
    if history is None:
        history = []
    history.append({"Food": top_class, "Calories (kcal)": calories_val})
    total_kcal = sum([item["Calories (kcal)"] for item in history])
    table_data = [[item["Food"], item["Calories (kcal)"]] for item in history]
    return predictions_with_calories, pred_time, table_data, total_kcal


title = "Calorie Tracker"
description = "Project made hoping Blackrose recruits me"

example_list = [["examples/" + example] for example in os.listdir("examples")]

with gr.Blocks() as demo:
    gr.Markdown("# Calorie Tracker 🍽️\nMade hoping Blackrose recruits me")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload food image")
            predict_btn = gr.Button("Analyze")
        with gr.Column():
            gr.Examples(
                examples=example_list,
                inputs=img_input,
                label="Example Images"
            )
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predictions")
            time_output = gr.Number(label="Prediction time (s)")
    
    gr.Markdown("## Cumulative Food History")
    table_output = gr.Dataframe(headers=["Food", "Calories (kcal)"], interactive=False)
    history_state = gr.State([])
    total_calories_output = gr.Number(label="Total Calories (kcal)", value=0)

    predict_btn.click(
        predict,
        inputs=[img_input, history_state],
        outputs=[label_output, time_output, table_output, total_calories_output],
    )


demo.launch()