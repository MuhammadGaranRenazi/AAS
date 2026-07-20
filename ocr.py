import os
import re
import base64
import pandas as pd
from openai import OpenAI
from jiwer import cer

# =====================================
# LM Studio
# =====================================
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# =====================================
# Dataset Recognition
# =====================================
IMAGE_DIR = "Indonesian License Plate Recognition Dataset/images/test"
LABEL_DIR = "Indonesian License Plate Recognition Dataset/labels/test"

# =====================================
# Character Mapping
# =====================================
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

results = []

# =====================================
# Clean Prediction
# =====================================
def clean_prediction(text):

    text = text.upper()

    quoted = re.findall(r'"([^"]+)"', text)

    if quoted:
        plate = quoted[0]
        plate = re.sub(r'[^A-Z0-9]', '', plate)
        return plate

    text = re.sub(r'[^A-Z0-9]', '', text)

    matches = re.findall(
        r'[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,4}',
        text
    )

    if matches:
        return max(matches, key=len)

    return text

# =====================================
# Read Ground Truth from YOLO OCR Labels
# =====================================
def read_ground_truth(label_path):

    chars = []

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])

            chars.append(
                (
                    x_center,
                    CLASSES[class_id]
                )
            )

    chars.sort(key=lambda x: x[0])

    plate = ''.join(
        char for _, char in chars
    )

    return plate

# =====================================
# OCR Loop
# =====================================
for filename in os.listdir(IMAGE_DIR):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    txt_name = os.path.splitext(
        filename
    )[0] + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
        txt_name
    )

    if not os.path.exists(label_path):
        continue

    ground_truth = read_ground_truth(
        label_path
    )

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(
            f.read()
        ).decode()

    try:

        response = client.chat.completions.create(
            model="llava-1.6-mistral-7b",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                                    You are an OCR system.

                                    Read ONLY the main Indonesian license plate.

                                    Ignore:
                                    - expiration date
                                    - small numbers below the plate
                                    - province code
                                    - stickers
                                    - screws
                                    - frame text

                                    Return ONLY the main plate.

                                    Examples:

                                    Image:
                                    B 1234 ABC
                                    07•25

                                    Output:
                                    B1234ABC

                                    Image:
                                    AA 1971 KJ
                                    01•24

                                    Output:
                                    AA1971KJ

                                    Do not explain.
                                    Do not add any text.
                                    Return only the plate number.
                                    """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],

            temperature=0,
            top_p=0.1,
            max_tokens=20
        )

        raw_prediction = (
            response
            .choices[0]
            .message
            .content
        )

        prediction = clean_prediction(
            raw_prediction
        )

    except Exception as e:

        prediction = "ERROR"
        print(
            f"ERROR on {filename}:",
            e
        )

    cer_score = cer(
        ground_truth,
        prediction
    )

    results.append({
        "image": filename,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "CER_score": round(cer_score, 4)
    })

    print(
        filename,
        "| GT:",
        ground_truth,
        "| Pred:",
        prediction,
        "| CER:",
        round(cer_score, 4),
        "| RAW:", raw_prediction
    )

# =====================================
# Save CSV
# =====================================
df = pd.DataFrame(results)

df.to_csv(
    "ocr_results.csv",
    index=False
)

print("\n================================")
print("Total Images :", len(df))
print("Average CER  :", round(df["CER_score"].mean(), 4))
print("================================")