import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CARD_WIDTH = 600
CARD_HEIGHT = 400

def create_card_image(row, output_path, variation=1):
    # Select background & styling based on variation
    if variation == 1:
        bg_color = (245, 247, 250)
        card_bg = (255, 255, 255)
        text_color = (30, 41, 59)
        accent_color = (37, 99, 235)
    else:
        bg_color = (240, 240, 240)
        card_bg = (255, 255, 250)
        text_color = (20, 20, 20)
        accent_color = (15, 118, 110)

    image = Image.new('RGB', (CARD_WIDTH, CARD_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Draw Inner Card Frame
    draw.rectangle([20, 20, CARD_WIDTH - 20, CARD_HEIGHT - 20], fill=card_bg, outline=accent_color, width=2)

    # Header
    draw.text((40, 35), f"CLAIM SUMMARY CARD: {row['Claim_ID']}", fill=accent_color)
    draw.line([(40, 60), (CARD_WIDTH - 40, 60)], fill=accent_color, width=2)

    # Claim Information (Excludes ML prediction or confidence scores)
    lines = [
        f"Product Category: {row['Product_Category']}",
        f"Brand / Model: {row['Brand']} ({row['Model_Number']})",
        f"Product Age: {row['Product_Age_Months']} Months",
        f"Remaining Warranty: {row['Remaining_Warranty_Months']} Months",
        f"Reported Fault: {row['Fault_Type']}",
        f"Receipt Available: {'YES' if row['Has_Receipt'] else 'NO'}",
        f"Serial Number Match: {'MATCHED' if row['Serial_Number_Match'] else 'MISMATCHED'}",
        f"Previous Unauth Repairs: {'YES' if row['Previous_Unauthorized_Repairs'] else 'NO'}",
        f"Missing Documents: {row['Missing_Documents_Count']}"
    ]

    y_pos = 80
    for line in lines:
        draw.text((40, y_pos), line, fill=text_color)
        y_pos += 32

    # Footer
    draw.text((40, CARD_HEIGHT - 45), f"AssureX Claim Engine System | ID: {row['Claim_ID']}", fill=(100, 116, 139))

    image.save(output_path)

def generate_cards_for_split(csv_path, output_dir, is_train=False):
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        cls_folder = os.path.join(output_dir, str(row['Claim_Class']).replace(" ", "_"))
        os.makedirs(cls_folder, exist_ok=True)
        
        # Variation 1
        img_name_1 = f"{row['Claim_ID']}_v1.png"
        create_card_image(row, os.path.join(cls_folder, img_name_1), variation=1)

        # Variation 2 for training data (to meet >= 2100 images requirement)
        if is_train:
            img_name_2 = f"{row['Claim_ID']}_v2.png"
            create_card_image(row, os.path.join(cls_folder, img_name_2), variation=2)

def main():
    print("Generating Claim Summary Cards...")
    generate_cards_for_split('data/train/train_claims.csv', 'data/train/cards', is_train=True)
    generate_cards_for_split('data/val/val_claims.csv', 'data/val/cards', is_train=False)
    generate_cards_for_split('data/test/test_claims.csv', 'data/test/cards', is_train=False)
    print("All Claim Summary Cards created successfully!")

if __name__ == '__main__':
    main()