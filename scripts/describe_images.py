import openai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import tempfile

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

input_folder = "input_to_annotate"
prompt_text = """
Please, describe the image in a concise manner.
The description should answer the following questions: who, what, when, where.
The yellow suited man in a yellow fedora is called Stalwart, he is the main character.
Always use his name in the description if he is on the panel.
Please, also include in the answer the style of the picture eg. "noir" or "web comic". 
Please, end all style descriptions with the following text: 'Stalwart style'
Please, also describe panel type using comic related expressions, like "action", "close up" etc. 
The goal is to annotate images for LORA training."""

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        original_path = os.path.join(input_folder, filename)

        # Create a resized and compressed copy
        with Image.open(original_path) as img:
            img = img.convert("RGB")
            img = img.resize((768, 768), Image.LANCZOS)

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name, format="JPEG", quality=85)
                compressed_path = tmp.name

        # Read and encode compressed image
        with open(compressed_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                #max_tokens=300
            )

            description = response.choices[0].message.content
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(input_folder, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(description)

            print(f"Description for {filename}:")
            print(description)
            print("-" * 40)

        finally:
            # Clean up temp file
            os.remove(compressed_path)
