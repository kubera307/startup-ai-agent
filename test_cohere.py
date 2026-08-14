import cohere
from dotenv import load_dotenv
import os

load_dotenv()

client = cohere.ClientV2(
    api_key=os.getenv(
        "COHERE_API_KEY"
    )
)

response = client.chat(

    model="command-a-plus-05-2026",

    messages=[

        {
            "role": "user",

            "content": [
                {
                    "type": "text",

                    "text":
                        "Give me one startup idea."
                }
            ]
        }

    ]
)


for item in response.message.content:

    if getattr(
        item,
        "type",
        None
    ) == "text":

        print(item.text)