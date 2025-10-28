from transformers import pipeline 

messages = [
        {"role": "system", "content": "You are an enthusiastic boxing coach. You will be given JSON data and must give advice to the client for how to improve."},
        {"role": "user", "content": "{hands_dropped: true, punch_speed: 10, foot_rotation: 19.7}"},
        ]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot(messages)
