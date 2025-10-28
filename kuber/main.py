import google.generativeai as genai
import json


API_KEY = "API_KEY"
genai.configure(api_key=API_KEY)

#analytics input
boxing_data = {
    "Punch Accuracy": {
        "Jab": 0.72,
        "Hook": 0.64,
        "Uppercut": 0.51
    },
    "Reaction Time": {
        "Jab": 0.42,
        "Hook": 0.58,
        "Uppercut": 0.60
    },
    "Punch Speed": 7.5,
    "Critical Hit Opportunities": {
        "Head": 0.68,
        "Body": 0.45
    },
    "% Avoided Punches": 0.73,
    "Stance Accuracy": 0.82,
    "Defense Consistency": 0.77
}

prompt = f"""
You are an AI boxing coach. 
Given these analytics from a boxing training session, analyze the performance and give
insightful feedback on strengths, weaknesses, and areas for improvement.

Analytics data:
{json.dumps(boxing_data, indent=2)}

Focus on:
- Punch accuracy (Jab/Hook/Uppercut)
- Punch reaction time
- Punch speed
- Critical Hit Opportunities (Head/Body)
- % of punches/obstacles avoided
- Stance accuracy and defensive consistency
"""
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(prompt)
print("\n===== BOXING PERFORMANCE FEEDBACK =====\n")
print(response.text)
