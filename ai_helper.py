"""Gemini helpers for health assistant replies and personalised guidance."""
import os

from google import genai


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_api_key():
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.getenv(name)
        if key:
            return key
    try:
        import streamlit as st
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = st.secrets.get(name)
            if key:
                return key
    except Exception:
        return None
    return None


def _client():
    key = get_api_key()
    if not key:
        return None
    return genai.Client(api_key=key)


def _generate(prompt: str) -> str:
    client = _client()
    if not client:
        return ("⚠️ This feature is not configured on this computer yet. "
                "Add your Gemini API key to the `.env` file to enable it.")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or "I could not generate a response. Please try again."
    except Exception as e:
        return f"⚠️ Assistant error: {e}"


def explain_prediction(inputs: dict, prediction: int, risk: float) -> str:
    """Generate plain-English explanation of why the model gave this result."""
    verdict = "diabetic" if prediction == 1 else "non-diabetic"
    prompt = f"""You are a friendly healthcare assistant explaining a diabetes
risk result to a non-technical user.

A prediction model classified the user as {verdict} with a risk probability of
{risk:.1f}%.

User health inputs:
- Pregnancies: {inputs['Pregnancies']}
- Glucose: {inputs['Glucose']} mg/dL
- Blood Pressure: {inputs['BloodPressure']} mm Hg
- Skin Thickness: {inputs['SkinThickness']} mm
- Insulin: {inputs['Insulin']} mu U/ml
- BMI: {inputs['BMI']} kg/m2
- Diabetes Pedigree Function: {inputs['DiabetesPedigreeFunction']}
- Age: {inputs['Age']} years

Explain in 4-6 short bullet points using plain English. Mention the most
important risk factors and how they may affect the result. Do not diagnose the
user. End with one short reminder to consult a qualified doctor for medical
advice.
"""
    return _generate(prompt)


def diet_and_lifestyle_plan(inputs: dict, prediction: int, risk: float) -> str:
    """Generate a personalized 7-day diet + lifestyle plan."""
    band = "low" if risk < 30 else "moderate" if risk < 60 else "high"
    prompt = f"""You are a practical diabetes lifestyle assistant. Create a
personalised 7-day diet and lifestyle plan for this user.

Risk band: {band} ({risk:.1f}% diabetes risk)
Key parameters: BMI={inputs['BMI']}, Glucose={inputs['Glucose']} mg/dL,
Blood Pressure={inputs['BloodPressure']} mm Hg, Age={inputs['Age']}.

Format your response in markdown with these exact sections:

### Daily Diet Principles
4-5 short bullets tailored to the numbers.

### 7-Day Meal Plan
A markdown table with columns: Day | Breakfast | Lunch | Dinner | Snack.
Use practical Indian and global food options with diabetes-friendly portions.

### Exercise Plan
5 bullets covering type, frequency, and duration.

### Lifestyle Tips
4 bullets covering sleep, stress, hydration, and monitoring.

### Foods to Avoid
5 bullets.

Keep it encouraging and practical. End with a doctor-consultation reminder.
"""
    return _generate(prompt)


def health_assistant_reply(chat_messages: list, latest_prediction: dict | None) -> str:
    """Generate a health assistant reply from Streamlit chat history."""
    ctx = ""
    if latest_prediction:
        ctx = (
            "\nLatest user risk result:\n"
            f"- Result: {'DIABETIC' if latest_prediction['pred'] == 1 else 'NON-DIABETIC'}\n"
            f"- Risk score: {latest_prediction['risk']:.1f}%\n"
            f"- Inputs: {latest_prediction['inputs']}\n"
        )

    history_lines = []
    for msg in chat_messages[-12:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")

    prompt = f"""You are a friendly healthcare assistant for a diabetes awareness
website. Answer questions about diabetes symptoms, prevention, diet, exercise,
lifestyle, and the user's latest risk result when available.

Rules:
- Use simple, clear language.
- Do not claim to diagnose or prescribe treatment.
- Give practical health-awareness guidance.
- Always remind the user to consult a qualified doctor for medical advice when
  the question is medical or personal.
{ctx}
Conversation:
{chr(10).join(history_lines)}

Assistant:"""
    return _generate(prompt)
