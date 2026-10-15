"""
Chatbot module for GroundedCXR.

Implements the conversational "GroundedCXR AI Assistant" layer, built on
top of (not replacing) the ResNet18 classifier. The assistant is grounded
on the current prediction_context dict produced by
inference.build_prediction_context(), so it answers using the actual
numbers for the X-ray currently on screen, not invented values.

Backend: Google Gemini via the current `google-genai` SDK (NOT the
deprecated `google-generativeai` package). The API key is read from the
GEMINI_API_KEY environment variable (via a local .env file) and is never
hardcoded, printed, or shown in the UI.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

DEFAULT_MODEL = "gemini-3-flash-preview"

# System-level behavior for the assistant. Combined at call time with the
# live prediction context for the current X-ray.
SYSTEM_INSTRUCTIONS = """
You are the "GroundedCXR AI Explanation Assistant", the conversational
explanation layer for an academic/research chest X-ray classification
prototype called GroundedCXR.

GroundedCXR's actual classifier is a trained ResNet18 model. You are NOT
the classifier and you do NOT reclassify or re-examine the X-ray yourself.
Your only job is to explain, in plain language, the prediction that the
ResNet18 model already produced, and to answer general educational
questions about chest X-rays, pneumonia, confidence scores, Grad-CAM, and
how ResNet18-style models work at a high level.

You will be given the CURRENT prediction context (model output) for the
X-ray the user is looking at, as a block of text before the conversation.
Always use those exact values when answering questions about "this"
prediction, confidence, or probabilities. Never invent or guess numbers
that were not given to you in that context.

Rules you must always follow:
- Never state or imply that the patient definitely has or does not have
  pneumonia. Only report what the model predicted and with what
  probability, e.g. "The GroundedCXR classification model predicted
  PNEUMONIA with 94.2% probability. This is an AI model prediction, not
  a clinical diagnosis."
- Never claim to be a radiologist or doctor, and never claim this tool
  replaces one.
- When explaining confidence, make clear it is a softmax probability
  from the model, not a measure of clinical certainty.
- When explaining Grad-CAM, describe it as highlighting the regions of
  the image that most influenced the model's prediction, and explicitly
  say it is NOT proof that disease is present in that region.
- Never invent symptoms, medical history, X-ray findings, or laboratory
  results that were not given to you.
- Never recommend treatment or medication based solely on this model.
- For general medical questions (e.g. "what is pneumonia?"), you may give
  brief, accurate, educational information, but always frame it as
  general knowledge, not a diagnosis for this specific patient.
- Always encourage consulting a qualified healthcare professional for
  actual medical interpretation or diagnosis.
- Keep answers concise, clear, and non-alarmist. Avoid jargon unless you
  also explain it, since the audience may include students and patients.
""".strip()


class ChatbotConfigError(Exception):
    """Raised when the chatbot cannot be used because of a configuration problem."""


def is_api_key_configured() -> bool:
    """Returns True if a GEMINI_API_KEY is available in the environment."""
    return bool(os.getenv("GEMINI_API_KEY"))


def _get_client() -> genai.Client:
    """
    Creates a Gemini client using the API key from the environment.

    Raises:
        ChatbotConfigError: if GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ChatbotConfigError(
            "Gemini API key is not configured. Please add GEMINI_API_KEY "
            "to your .env file."
        )
    return genai.Client(api_key=api_key)


def _get_model_name() -> str:
    """Reads the Gemini model name from GEMINI_MODEL, with a safe default."""
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _format_prediction_context(prediction_context: dict) -> str:
    """
    Turns the structured prediction_context dict (from
    inference.build_prediction_context()) into a short text block the
    model can read as grounding context.
    """
    return (
        "Current GroundedCXR model output for the X-ray in this session:\n"
        "- Classification model: ResNet18 (binary: NORMAL vs PNEUMONIA)\n"
        f"- Prediction: {prediction_context['prediction']}\n"
        f"- Confidence (probability of the predicted class): "
        f"{prediction_context['confidence'] * 100:.2f}%\n"
        f"- NORMAL probability: {prediction_context['normal_probability'] * 100:.2f}%\n"
        f"- PNEUMONIA probability: {prediction_context['pneumonia_probability'] * 100:.2f}%\n"
        "- Grad-CAM explanation: available for this image (highlights "
        "regions that influenced the prediction)\n"
    )


def _to_gemini_contents(chat_history: list, user_question: str) -> list:
    """
    Converts our {"role": "user"/"assistant", "content": str} chat_history
    format into the list of role/parts dicts the Gemini API expects.
    Gemini uses "model" (not "assistant") for the assistant's turns.
    """
    contents = []
    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_question}]})
    return contents


def get_chatbot_response(
    user_question: str,
    prediction_context: dict,
    chat_history: list,
) -> str:
    """
    Generates a grounded conversational response about the current X-ray
    prediction using the Gemini API.

    Args:
        user_question: the user's current question.
        prediction_context: the structured dict produced by
            inference.build_prediction_context() for the X-ray currently
            being discussed.
        chat_history: list of {"role": "user"|"assistant", "content": str}
            dicts representing the conversation so far, for THIS X-ray
            only (the caller is responsible for resetting this list when
            a new X-ray is uploaded).

    Returns:
        str: the assistant's reply, or a user-friendly "⚠️ ..." message
        if the question was empty or the API call failed. This function
        never raises for API/config errors; it returns an error string
        instead so the UI can display it directly.
    """
    if not user_question or not user_question.strip():
        return "⚠️ Please enter a question."

    try:
        client = _get_client()
    except ChatbotConfigError as e:
        return f"⚠️ {e}"

    # Build the full system prompt: fixed behavior rules + the live
    # numbers for this specific X-ray, so the model always answers using
    # the real prediction instead of guessing.
    full_instructions = SYSTEM_INSTRUCTIONS + "\n\n" + _format_prediction_context(prediction_context)
    contents = _to_gemini_contents(chat_history, user_question)

    try:
        response = client.models.generate_content(
            model=_get_model_name(),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=full_instructions,
            ),
        )
        return response.text
    except Exception as e:
        error_text = str(e).lower()
        if "quota" in error_text or "rate" in error_text or "429" in error_text:
            return (
                "⚠️ The Gemini API quota/rate limit has been reached. "
                "Please try again later or check your API plan."
            )
        return (
            "⚠️ The GroundedCXR Assistant could not generate a response "
            f"right now ({e}). Please check your internet connection and "
            "API key, then try again."
        )