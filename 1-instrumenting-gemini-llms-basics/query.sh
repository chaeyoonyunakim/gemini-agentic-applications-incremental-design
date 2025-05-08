curl \
  -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY} \
  -H 'Content-Type: application/json' \
  -d @<(echo '{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Tell me about Google. I need the cie description, the cie symbol, the sector, and the industry."
        }
      ]
    }
  ],
  "generationConfig": {
    "temperature": 0,
    "topK": 1,
    "maxOutputTokens": 8192,
    "responseMimeType": "application/json"
  }
}')
