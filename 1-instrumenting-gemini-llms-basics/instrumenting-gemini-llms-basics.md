# instrumenting LLMs basics

## unstructured text response vs structured output from an end-user perspective

- prerequisites from `README.md`
- the most straightforward way to get a plain text response from text input with Google's LLMs is to go to `https://gemini.google.com/app`
- now if you feel like tinkering around but you don't know where to start, you can go try [Google AI Studio](https://aistudio.google.com/) to have a feel at what you can do with the Gemini API

- let's try to get a structured output from  the text input `Tell me about Google. I need the cie description, the cie symbol, the sector, and the industry.`, we get this result at an impressive speed:

```json
{
  "company_description": "Google LLC is a global technology company focusing on search engine technology, online advertising, cloud computing, software, and hardware.",
  "company_symbol": "GOOGL (Alphabet Inc. Class A shares), GOOG (Alphabet Inc. Class C shares)",
  "sector": "Communication Services",
  "industry": "Internet Content & Information"
}
```

- now, from the AI Studio, we can get the code directly to get such structured outputs directly from our terminal (you'll need an API key for this):

```bash
GEMINI_API_KEY="YOUR_API_KEY"

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
```

- you'll notice in the complete response object that you get citation metadata: this is because web search is a native Gemini 2 Flash tool that is called automatically by the model if required; however, you need to be aware that there are costs associated with this feature and that there is also a rate limit for grounding the LLM's responses with a web search when using the Gemini API

- Structured outputs are very interesting when you need to integrate with existing systems or when you need to format information for whatever business logic you need to apply. For instance, you can parse data from documents, extract data from images, format unstructured text to feed a CRM, etc. 
You can generate structured outputs real easily with the Gemini API with just a prompt and the specification of the output MIME type as `application/json`.
To get a more consistent and deterministic output (with the LLM selecting the highest probability tokens), you need to use a lower `temperature`  if you are to use structured outputs in a production environment. You can also set the `topK` parameter (how many tokens to consider for generating each word) to 1 so that the LLM only selects the most probable token (greedy decoding).

## unstructured text response vs structured output with Python code

- from the folder `1-instrumenting-gemini-llms-basics`, run `python3 -m venv venv && source venv/bin/activate`
- create a `requirements.txt` file with the following content:

```
langchain-core
langchain-google-genai
pydantic
python-dotenv
```

- `pip install -r requirements.txt`
- get plain text response from text input using LangChain

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = ChatPromptTemplate.from_template("Process this text: {input}")

llm = ChatGoogleGenerativeAI(
    max_tokens=None,
    model="gemini-2.0-flash",
    temperature=0,
    top_k=1
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "input": "Google LLC is a global technology company focusing on search engine technology, online advertising, cloud computing, software, and hardware."
})
print(result)
```

- oftentimes, you need to be absolutely sure what object you may get from the LLM when using structured output; for that you can use Pydantic classes:

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")

prompt = ChatPromptTemplate.from_messages(
    [("system", "Extract information from this text:"), ("human", "{input}")]
)

llm = ChatGoogleGenerativeAI(
    max_tokens=None,
    model="gemini-2.0-flash",
    temperature=0,
    top_k=1
)
structured_model = llm.with_structured_output(CompanyInfo)

chain = prompt | structured_model

result = chain.invoke(
    {
        "input": "Google LLC is a global technology company focusing on search engine technology, online advertising, cloud computing, software, and hardware."
    }
)
print(result)
```

## multimodal inputs

- you can also get a plain text response from image input =>

```python
from dotenv import load_dotenv
load_dotenv()

import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

vision_model = ChatGoogleGenerativeAI(
    max_tokens=None,
    model="gemini-2.0-flash",
    temperature=0,
    top_k=1
)
with open("example.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = vision_model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
    ])
])

print(response)
```

- structured output from image input

```python
from dotenv import load_dotenv

load_dotenv()

import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")


vision_model = ChatGoogleGenerativeAI(
    max_tokens=None, model="gemini-2.0-flash", temperature=0, top_k=1
).with_structured_output(CompanyInfo)

with open("example.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

response = vision_model.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Tell me about this company"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                },
            ]
        )
    ]
)

print(response)
```

## batching LLM calls

- get plain text responses from several text inputs using LangChain

```python
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    max_tokens=None, model="gemini-2.0-flash", temperature=0, top_k=1
)

prompt_template = ChatPromptTemplate.from_template("process this text: {input}")

chain = prompt_template | llm | StrOutputParser()

prompt_values = [
    {"input": "Apple Inc. is a multinational technology company headquartered in Cupertino, "
            "California that designs, manufactures, and markets consumer electronics, "
            "computer software, and online services."},
    {"input": "Google is a multinational technology company headquartered in Cupertino, "
            "California that designs, manufactures, and markets consumer electronics, "
            "computer software, and online services."},
]

result = chain.batch(prompt_values)
print(result) # you get a list with the two plain text outputs
```

... this is especially useful when you need to process inputs at scale; however be aware that the Gemini API has the following limits: 2000 requests and 4M tokens per minute (you'll get a 429 HTTP error if you exceed this limit).

- get structured output from several text inputs using LangChain

```python
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    cie_description: str = Field(description="Description of the company")
    cie_symbol: str = Field(description="Symbol of the company")
    industry: str = Field(description="Industry of the company")
    sector: str = Field(description="Sector of the company")

llm = ChatGoogleGenerativeAI(
    max_tokens=None, model="gemini-2.0-flash", temperature=0, top_k=1
).with_structured_output(CompanyInfo)

prompt_template = ChatPromptTemplate.from_template("process this text: {input}")

chain = prompt_template | llm

prompt_values = [
    {"input": "Apple Inc. is a multinational technology company headquartered in Cupertino, "
            "California that designs, manufactures, and markets consumer electronics, "
            "computer software, and online services."},
    {"input": "Google is a multinational technology company headquartered in Cupertino, "
            "California that designs, manufactures, and markets consumer electronics, "
            "computer software, and online services."},
]

result = chain.batch(prompt_values)
print(result)
```

