# AelfDocs

AelfDocs is a RAG enabled model for the Aelf blockchain documentation

## Technologies

1. Atlas MongoDB
2. Llama Index
3. Google's Gemini
4. Streamlit

## Setup

Clone this repo: `git clone https://github.com/Dwhitehatguy/aelfdocs ; cd aelfdocs`

Then, install dependencies: `poetry install`

## Run AelfDocs

Populate your environment variables

```
    cp .streamlit/example.secrets.toml secrets.toml
```

Enter your google api key in secrets.toml!

Then, run: `poetry shell ; streamlit run aelfdocs/aelfdocs_ui/AelfDocs.py`
