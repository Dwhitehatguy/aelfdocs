# AelfDocs

AelfDocs is a RAG-enabled chat model for the Aelf blockchain documentation. This is to help developers understand the aelf blockchain better.

- [Read project's write-up](https://docs.google.com/document/d/1VRRnNzpAlYGCbKqTI3AqgeOMXc_NPqcxCpwm4x5NhSg/edit?usp=sharing)

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
    cp .streamlit/example.secrets.toml .streamlit/secrets.toml
```

Enter your google api key in secrets.toml!

Then, run:
```
    poetry shell ; streamlit run aelfdocs/aelfdocs_ui/AelfDocs.py
```
