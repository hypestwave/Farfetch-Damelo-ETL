# Farfetch Scraper Microservice

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

## Usage

POST /scrape

Body:
[
  "https://www.farfetch.com/..."
]

## Output

- Excel file (output.xlsx)
- Images stored in /images/{sku}/


## api key chatgpt

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


## ejecución

cd C:\Users\SISTEMAS\Downloads\farfetch_microservice
python -m uvicorn app:app --reload
