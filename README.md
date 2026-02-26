# TradingWeather

A learning project exploring LangChain implementation — Streamlit app that combines weather lookup, Alpaca trading account info, RSI, and VWAP via an LLM agent.

## Run locally

```bash
cd TradingWeather
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run tradingweather.py
```

## Deploy (Render + custom domain)

This repo includes a `Dockerfile` so it can be deployed as a container.

1. Create a new **Web Service** on Render and connect this GitHub repo.
2. Choose **Docker** as the environment.
3. Set environment variables in Render (do **not** upload `.env`):
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL` (optional)
   - `OPENWEATHER_API_KEY`
   - `AlpKey`
   - `AlpSecret`
4. Deploy.
5. In Render → your service → **Settings → Custom Domains**, add:
   - `moneynagar.com`
   - `www.moneynagar.com`
6. Update DNS at your domain provider (or Cloudflare) with the records Render shows.
