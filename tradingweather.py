import os
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import ChatOpenAI
from alpaca.trading.client import TradingClient


load_dotenv()


def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OPENWEATHER_API_KEY is not set."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        temperature = data["main"]["temp"]
        condition = data["weather"][0]["description"]
        return f"Weather in {city}: {temperature}°C, {condition}."
    except Exception as exc:
        return f"Failed to fetch weather for {city}: {exc}"


def get_account_balance(_: str = "") -> str:
    alp_key = os.getenv("AlpKey")
    alp_secret = os.getenv("AlpSecret")

    if not alp_key or not alp_secret:
        return "AlpKey/AlpSecret are not set."

    try:
        trading_client = TradingClient(alp_key, alp_secret)
        account = trading_client.get_account()
        if account.trading_blocked:
            return "Account is currently restricted from trading."
        return f"${account.buying_power} is available as buying power."
    except Exception as exc:
        return f"Failed to fetch account balance: {exc}"


@st.cache_resource
def build_agent():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    weather_tool = Tool(
        name="GetWeather",
        func=get_weather,
        description="Get weather information for a city. Input should be a city name.",
    )

    account_tool = Tool(
        name="GetAccountBalance",
        func=get_account_balance,
        description="Get Alpaca account buying power.",
    )

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0.3)

    agent = initialize_agent(
        tools=[weather_tool, account_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


st.title("🌤️ Weather and Trading Agent")
prompt = st.text_input("Enter your prompt")

if st.button("Get Result"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Running agent..."):
            try:
                agent = build_agent()
                response = agent.invoke({"input": prompt})
                output = response.get("output") if isinstance(response, dict) else str(response)
                st.write(output)
            except Exception as exc:
                st.error(f"Request failed: {exc}")