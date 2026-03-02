import os
import tempfile
import requests
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


load_dotenv()


def get_vwap_tool(symbol: str) -> str:
    """
    Calculates today's VWAP (Volume Weighted Average Price) for a stock using
    intraday minute bars. VWAP = Sum(TypicalPrice * Volume) / Sum(Volume).
    """
    symbol = symbol.strip().upper()
    try:
        alp_key = os.getenv("AlpKey")
        alp_secret = os.getenv("AlpSecret")
        if not alp_key or not alp_secret:
            return "AlpKey/AlpSecret are not set."

        data_client = StockHistoricalDataClient(alp_key, alp_secret)

        today = date.today()
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(today, datetime.min.time()),
        )
        bars_df = data_client.get_stock_bars(request_params).df

        if bars_df.empty:
            return f"No intraday data available for {symbol} today. Market may be closed."

        if hasattr(bars_df.index, "levels"):
            bars_df = bars_df.xs(symbol, level="symbol")

        bars_df["typical_price"] = (bars_df["high"] + bars_df["low"] + bars_df["close"]) / 3
        bars_df["pv"] = bars_df["typical_price"] * bars_df["volume"]

        cumulative_volume = bars_df["volume"].sum()
        if cumulative_volume == 0:
            return f"Volume for {symbol} is zero; cannot calculate VWAP."

        vwap = bars_df["pv"].sum() / cumulative_volume
        last_price = bars_df["close"].iloc[-1]
        relation = "above" if last_price > vwap else "below"

        return (
            f"VWAP for {symbol}: ${vwap:.2f}\n"
            f"Last price: ${last_price:.2f} (trading {relation} VWAP)"
        )

    except Exception as e:
        return f"Error calculating VWAP for {symbol}: {str(e)}"


def get_rsi_tool(symbol: str, window: int = 14) -> str:
    """
    Calculates the 14-period RSI for a given stock symbol.
    Identifies overbought (>70) or oversold (<30) conditions.
    """
    symbol = symbol.strip().upper()
    try:
        alp_key = os.getenv("AlpKey")
        alp_secret = os.getenv("AlpSecret")
        if not alp_key or not alp_secret:
            return "AlpKey/AlpSecret are not set."

        data_client = StockHistoricalDataClient(alp_key, alp_secret)

        start_date = datetime.now() - timedelta(days=window * 2)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
        )
        bars_df = data_client.get_stock_bars(request_params).df

        if bars_df.empty:
            return f"No data found for {symbol}."

        if hasattr(bars_df.index, "levels"):
            bars_df = bars_df.xs(symbol, level="symbol")

        close_prices = bars_df["close"]
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1]
        signal = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Neutral"
        return f"The current {window}-day RSI for {symbol} is {latest_rsi:.2f} [{signal}]."

    except Exception as e:
        return f"Error calculating RSI for {symbol}: {str(e)}"


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


def process_pdf(uploaded_file) -> tuple[FAISS, int]:
    """Load a PDF, split into chunks, embed and return a FAISS vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store, len(docs)


def answer_pdf_question(question: str) -> str:
    """Answer a question grounded in the uploaded PDF document via RAG."""
    vector_store = st.session_state.get("pdf_vector_store")
    if vector_store is None:
        return (
            "No PDF has been uploaded yet. "
            "Please upload a PDF using the sidebar uploader and try again."
        )
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    )
    return qa_chain.invoke({"query": question})["result"]


@st.cache_resource
def build_agent():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    weather_tool = Tool(
        name="GetWeather",
        func=get_weather,
        description=(
            "Use this ONLY for weather-related questions: current weather, temperature, "
            "conditions, forecast, or climate in a city. Input must be a city name (e.g. London, New York)."
        ),
    )

    account_tool = Tool(
        name="GetAccountBalance",
        func=get_account_balance,
        description=(
            "Use this ONLY for trading account questions: Alpaca account balance, buying power, "
            "account status, or how much you can trade with. Input is ignored."
        ),
    )

    rsi_tool = Tool(
        name="GetRSI",
        func=get_rsi_tool,
        description=(
            "Use this ONLY when the user asks about RSI, momentum, overbought or oversold signals "
            "for a specific stock. Input must be a stock ticker symbol (e.g. AAPL, TSLA, MSFT)."
        ),
    )

    vwap_tool = Tool(
        name="GetVWAP",
        func=get_vwap_tool,
        description=(
            "Use this ONLY when the user asks about VWAP (Volume Weighted Average Price), "
            "intraday price benchmark, or whether a stock is trading above/below VWAP. "
            "Input must be a stock ticker symbol (e.g. AAPL, TSLA, MSFT)."
        ),
    )

    pdf_tool = Tool(
        name="AnswerPDFQuestion",
        func=answer_pdf_question,
        description=(
            "Use this when the user asks any question about the content of the uploaded PDF document. "
            "Input must be the user's full question as a string."
        ),
    )

    tool_selection_prefix = """You have access to the following tools. Choose exactly one based on the user's intent:

- WEATHER: User asks about weather, temperature, forecast, or climate in a city -> use GetWeather with the city name.
- TRADING ACCOUNT: User asks about Alpaca account balance, buying power, or account status -> use GetAccountBalance (no input needed).
- RSI: User asks about RSI, momentum, overbought/oversold levels for a stock -> use GetRSI with the ticker symbol.
- VWAP: User asks about VWAP, intraday price average, or if a stock is above/below VWAP -> use GetVWAP with the ticker symbol.
- PDF DOCUMENT: User asks anything about the contents of the uploaded PDF document -> use AnswerPDFQuestion with the full question.

Do not mix tools. Use the ticker symbol exactly as given (e.g. AAPL, TSLA). Answer briefly and helpfully after using the correct tool.

"""

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0.3)

    agent = initialize_agent(
        tools=[weather_tool, account_tool, rsi_tool, vwap_tool, pdf_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": tool_selection_prefix},
    )
    return agent


st.title("🌤️ Weather, Trading & PDF Agent")

# --- Sidebar: PDF uploader ---
with st.sidebar:
    st.header("PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with", type="pdf")

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("pdf_file_id") != file_id:
            with st.spinner("Indexing PDF…"):
                try:
                    vector_store, chunk_count = process_pdf(uploaded_file)
                    st.session_state["pdf_vector_store"] = vector_store
                    st.session_state["pdf_file_id"] = file_id
                    st.session_state["pdf_name"] = uploaded_file.name
                    st.success(f"Indexed **{uploaded_file.name}** ({chunk_count} chunks)")
                except Exception as exc:
                    st.error(f"Failed to process PDF: {exc}")
        else:
            st.info(f"Active PDF: **{st.session_state.get('pdf_name')}**")

    if st.session_state.get("pdf_vector_store") and st.button("Clear PDF"):
        del st.session_state["pdf_vector_store"]
        del st.session_state["pdf_file_id"]
        del st.session_state["pdf_name"]
        st.rerun()

# --- Main chat ---
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
