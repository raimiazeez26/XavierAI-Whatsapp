#https://www.twilio.com/blog/ai-chatbot-whatsapp-python-twilio-openai

#internal scripts
from datetime import datetime as dt
from MarketWatcher import MarketWatch
from def_symbols_tv import get_symbol_names
from tvDatafeed import Interval

market = MarketWatch()

timeframe_dict = {
    'monthly' : 'MN1',
    'weekly' : 'W1',
    'daily' : 'D1',
    '4 hour' : 'H4',
    '1 hour' : 'H1',
    '15 minutes' : 'M15',
    '5 minutes' : 'M5',
}

timeframe_tv = {
    'MN1' : Interval.in_monthly,
    'W1' : Interval.in_weekly,
    'D1' : Interval.in_daily,
    'H4' : Interval.in_4_hour,
    'H1' : Interval.in_1_hour,
    'M15' : Interval.in_15_minute,
    'M5' : Interval.in_5_minute,
}

symbols = get_symbol_names()
ff_fundamental = market.ff_fundamentals(deploy = False)
#week_fundamental = market.ff_fundamentals(url = 'https://www.forexfactory.com/calendar?week=this', deploy = False)
curr_strength = market.strength_meter()
gen_fundamental = market.general_forexnews()
#top = market.tops()
#buyers = top[0]
#sellers = top[1]

#current date and time
now = dt.now()
current_dt = now.strftime("%d/%m/%Y %H:%M:%S")

# Third-party imports
import openai
from fastapi import FastAPI, Form, Depends
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Internal imports
from models import Conversation, SessionLocal
from utils import send_message, logger


app = FastAPI()
# Set up the OpenAI API client
openai.api_key = config("OPENAI_API_KEY")
whatsapp_number = config("TO_NUMBER")

# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

@app.post("/message")
async def reply(Body: str = Form(), db: Session = Depends(get_db)):
    # Call the OpenAI API to generate text with GPT-3.5

    req = Body
    messages = [{"role": "system", "content":
              f"You are a intelligent Financial Adviser named XavierAI at Xavier Mcallister who \
              only answers questions related to finance.\
              You have the following information to provide assistance:\
              Current Date and time: {current_dt},\
              Currency strenghts: {curr_strength},\
              Fundamental News Data Line up: {ff_fundamental}\
              Latest Forex Market News {gen_fundamental}\
              If you are asked to provide trade recommendations, advice that a trade request be made\
              using the #request keyword."}]

    if '#request' in req.lower():

        message = f'Given the request: {req}, and a python dictionary {timeframe_dict} which contains timeframe key - value pairs.\
                        Extract the forex ticker symbol in the request and corresponding timeframe value from the dictionary {timeframe_dict}.\
                        Note that the default timeframe value if not provided is D1\
                        Please respond with a python dictionary containing the symbol and timeframe value only in a single line\
                        without any other text'

        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.3,
            )
        details = chat.choices[0].message.content
        # print(f'first reply = {reply}')
        resp = eval(details)

        if resp['symbol'] in symbols:
            conf = market.screener(resp['symbol'], timeframe_tv[resp['timeframe']])
            signal = conf[0]  # Signal direction of data
            df = conf[1]  # dataframe of asset + indicators
            key_levels = market.fib_levels(df)  # key price levels
            key_levels = [round(x, 4) for x in key_levels]
            price = round(market.pull_data(resp['symbol'], Interval.in_1_hour).Close[-1], 2)  # current price of asset

            pair_fundamental = market.forexnews_pair(resp['symbol'])

            # print('making second request')
            message = f'The signal of {resp["symbol"]} generated from market data is {signal}, current price of {price} and\
                    key price levels of {key_levels}, the currency strenghts at the moment are {curr_strength}.\
                    Recommend trading direction/action and with multiple SL and two TP levels using the key_levels calculated from market data.\
                    Also, consider the fundamental data line up for the day {ff_fundamental} and latest fundamental news around the asset \
                    to be {pair_fundamental}. Explain only the fundamentals that may affect trade direction and how they may affect trade \
                    direction. And clearly inndicate SL and TP Levels. If direction is bullish, recommend SL and level from the key levels below \
                    and above the current price respectively, if direction is bearish, recommend SL and TL levels above and below the current price\
                    respectvely. \
                    RESPOND WITH NATURAL LANGUAGE AND DO NOT LIST OUT THE VARIABLE DETAILS PROVIDED TO YOU IN YOUR RESPONSE'  # what should I do?'

            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages,
                    max_tokens=500,
                    n=1,
                    stop=None,
                    temperature=0.3,
                )

            reply = chat.choices[0].message.content
            # print(f"ChatGPT: {reply}")
            messages.append({"role": "assistant", "content": reply})

        else:
            message = f'You have been asked for trade recommendations but the requested ticker symbol or time frame is not supported,\
                    explicitly tell the user to check the list of supported ticker symbols and try the request again'

            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages,
                    max_tokens=500,
                    n=1,
                    stop=None,
                    temperature=0.3,
                )

            reply = chat.choices[0].message.content
            # print(f"ChatGPT: {reply}")
            messages.append({"role": "assistant", "content": reply})

    elif '#tops' in req.lower():
        top = market.tops()
        buyers = top[0]
        sellers = top[1]

        message = f'You have the following information to respond to requests\
                      Top buying pairs in the FOREX market: {buyers},\
                      Top selling pairs in the FOREX market: {sellers},\
                      Here is your request: {req}'

        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.3,
            )
        reply = chat.choices[0].message.content
        # print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

    else:
        message = req

        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.3,
            )
        reply = chat.choices[0].message.content
        # print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

    chat_response = reply

    # Store the conversation in the database
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=chat_response
            )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in database: {e}")
    send_message(whatsapp_number, chat_response)
    return ""
