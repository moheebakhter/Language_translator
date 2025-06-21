from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(gemini_api_key)

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
translator = Agent (
    name = "Translator Agent",
    instructions="Your a translate agent. Translate urdu into english."
)

response = Runner.run_sync(
    translator,
input = """
اجینٹک اے آئی ایک ایسا مصنوعی ذہانت کا نظام ہوتا ہے جو صرف معلومات کا تجزیہ کرنے یا جواب دینے تک محدود نہیں ہوتا،
بلکہ خود سے فیصلے لے سکتا ہے، مسائل کو پہچان سکتا ہے، اور مختلف کاموں کو خودکار طریقے سے مکمل کر سکتا ہے۔
یہ ایجنٹس مخصوص مقصد کے لیے بنائے جاتے ہیں، جیسے ترجمہ کرنا، ای میلز کا جواب دینا، یا کسی پروجیکٹ پر کام کرنا۔
اجینٹک اے آئی انسانی رویے کی نقل کرتے ہوئے خود مختار طریقے سے عمل کرتا ہے، جس سے یہ زیادہ ہوشیار، لچکدار اور عملی بن جاتا ہے۔
""",

    run_config = config 
)
print(response.final_output)