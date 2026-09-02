from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai.types.responses import response

load_dotenv()

# Init LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

sample_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expetations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y becoame the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
SUpply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce consts."""

prompt = f"""
YOu are a text chunking expert. Split this text logical chunks.
Rules:
- Each chunk should be around 200 characters or less
- Split at natural topc boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{sample_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

print("∆ Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Split the text at the markers
chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

print("\n § AGENTIC CHUNKIG RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks,1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
