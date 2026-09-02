from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

sample_text = """ Tesla's Q3 Results

Tesla reported record revenue fo $25.2B in Q3 2024.

Model Y Performace

"""

print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", ", "]
    chunk_size=100,
    chunk_overlap=0,
)

chunk2 = recursive_splitter.split_text(sample_text)

print(f"Same problem text, but with RecursiveCharacterTextSplitter")
for i, chunk in enumerate(chunk2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
