from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

sample_text = """ Tesla's Q3 Results

Tesla reported record revenue fo $25.2B in Q3 2024.

Model Y Performace

"""

splitter1 = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "
)
