import sqlite3

# Connect to database
conn = sqlite3.connect('./data/ragbot.db')
cursor = conn.cursor()

# Delete all old embeddings
cursor.execute("DELETE FROM embeddings")
print(f"✅ Deleted {cursor.rowcount} old embeddings")

# Reset the embedding flags on papers
cursor.execute("UPDATE papers SET embedding_created = 0")
print(f"✅ Reset {cursor.rowcount} papers for re-embedding")

conn.commit()
conn.close()
print("✅ Ready to create new OpenAI embeddings!")