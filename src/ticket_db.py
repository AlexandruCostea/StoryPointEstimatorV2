import os
import glob
import pandas as pd
import chromadb
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer



class TicketDB:
    def __init__(self,):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.csv_path = os.getenv("DATA_PATH")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("ticket_collection")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        self.setup_collection()

    
    def setup_collection(self):
        batch_size = 32
        all_files = glob.glob(os.path.join(self.csv_path, "*.csv"))
        ticket_records = []

        for file in all_files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                title = str(row['title'])
                description = str(row['description']) if pd.notna(row['description']) else ""
                combined_text = f"{title} {description}"
                ticket_records.append({
                    "issuekey": row['issuekey'],
                    "text": combined_text,
                    "storypoint": row['storypoint'],
                    "sourcefile": os.path.basename(file)
                })
        
        all_docs = [record['text'] for record in ticket_records]
        all_ids = [record['issuekey'] for record in ticket_records]
        all_metadatas = [
            {
                "issuekey": record['issuekey'],
                "storypoint": int(record['storypoint']),
                "sourcefile": record['sourcefile']
            } for record in ticket_records
        ]

        all_embeddings = self.embedder.encode(all_docs, batch_size=batch_size, show_progress_bar=True)

        batch_size = 5000

        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            batch_embeddings = [emb.tolist() for emb in all_embeddings[i:i + batch_size]]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            self.collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
    

    def get_similar_tickets(self, new_title, new_description, n_results=5):
        query_text = f"{new_title} {new_description}"
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        similar_tickets = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            similar_tickets.append({
                "title": doc,
                "issuekey": meta['issuekey'],
                "storypoint": meta['storypoint'],
                "sourcefile": meta['sourcefile']
            })
        return similar_tickets