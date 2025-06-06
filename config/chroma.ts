/**
 * Change the namespace to the namespace on Pinecone you'd like to store your embeddings.
 */

if (!process.env.COLLECTION_NAME) {
  throw new Error('Missing collection name in .env file');
}

export const CHROMA_URL = process.env.CHROMA_URL ?? 'http://localhost:8000';
export const COLLECTION_NAME = process.env.COLLECTION_NAME ?? 'default';