import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { makeChain } from '@/utils/makechain';
import { CHROMA_URL, COLLECTION_NAME } from '@/config/chroma';
import { ChromaClient } from 'chromadb';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  console.log('question', question);

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    const client = new ChromaClient({path: CHROMA_URL});
    const collection = await client.getCollection({name: COLLECTION_NAME});
    const count = await collection.count();
    console.log(`[Chroma] Document count: ${count}`);
    
    console.log(COLLECTION_NAME);
    console.log(CHROMA_URL);
    console.log('creating vector store...');
    /* create vectorstore*/
    const vectorStore = new Chroma(new OpenAIEmbeddings(), {
      collectionName: COLLECTION_NAME,
      url: CHROMA_URL,
    });

    //create chain
    const chain = makeChain(vectorStore);
    //Ask a question using chat history
    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: history || [],
    });

    console.log('response', response);
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}
