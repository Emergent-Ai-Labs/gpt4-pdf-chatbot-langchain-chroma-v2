import fs from 'fs';
import path from 'path';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { v4 as uuidv4 } from 'uuid';
import { CHROMA_URL, COLLECTION_NAME } from '@/config/chroma';


const filePath = 'docs';
export const run = async () => {
  try {
    const files = fs.readdirSync(filePath);
    const pdfFiles = files.filter(file => path.extname(file).toLowerCase() === '.pdf');

    const rawDocs = [];
    for (const file of pdfFiles) {
      const fullPath = path.join(filePath, file);
      const loader = new PDFLoader(fullPath);
      const docs = await loader.load();
      rawDocs.push(...docs);
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    docs.forEach(doc => {
      doc.metadata.id = uuidv4();
    });

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    // Sanitize metadata (flatten it)
    for (const doc of docs) {
      if (doc.metadata && typeof doc.metadata !== 'object') {
        doc.metadata = {};
      } else {
        // optional: remove nested metadata
        for (const key in doc.metadata) {
          if (typeof doc.metadata[key] === 'object') {
            delete doc.metadata[key]; // or stringify if needed
          }
        }
      }
    }

    console.log("Sample split doc:", docs[0]);
    const chroma = await Chroma.fromDocuments(docs, embeddings, {
      collectionName: COLLECTION_NAME,
      url: CHROMA_URL,
    });

    console.log('Ingestion complete');
  } catch (error) {
    console.error('Error during ingestion:', error);
  }
};

run();