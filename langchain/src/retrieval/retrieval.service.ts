import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";

import { z } from "zod";
import * as fs from "fs";
import * as uuid from "uuid";

import {
  CharacterTextSplitter,
  RecursiveCharacterTextSplitter,
} from "langchain/text_splitter";
import { pull } from "langchain/hub";
import { Document } from "langchain/document";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { FaissStore } from "langchain/vectorstores/faiss";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { InMemoryStore } from "langchain/storage/in_memory";
import { RunnableSequence } from "langchain/schema/runnable";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { BaseOutputParser } from "langchain/schema/output_parser";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { InMemoryDocstore } from "langchain/stores/doc/in_memory";
import { AttributeInfo } from "langchain/schema/query_constructor";
import { StringOutputParser } from "langchain/schema/output_parser";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import { JsonKeyOutputFunctionsParser } from "langchain/output_parsers";
import { MultiVectorRetriever } from "langchain/retrievers/multi_vector";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { ParentDocumentRetriever } from "langchain/retrievers/parent_document";
import { LLMChain, RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { FunctionalTranslator } from "langchain/retrievers/self_query/functional";
import { TimeWeightedVectorStoreRetriever } from "langchain/retrievers/time_weighted";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { createMetadataTaggerFromZod } from "langchain/document_transformers/openai_functions";

@Injectable()
export class RetrievalService {
  constructor(private readonly configService: ConfigService) {}

  async documentLoaders() {
    const loader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.txt`
    );
    return await loader.load();
  }

  async creatingDocument() {
    interface Document {
      pageContent: string;
      metadata: Record<string, any>;
    }

    const doc: Document = new Document({
      pageContent: "foo",
      metadata: { source: "baz" },
    });

    return JSON.stringify(doc);
  }

  async csvAllColumns() {
    const loader = new CSVLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.csv`
    );
    return await loader.load();
  }

  async csvSingleColumn() {
    const loader = new CSVLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.csv`,
      "text"
    );
    return await loader.load();
  }

  async jsonNoPointer() {
    const loader = new JSONLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.json`
    );
    return await loader.load();
  }

  async jsonPointer() {
    const loader = new JSONLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.json`,
      ["/from", "/surname"]
    );
    return await loader.load();
  }

  async fileDirectory() {
    const loader = new DirectoryLoader(
      `${__dirname}/../../src/retrieval/knowledge-base`,
      {
        ".json": (path) => new JSONLoader(path),
        ".txt": (path) => new TextLoader(path),
        ".csv": (path) => new CSVLoader(path),
      }
    );
    return await loader.load();
  }

  async textSplitterRawText() {
    const text = `Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.\n\n
Bye!\n\n-H.`;
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10,
      chunkOverlap: 1,
    });

    const output = await splitter.createDocuments([text]);

    return output;
  }

  async textSplitterDocument() {
    const text = `Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.\n\n
Bye!\n\n-H.`;
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10,
      chunkOverlap: 1,
    });

    const docOutput = await splitter.splitDocuments([
      new Document({ pageContent: text }),
    ]);

    return docOutput;
  }

  async textSplitterCustomized() {
    const text = `Some other considerations include:

  - Do you deploy your backend and frontend together, or separately?
  - Do you deploy your backend co-located with your database, or separately?
  
  **Production Support:** As you move your LangChains into production, we'd love to offer more hands-on support.
  Fill out [this form](https://airtable.com/appwQzlErAS2qiP0L/shrGtGaVBVAz7NcV2) to share more about what you're building, and our team will get in touch.
  
  ## Deployment Options
  
  See below for a list of deployment options for your LangChain app. If you don't see your preferred option, please get in touch and we can add it to this list.`;

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 50,
      chunkOverlap: 1,
      separators: ["|", "##", ">", "-"],
    });

    const docOutput = await splitter.splitDocuments([
      new Document({ pageContent: text }),
    ]);

    return docOutput;
  }

  async metadataTagger() {
    const zodSchema = z.object({
      movie_title: z.string(),
      critic: z.string(),
      tone: z.enum(["positive", "negative"]),
      rating: z
        .optional(z.number())
        .describe("The number of stars the critic rated the movie"),
    });

    const metadataTagger = createMetadataTaggerFromZod(zodSchema, {
      llm: new ChatOpenAI({ modelName: "gpt-4" }),
    });

    const documents = [
      new Document({
        pageContent:
          "Review of The Bee Movie\nBy Roger Ebert\nThis is the greatest movie ever made. 4 out of 5 stars.",
      }),
      new Document({
        pageContent:
          "Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.",
        metadata: { reliable: false },
      }),
    ];
    const taggedDocuments = await metadataTagger.transformDocuments(documents);

    return taggedDocuments;
  }

  async metadataTaggerCustomized() {
    const taggingChainTemplate = `Extract the desired information from the following passage.
                                  Anonymous critics are actually Roger Ebert.

                                  Passage:
                                  {input}
                                  `;

    const zodSchema = z.object({
      movie_title: z.string(),
      critic: z.string(),
      tone: z.enum(["positive", "negative"]),
      rating: z
        .optional(z.number())
        .describe("The number of stars the critic rated the movie"),
    });

    const metadataTagger = createMetadataTaggerFromZod(zodSchema, {
      llm: new ChatOpenAI({ modelName: "gpt-4" }),
      prompt: PromptTemplate.fromTemplate(taggingChainTemplate),
    });

    const documents = [
      new Document({
        pageContent:
          "Review of The Bee Movie\nBy Roger Ebert\nThis is the greatest movie ever made. 4 out of 5 stars.",
      }),
      new Document({
        pageContent:
          "Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.",
        metadata: { reliable: false },
      }),
    ];
    const taggedDocuments = await metadataTagger.transformDocuments(documents);

    return taggedDocuments;
  }

  async codeSplitJavascript() {
    const jsCode = `function helloWorld() {
                      console.log("Hello, World!");
                    }
                    // Call the function
                    helloWorld();`;

    const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
      chunkSize: 32,
      chunkOverlap: 0,
    });
    const jsOutput = await splitter.createDocuments([jsCode]);
    return jsOutput;
  }

  async contextualChunkHeaders() {
    const splitter = new CharacterTextSplitter({
      chunkSize: 1536,
      chunkOverlap: 200,
    });

    const jimDocs = await splitter.createDocuments(
      [`My favorite color is blue.`],
      [],
      {
        chunkHeader: `DOCUMENT NAME: Jim Interview\n\n---\n\n`,
        appendChunkOverlapHeader: true,
      }
    );

    const pamDocs = await splitter.createDocuments(
      [`My favorite color is red.`],
      [],
      {
        chunkHeader: `DOCUMENT NAME: Pam Interview\n\n---\n\n`,
        appendChunkOverlapHeader: true,
      }
    );

    const vectorStore = await HNSWLib.fromDocuments(
      jimDocs.concat(pamDocs),
      new OpenAIEmbeddings()
    );

    const model = new OpenAI({ temperature: 0 });

    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(model),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
    });

    const res = await chain.call({
      query: "What is Pam's favorite color?",
    });

    return res;
  }

  async basicExample() {
    const embeddings = new OpenAIEmbeddings();
    const res = await embeddings.embedQuery("Hello world");
    const documentRes = await embeddings.embedDocuments([
      "Hello world",
      "Bye bye",
    ]);

    return { res, documentRes };
  }

  async newIndexFromText() {
    const vectorStore = await MemoryVectorStore.fromTexts(
      ["Hello world", "Bye bye", "hello nice world"],
      [{ id: 2 }, { id: 1 }, { id: 3 }],
      new OpenAIEmbeddings()
    );

    return await vectorStore.similaritySearch("hello world", 1);
  }

  async newIndexFromLoader() {
    const loader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/example.txt`
    );
    const docs = await loader.load();

    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    return await vectorStore.similaritySearch(
      "document loaders expose methods",
      1
    );
  }

  async retrieversBasicExample() {
    const model = new OpenAI({});
    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);

    // Create a vector store from the documents
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    // Initialize a retriever wrapper around the vector store
    const vectorStoreRetriever = vectorStore.asRetriever();

    // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);

    return await chain.call({
      query: "What did the president say about Justice Breyer?",
    });
  }

  async contextualCompression() {
    const model = new OpenAI();
    const baseCompressor = LLMChainExtractor.fromLLM(model);

    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);

    // Create a vector store from the documents.
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    const retriever = new ContextualCompressionRetriever({
      baseCompressor,
      baseRetriever: vectorStore.asRetriever(),
    });

    const chain = RetrievalQAChain.fromLLM(model, retriever);

    return await chain.call({
      query: "What did the speaker say about Justice Breyer?",
    });
  }

  async multiqueryRetriever() {
    const vectorstore = await MemoryVectorStore.fromTexts(
      [
        "Buildings are made out of brick",
        "Buildings are made out of wood",
        "Buildings are made out of stone",
        "Cars are made out of metal",
        "Cars are made out of plastic",
        "mitochondria is the powerhouse of the cell",
        "mitochondria is made of lipids",
      ],
      [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
      new OpenAIEmbeddings()
    );
    const model = new ChatOpenAI({});
    const retriever = MultiQueryRetriever.fromLLM({
      llm: model,
      retriever: vectorstore.asRetriever(),
      verbose: true,
    });

    const query = "What are mitochondria made of?";
    return await retriever.getRelevantDocuments(query);
  }

  async multiqueryRetrieverCustomized() {
    type LineList = {
      lines: string[];
    };

    class LineListOutputParser extends BaseOutputParser<LineList> {
      static lc_name() {
        return "LineListOutputParser";
      }

      lc_namespace = ["langchain", "retrievers", "multiquery"];

      async parse(text: string): Promise<LineList> {
        const startKeyIndex = text.indexOf("<questions>");
        const endKeyIndex = text.indexOf("</questions>");
        const questionsStartIndex =
          startKeyIndex === -1 ? 0 : startKeyIndex + "<questions>".length;
        const questionsEndIndex =
          endKeyIndex === -1 ? text.length : endKeyIndex;
        const lines = text
          .slice(questionsStartIndex, questionsEndIndex)
          .trim()
          .split("\n")
          .filter((line) => line.trim() !== "");
        return { lines };
      }

      getFormatInstructions(): string {
        throw new Error("Not implemented.");
      }
    }

    // Default prompt is available at: https://smith.langchain.com/hub/jacob/multi-vector-retriever
    const prompt: PromptTemplate = await pull(
      "jacob/multi-vector-retriever-german"
    );

    const vectorstore = await MemoryVectorStore.fromTexts(
      [
        "Gebäude werden aus Ziegelsteinen hergestellt",
        "Gebäude werden aus Holz hergestellt",
        "Gebäude werden aus Stein hergestellt",
        "Autos werden aus Metall hergestellt",
        "Autos werden aus Kunststoff hergestellt",
        "Mitochondrien sind die Energiekraftwerke der Zelle",
        "Mitochondrien bestehen aus Lipiden",
      ],
      [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
      new OpenAIEmbeddings()
    );
    const model = new ChatOpenAI({});
    const llmChain = new LLMChain({
      llm: model,
      prompt,
      outputParser: new LineListOutputParser(),
    });
    const retriever = new MultiQueryRetriever({
      retriever: vectorstore.asRetriever(),
      llmChain,
      verbose: true,
    });

    const query = "What are mitochondria made of?";
    return await retriever.getRelevantDocuments(query);
  }

  async multivectorRetrieverSmallerChunks() {
    const textLoader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const parentDocuments = await textLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 20,
    });

    const docs = await splitter.splitDocuments(parentDocuments);

    const idKey = "doc_id";
    const docIds = docs.map((_) => uuid.v4());

    const childSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 400,
      chunkOverlap: 0,
    });

    const subDocs = [];
    for (let i = 0; i < docs.length; i += 1) {
      const childDocs = await childSplitter.splitDocuments([docs[i]]);
      const taggedChildDocs = childDocs.map((childDoc) => {
        childDoc.metadata[idKey] = docIds[i];
        return childDoc;
      });
      subDocs.push(...taggedChildDocs);
    }

    const keyValuePairs: [string, Document][] = docs.map((doc, i) => [
      docIds[i],
      doc,
    ]);

    // The docstore to use to store the original chunks
    const docstore = new InMemoryStore();
    await docstore.mset(keyValuePairs);

    // The vectorstore to use to index the child chunks
    const vectorstore = await FaissStore.fromDocuments(
      subDocs,
      new OpenAIEmbeddings()
    );

    const retriever = new MultiVectorRetriever({
      vectorstore,
      docstore,
      idKey,
      // Optional `k` parameter to search for more child documents in VectorStore.
      // Note that this does not exactly correspond to the number of final (parent) documents
      // retrieved, as multiple child documents can point to the same parent.
      childK: 20,
      // Optional `k` parameter to limit number of final, parent documents returned from this
      // retriever and sent to LLM. This is an upper-bound, and the final count may be lower than this.
      parentK: 5,
    });

    // Vectorstore alone retrieves the small chunks
    const vectorstoreResult =
      await retriever.vectorstore.similaritySearch("justice breyer");

    // Retriever returns larger result
    const retrieverResult =
      await retriever.getRelevantDocuments("justice breyer");

    return { vectorstoreResult, retrieverResult };
  }

  async multivectorRetrieverSummary() {
    const textLoader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const parentDocuments = await textLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 20,
    });

    const docs = await splitter.splitDocuments(parentDocuments);

    const chain = RunnableSequence.from([
      { content: (doc: Document) => doc.pageContent },
      PromptTemplate.fromTemplate(
        `Summarize the following document:\n\n{content}`
      ),
      new ChatOpenAI({
        maxRetries: 0,
      }),
      new StringOutputParser(),
    ]);

    const summaries = await chain.batch(
      docs,
      {},
      {
        maxConcurrency: 5,
      }
    );

    const idKey = "doc_id";
    const docIds = docs.map((_) => uuid.v4());
    const summaryDocs = summaries.map((summary, i) => {
      const summaryDoc = new Document({
        pageContent: summary,
        metadata: {
          [idKey]: docIds[i],
        },
      });
      return summaryDoc;
    });

    const keyValuePairs: [string, Document][] = docs.map((originalDoc, i) => [
      docIds[i],
      originalDoc,
    ]);

    // The docstore to use to store the original chunks
    const docstore = new InMemoryStore();
    await docstore.mset(keyValuePairs);

    // The vectorstore to use to index the child chunks
    const vectorstore = await FaissStore.fromDocuments(
      summaryDocs,
      new OpenAIEmbeddings()
    );

    const retriever = new MultiVectorRetriever({
      vectorstore,
      docstore,
      idKey,
    });

    // We could also add the original chunks to the vectorstore if we wish
    // const taggedOriginalDocs = docs.map((doc, i) => {
    //   doc.metadata[idKey] = docIds[i];
    //   return doc;
    // });
    // retriever.vectorstore.addDocuments(taggedOriginalDocs);

    // Vectorstore alone retrieves the small chunks
    const vectorstoreResult =
      await retriever.vectorstore.similaritySearch("justice breyer");

    // Retriever returns larger result
    const retrieverResult =
      await retriever.getRelevantDocuments("justice breyer");

    return { vectorstoreResult, retrieverResult };
  }

  async multivectorRetrieverHypotheticalQueries() {
    const textLoader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const parentDocuments = await textLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 20,
    });
    const docs = await splitter.splitDocuments(parentDocuments);

    const functionsSchema = [
      {
        name: "hypothetical_questions",
        description: "Generate hypothetical questions",
        parameters: {
          type: "object",
          properties: {
            questions: {
              type: "array",
              items: {
                type: "string",
              },
            },
          },
          required: ["questions"],
        },
      },
    ];

    const functionCallingModel = new ChatOpenAI({
      maxRetries: 0,
      modelName: "gpt-4",
    }).bind({
      functions: functionsSchema,
      function_call: { name: "hypothetical_questions" },
    });

    const chain = RunnableSequence.from([
      { content: (doc: Document) => doc.pageContent },
      PromptTemplate.fromTemplate(
        `Generate a list of 3 hypothetical questions that the below document could be used to answer:\n\n{content}`
      ),
      functionCallingModel,
      new JsonKeyOutputFunctionsParser<string[]>({ attrName: "questions" }),
    ]);

    const hypotheticalQuestions = await chain.batch(
      docs,
      {},
      {
        maxConcurrency: 5,
      }
    );

    const idKey = "doc_id";
    const docIds = docs.map((_) => uuid.v4());
    const hypotheticalQuestionDocs = hypotheticalQuestions
      .map((questionArray, i) => {
        const questionDocuments = questionArray.map((question) => {
          const questionDocument = new Document({
            pageContent: question,
            metadata: {
              [idKey]: docIds[i],
            },
          });
          return questionDocument;
        });
        return questionDocuments;
      })
      .flat();

    const keyValuePairs: [string, Document][] = docs.map((originalDoc, i) => [
      docIds[i],
      originalDoc,
    ]);

    // The docstore to use to store the original chunks
    const docstore = new InMemoryStore();
    await docstore.mset(keyValuePairs);

    // The vectorstore to use to index the child chunks
    const vectorstore = await FaissStore.fromDocuments(
      hypotheticalQuestionDocs,
      new OpenAIEmbeddings()
    );

    const retriever = new MultiVectorRetriever({
      vectorstore,
      docstore,
      idKey,
    });

    // We could also add the original chunks to the vectorstore if we wish
    // const taggedOriginalDocs = docs.map((doc, i) => {
    //   doc.metadata[idKey] = docIds[i];
    //   return doc;
    // });
    // retriever.vectorstore.addDocuments(taggedOriginalDocs);

    // Vectorstore alone retrieves the small chunks
    const vectorstoreResult =
      await retriever.vectorstore.similaritySearch("justice breyer");

    // Retriever returns larger result
    const retrieverResult =
      await retriever.getRelevantDocuments("justice breyer");

    return { vectorstoreResult, retrieverResult };
  }

  async parentDocumentRetriever() {
    const vectorstore = new MemoryVectorStore(new OpenAIEmbeddings());
    const docstore = new InMemoryDocstore();
    const retriever = new ParentDocumentRetriever({
      vectorstore,
      docstore,
      // Optional, not required if you're already passing in split documents
      parentSplitter: new RecursiveCharacterTextSplitter({
        chunkOverlap: 0,
        chunkSize: 500,
      }),
      childSplitter: new RecursiveCharacterTextSplitter({
        chunkOverlap: 0,
        chunkSize: 50,
      }),
      // Optional `k` parameter to search for more child documents in VectorStore.
      // Note that this does not exactly correspond to the number of final (parent) documents
      // retrieved, as multiple child documents can point to the same parent.
      childK: 20,
      // Optional `k` parameter to limit number of final, parent documents returned from this
      // retriever and sent to LLM. This is an upper-bound, and the final count may be lower than this.
      parentK: 5,
    });
    const textLoader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const parentDocuments = await textLoader.load();

    // We must add the parent documents via the retriever's addDocuments method
    await retriever.addDocuments(parentDocuments);

    // Retrieved chunks are the larger parent chunks
    return await retriever.getRelevantDocuments("justice breyer");
  }

  async selfQuerying() {
    /**
     * First, we create a bunch of documents. You can load your own documents here instead.
     * Each document has a pageContent and a metadata field. Make sure your metadata matches the AttributeInfo below.
     */
    const docs = [
      new Document({
        pageContent:
          "A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata: { year: 1993, rating: 7.7, genre: "science fiction" },
      }),
      new Document({
        pageContent:
          "Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata: { year: 2010, director: "Christopher Nolan", rating: 8.2 },
      }),
      new Document({
        pageContent:
          "A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata: { year: 2006, director: "Satoshi Kon", rating: 8.6 },
      }),
      new Document({
        pageContent:
          "A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata: { year: 2019, director: "Greta Gerwig", rating: 8.3 },
      }),
      new Document({
        pageContent: "Toys come alive and have a blast doing so",
        metadata: { year: 1995, genre: "animated" },
      }),
      new Document({
        pageContent:
          "Three men walk into the Zone, three men walk out of the Zone",
        metadata: {
          year: 1979,
          director: "Andrei Tarkovsky",
          genre: "science fiction",
          rating: 9.9,
        },
      }),
    ];

    /**
     * Next, we define the attributes we want to be able to query on.
     * in this case, we want to be able to query on the genre, year, director, rating, and length of the movie.
     * We also provide a description of each attribute and the type of the attribute.
     * This is used to generate the query prompts.
     */
    const attributeInfo: AttributeInfo[] = [
      {
        name: "genre",
        description: "The genre of the movie",
        type: "string or array of strings",
      },
      {
        name: "year",
        description: "The year the movie was released",
        type: "number",
      },
      {
        name: "director",
        description: "The director of the movie",
        type: "string",
      },
      {
        name: "rating",
        description: "The rating of the movie (1-10)",
        type: "number",
      },
      {
        name: "length",
        description: "The length of the movie in minutes",
        type: "number",
      },
    ];

    /**
     * Next, we instantiate a vector store. This is where we store the embeddings of the documents.
     * We also need to provide an embeddings object. This is used to embed the documents.
     */
    const embeddings = new OpenAIEmbeddings();
    const llm = new OpenAI();
    const documentContents = "Brief summary of a movie";
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
    const selfQueryRetriever = SelfQueryRetriever.fromLLM({
      llm,
      vectorStore,
      documentContents,
      attributeInfo,
      /**
       * We need to use a translator that translates the queries into a
       * filter format that the vector store can understand. We provide a basic translator
       * translator here, but you can create your own translator by extending BaseTranslator
       * abstract class. Note that the vector store needs to support filtering on the metadata
       * attributes you want to query on.
       */
      structuredQueryTranslator: new FunctionalTranslator(),
      // You can also pass in a default filter when initializing the self-query
      // retriever that will be used in combination with or as a fallback to
      // the generated query. For example, if you wanted to ensure that your
      // query documents tagged as genre: "animated"
      searchParams: {
        filter: (doc: Document) =>
          doc.metadata && doc.metadata.genre === "animated",
        mergeFiltersOperator: "and",
      },
    });

    /**
     * Now we can query the vector store.
     * We can ask questions like "Which movies are less than 90 minutes?" or "Which movies are rated higher than 8.5?".
     * We can also ask questions like "Which movies are either comedy or drama and are less than 90 minutes?".
     * The retriever will automatically convert these questions into queries that can be used to retrieve documents.
     */
    const query1 = await selfQueryRetriever.getRelevantDocuments(
      "Which movies are less than 90 minutes?"
    );
    const query2 = await selfQueryRetriever.getRelevantDocuments(
      "Which movies are rated higher than 8.5?"
    );
    const query3 = await selfQueryRetriever.getRelevantDocuments(
      "Which movies are directed by Greta Gerwig?"
    );
    const query4 = await selfQueryRetriever.getRelevantDocuments(
      "Which movies are either comedy or drama and are less than 90 minutes?"
    );

    return { query1, query2, query3, query4 };
  }

  async similarityScoreThreshold() {
    const vectorStore = await MemoryVectorStore.fromTexts(
      [
        "Buildings are made out of brick",
        "Buildings are made out of wood",
        "Buildings are made out of stone",
        "Buildings are made out of atoms",
        "Buildings are made out of building materials",
        "Cars are made out of metal",
        "Cars are made out of plastic",
      ],
      [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
      new OpenAIEmbeddings()
    );

    const retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore, {
      minSimilarityScore: 0.9, // Finds results with at least this similarity score
      maxK: 100, // The maximum K value to use. Use it based to your chunk size to make sure you don't run out of tokens
      kIncrement: 2, // How much to increase K by each time. It'll fetch N results, then N + kIncrement, then N + kIncrement * 2, etc.
    });

    return await retriever.getRelevantDocuments(
      "What are buildings made out of?"
    );
  }

  async timeWeightedVectorStore() {
    const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());

    const retriever = new TimeWeightedVectorStoreRetriever({
      vectorStore,
      memoryStream: [],
      searchKwargs: 2,
    });

    const documents = [
      "My name is John.",
      "My name is Bob.",
      "My favourite food is pizza.",
      "My favourite food is pasta.",
      "My favourite food is sushi.",
    ].map((pageContent) => ({ pageContent, metadata: {} }));

    // All documents must be added using this method on the retriever (not the vector store!)
    // so that the correct access history metadata is populated
    await retriever.addDocuments(documents);

    const results1 = await retriever.getRelevantDocuments(
      "What is my favourite food?"
    );

    const results2 = await retriever.getRelevantDocuments(
      "What is my favourite food?"
    );

    return { results1, results2 };
  }
}
