import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";

import * as fs from "fs";

import {
  LLMChain,
  APIChain,
  SequentialChain,
  loadQAStuffChain,
  RetrievalQAChain,
  loadQARefineChain,
  ConversationChain,
  loadQAMapReduceChain,
  SimpleSequentialChain,
  ConversationalRetrievalQAChain,
} from "langchain/chains";
import { Document } from "langchain/document";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory } from "langchain/memory";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, PromptTemplate } from "langchain/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

@Injectable()
export class ChainsService {
  constructor(private readonly configService: ConfigService) {}

  async basicExample() {
    // We can construct an LLMChain from a PromptTemplate and an LLM.
    const llm = new OpenAI({ temperature: 0 });

    const prompt = PromptTemplate.fromTemplate(
      "What is a good name for a company that makes {product}?"
    );
    const prompt2 = PromptTemplate.fromTemplate(
      "What is a good name for {company} that makes {product}?"
    );

    const chain = new LLMChain({ llm, prompt });
    const chain2 = new LLMChain({ llm, prompt: prompt2 });

    // single-input, single-output LLMChains can use `run` method. It is a convenience method
    // that takes in a string and returns the value of the output key field in the response
    return {
      chain: await chain.run("colorful socks"),
      chain2: await chain.call({
        company: "a startup",
        product: "colorful socks",
      }),
    };
  }

  async basicExampleChatModel() {
    // We can also construct an LLMChain from a ChatPromptTemplate and a chat model.
    const chat = new ChatOpenAI({ temperature: 0 });
    const chatPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a helpful assistant that translates {input_language} to {output_language}.",
      ],
      ["human", "{text}"],
    ]);
    const chain = new LLMChain({
      prompt: chatPrompt,
      llm: chat,
    });

    return await chain.call({
      input_language: "English",
      output_language: "French",
      text: "I love programming.",
    });
  }

  async debuggingChains() {
    const chat = new ChatOpenAI({});

    // This chain automatically initializes and uses a `BufferMemory` instance
    // as well as a default prompt.
    const chain = new ConversationChain({ llm: chat, verbose: true });
    return await chain.call({ input: "What is ChatGPT?" });
  }

  async addStateMemory() {
    const chat = new ChatOpenAI({});

    const memory = new BufferMemory();

    // This particular chain automatically initializes a BufferMemory instance if none is provided,
    // but we pass it explicitly here. It also has a default prompt.
    const chain = new ConversationChain({ llm: chat, memory });

    const res1 = await chain.run(
      "Answer briefly. What are the first 3 colors of a rainbow?"
    );

    const res2 = await chain.run("And the next 4?");

    return { res1, res2 };
  }

  async useStreamCallback() {
    // Create a new LLMChain from a PromptTemplate and an LLM in streaming mode.
    const model = new OpenAI({ temperature: 0.9, streaming: true });
    const prompt = PromptTemplate.fromTemplate(
      "What is a good name for a company that makes {product}?"
    );
    const chain = new LLMChain({ llm: model, prompt });

    // Call the chain with the inputs and a callback for the streamed tokens
    const res = await chain.call({ product: "colorful socks" }, [
      {
        handleLLMNewToken(token: string) {
          process.stdout.write(token);
        },
      },
    ]);
    return res;
  }

  async abortRequest() {
    // Create a new LLMChain from a PromptTemplate and an LLM in streaming mode.
    const model = new OpenAI({ temperature: 0.9, streaming: true });
    const prompt = PromptTemplate.fromTemplate(
      "Give me a long paragraph about {product}?"
    );
    const chain = new LLMChain({ llm: model, prompt });
    const controller = new AbortController();

    // Call `controller.abort()` somewhere to cancel the request.
    setTimeout(() => {
      controller.abort();
    }, 100);

    try {
      // Call the chain with the inputs and a callback for the streamed tokens
      return await chain.call(
        { product: "colorful socks", signal: controller.signal },
        [
          {
            handleLLMNewToken(token: string) {
              process.stdout.write(token);
            },
          },
        ]
      );
    } catch (e) {
      return e.message;
    }
  }

  async simpleSequentialChain() {
    // This is an LLMChain to write a synopsis given a title of a play.
    const llm = new OpenAI({ temperature: 0 });
    const template = `You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
 
    Title: {title}
    Playwright: This is a synopsis for the above play:`;

    const promptTemplate = new PromptTemplate({
      template,
      inputVariables: ["title"],
    });
    const synopsisChain = new LLMChain({ llm, prompt: promptTemplate });

    // This is an LLMChain to write a review of a play given a synopsis.
    const reviewLLM = new OpenAI({ temperature: 0 });
    const reviewTemplate = `You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
 
    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:`;

    const reviewPromptTemplate = new PromptTemplate({
      template: reviewTemplate,
      inputVariables: ["synopsis"],
    });

    const reviewChain = new LLMChain({
      llm: reviewLLM,
      prompt: reviewPromptTemplate,
    });

    const overallChain = new SimpleSequentialChain({
      chains: [synopsisChain, reviewChain],
      verbose: true,
    });

    // variable review contains the generated play review based on the input title and
    //  synopsis generated in the first step:
    const review = await overallChain.run("Tragedy at sunset on the beach");
    return review;
  }

  async sequentialChain() {
    // This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
    const llm = new OpenAI({ temperature: 0 });
    const template = `You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:`;

    const promptTemplate = new PromptTemplate({
      template,
      inputVariables: ["title", "era"],
    });
    const synopsisChain = new LLMChain({
      llm,
      prompt: promptTemplate,
      outputKey: "synopsis",
    });

    // This is an LLMChain to write a review of a play given a synopsis.
    const reviewLLM = new OpenAI({ temperature: 0 });
    const reviewTemplate = `You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
  
    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:`;

    const reviewPromptTemplate = new PromptTemplate({
      template: reviewTemplate,
      inputVariables: ["synopsis"],
    });
    const reviewChain = new LLMChain({
      llm: reviewLLM,
      prompt: reviewPromptTemplate,
      outputKey: "review",
    });

    const overallChain = new SequentialChain({
      chains: [synopsisChain, reviewChain],
      inputVariables: ["era", "title"],
      // Here we return multiple variables
      outputVariables: ["synopsis", "review"],
      verbose: true,
    });

    // contains final review and intermediate synopsis (as specified by outputVariables)
    // The data is generated based on the input title and era:
    const chainExecutionResult = await overallChain.call({
      title: "Tragedy at sunset on the beach",
      era: "Victorian England",
    });

    return chainExecutionResult;
  }

  async documentsStuff() {
    const llmA = new OpenAI({});
    const chainA = loadQAStuffChain(llmA);
    const docs = [
      new Document({ pageContent: "Harrison went to Harvard." }),
      new Document({ pageContent: "Ankush went to Princeton." }),
    ];
    const resA = await chainA.call({
      input_documents: docs,
      question: "Where did Harrison go to college?",
    });

    return resA;
  }

  async documentsRefine() {
    // Create the models and chain
    const embeddings = new OpenAIEmbeddings();
    const model = new OpenAI({ temperature: 0 });
    const chain = loadQARefineChain(model);

    // Load the documents and create the vector store
    const loader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const docs = await loader.loadAndSplit();
    const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

    // Select the relevant documents
    const question = "What did the president say about Justice Breyer";
    const relevantDocs = await store.similaritySearch(question);

    // Call the chain
    return await chain.call({
      input_documents: relevantDocs,
      question,
    });
  }

  async documentsRefineCustomized() {
    const questionPromptTemplateString = `Context information is below.
      ---------------------
      {context}
      ---------------------
      Given the context information and no prior knowledge, answer the question: {question}`;
    const questionPrompt = new PromptTemplate({
      inputVariables: ["context", "question"],
      template: questionPromptTemplateString,
    });

    const refinePromptTemplateString = `The original question is as follows:{question}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer
    (only if needed) with some more context below.
    ------------
    {context}
    ------------
    Given the new context, refine the original answer to better answer the question.
    You must provide a response, either original answer or refined answer.`;
    const refinePrompt = new PromptTemplate({
      inputVariables: ["question", "existing_answer", "context"],
      template: refinePromptTemplateString,
    });

    // Create the models and chain
    const embeddings = new OpenAIEmbeddings();
    const model = new OpenAI({ temperature: 0 });
    const chain = loadQARefineChain(model, {
      questionPrompt,
      refinePrompt,
    });

    // Load the documents and create the vector store
    const loader = new TextLoader(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`
    );
    const docs = await loader.loadAndSplit();
    const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

    // Select the relevant documents
    const question = "What did the president say about Justice Breyer";
    const relevantDocs = await store.similaritySearch(question);

    // Call the chain
    const res = await chain.call({
      input_documents: relevantDocs,
      question,
    });

    return res;
  }

  async mapReduce() {
    // Optionally limit the number of concurrent requests to the language model.
    const model = new OpenAI({ temperature: 0, maxConcurrency: 10 });
    const chain = loadQAMapReduceChain(model);
    const docs = [
      new Document({ pageContent: "harrison went to harvard" }),
      new Document({ pageContent: "ankush went to princeton" }),
    ];

    return await chain.call({
      input_documents: docs,
      question: "Where did harrison go to college",
    });
  }

  async apiChains() {
    const OPEN_METEO_DOCS = `BASE URL: https://api.open-meteo.com/

API Documentation
The API endpoint /v1/forecast accepts a geographical coordinate, a list of weather variables and responds with a JSON hourly weather forecast for 7 days. Time always starts at 0:00 today and contains 168 hours. All URL parameters are listed below:

Parameter	Format	Required	Default	Description
latitude, longitude	Floating point	Yes		Geographical WGS84 coordinate of the location
hourly	String array	No		A list of weather variables which should be returned. Values can be comma separated, or multiple &hourly= parameter in the URL can be used.
daily	String array	No		A list of daily weather variable aggregations which should be returned. Values can be comma separated, or multiple &daily= parameter in the URL can be used. If daily weather variables are specified, parameter timezone is required.
current_weather	Bool	No	false	Include current weather conditions in the JSON output.
temperature_unit	String	No	celsius	If fahrenheit is set, all temperature values are converted to Fahrenheit.
windspeed_unit	String	No	kmh	Other wind speed speed units: ms, mph and kn
precipitation_unit	String	No	mm	Other precipitation amount units: inch
timeformat	String	No	iso8601	If format unixtime is selected, all time values are returned in UNIX epoch time in seconds. Please note that all timestamp are in GMT+0! For daily values with unix timestamps, please apply utc_offset_seconds again to get the correct date.
timezone	String	No	GMT	If timezone is set, all timestamps are returned as local-time and data is returned starting at 00:00 local-time. Any time zone name from the time zone database is supported. If auto is set as a time zone, the coordinates will be automatically resolved to the local time zone.
past_days	Integer (0-2)	No	0	If past_days is set, yesterday or the day before yesterday data are also returned.
start_date
end_date	String (yyyy-mm-dd)	No		The time interval to get weather data. A day must be specified as an ISO8601 date (e.g. 2022-06-30).
models	String array	No	auto	Manually select one or more weather models. Per default, the best suitable weather models will be combined.

Variable	Valid time	Unit	Description
temperature_2m	Instant	°C (°F)	Air temperature at 2 meters above ground
snowfall	Preceding hour sum	cm (inch)	Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7. E.g. 7 cm snow = 10 mm precipitation water equivalent
rain	Preceding hour sum	mm (inch)	Rain from large scale weather systems of the preceding hour in millimeter
showers	Preceding hour sum	mm (inch)	Showers from convective precipitation in millimeters from the preceding hour
weathercode	Instant	WMO code	Weather condition as a numeric code. Follow WMO weather interpretation codes. See table below for details.
snow_depth	Instant	meters	Snow depth on the ground
freezinglevel_height	Instant	meters	Altitude above sea level of the 0°C level
visibility	Instant	meters	Viewing distance in meters. Influenced by low clouds, humidity and aerosols. Maximum visibility is approximately 24 km.`;

    const model = new OpenAI({ modelName: "text-davinci-003" });
    const chain = APIChain.fromLLMAndAPIDocs(model, OPEN_METEO_DOCS, {
      headers: {
        // These headers will be used for API requests made by the chain.
      },
    });

    return await chain.call({
      question:
        "What is the weather like right now in Munich, Germany in degrees Farenheit?",
    });
  }

  async retrievalQA() {
    // Initialize the LLM to use to answer the question.
    const model = new OpenAI({});
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

    // Initialize a retriever wrapper around the vector store
    const vectorStoreRetriever = vectorStore.asRetriever();

    // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
    return await chain.call({
      query: "What did the president say about Justice Breyer?",
    });
  }

  async retrievalQACustomChain() {
    // Initialize the LLM to use to answer the question.
    const model = new OpenAI({});
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

    // Create a chain that uses a map reduce chain and HNSWLib vector store.
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAMapReduceChain(model),
      retriever: vectorStore.asRetriever(),
    });
    return await chain.call({
      query: "What did the president say about Justice Breyer?",
    });
  }

  async retrievalQACustomPrompt() {
    const promptTemplate = `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}
    
    Question: {question}
    Answer in Italian:`;
    const prompt = PromptTemplate.fromTemplate(promptTemplate);

    // Initialize the LLM to use to answer the question.
    const model = new OpenAI({});
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

    // Create a chain that uses a stuff chain and HNSWLib vector store.
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(model, { prompt }),
      retriever: vectorStore.asRetriever(),
    });
    return await chain.call({
      query: "What did the president say about Justice Breyer?",
    });
  }

  async retrievalQAReturnSourceDocuments() {
    // Initialize the LLM to use to answer the question.
    const model = new OpenAI({});
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

    // Create a chain that uses a map reduce chain and HNSWLib vector store.
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
      returnSourceDocuments: true, // Can also be passed into the constructor
    });

    return await chain.call({
      query: "What did the president say about Justice Breyer?",
    });
    // console.log(JSON.stringify(res, null, 2));
  }

  async conversationalRetrievalQA() {
    /* Initialize the LLM to use to answer the question */
    const model = new ChatOpenAI({});
    /* Load in the file we want to do question answering over */
    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );

    /* Split the text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);

    /* Create the vectorstore */
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    /* Create the chain */
    const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStore.asRetriever(),
      {
        memory: new BufferMemory({
          memoryKey: "chat_history", // Must be set to "chat_history"
        }),
      }
    );

    /* Ask it a question */
    const question = "What did the president say about Justice Breyer?";
    const res = await chain.call({ question });

    /* Ask it a follow up question */
    const followUpRes = await chain.call({
      question: "Was that nice?",
    });

    return { res, followUpRes };
  }

  async conversationalRetrievalQABuiltinMemory() {
    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    const fasterModel = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
    });
    const slowerModel = new ChatOpenAI({
      modelName: "gpt-4",
    });

    const chain = ConversationalRetrievalQAChain.fromLLM(
      slowerModel,
      vectorStore.asRetriever(),
      {
        returnSourceDocuments: true,
        memory: new BufferMemory({
          memoryKey: "chat_history",
          inputKey: "question", // The key for the input to the chain
          outputKey: "text", // The key for the final conversational output of the chain
          returnMessages: true, // If using with a chat model (e.g. gpt-3.5 or gpt-4)
        }),
        questionGeneratorChainOptions: {
          llm: fasterModel,
        },
      }
    );

    /* Ask it a question */
    const question = "What did the president say about Justice Breyer?";
    const res = await chain.call({ question });

    const followUpRes = await chain.call({ question: "Was that nice?" });

    return { res, followUpRes };
  }

  async conversationalRetrievalStreaming() {
    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    let streamedResponse = "";
    const streamingModel = new ChatOpenAI({
      streaming: true,
      callbacks: [
        {
          handleLLMNewToken(token) {
            streamedResponse += token;
          },
        },
      ],
    });

    const nonStreamingModel = new ChatOpenAI({});

    const chain = ConversationalRetrievalQAChain.fromLLM(
      streamingModel,
      vectorStore.asRetriever(),
      {
        returnSourceDocuments: true,
        memory: new BufferMemory({
          memoryKey: "chat_history",
          inputKey: "question", // The key for the input to the chain
          outputKey: "text", // The key for the final conversational output of the chain
          returnMessages: true, // If using with a chat model
        }),
        questionGeneratorChainOptions: {
          llm: nonStreamingModel,
        },
      }
    );

    /* Ask it a question */
    return await chain.call({
      question: "What did the president say about Justice Breyer?",
    });
  }

  async conversationalRetrievalExternallyManagedMemory() {
    /* Initialize the LLM to use to answer the question */
    const model = new OpenAI({});

    /* Load in the file we want to do question answering over */
    const text = fs.readFileSync(
      `${__dirname}/../../src/retrieval/knowledge-base/state_of_the_union.txt`,
      "utf8"
    );

    /* Split the text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);

    /* Create the vectorstore */
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );

    /* Create the chain */
    const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStore.asRetriever()
    );

    /* Ask it a question */
    const question = "What did the president say about Justice Breyer?";

    /* Can be a string or an array of chat messages */
    const res = await chain.call({ question, chat_history: "" });

    /* Ask it a follow up question */
    const chatHistory = `${question}\n${res.text}`;
    const followUpRes = await chain.call({
      question: "Was that nice?",
      chat_history: chatHistory,
    });

    return { res, followUpRes };
  }

  async conversationalRetrievalPromptCustomization() {
    const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `Given the following conversation and a follow up question, return the conversation history excerpt that includes any relevant context to the question if it exists and rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Your answer should follow the following format:
    \`\`\`
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    <Relevant chat history excerpt as context here>
    Standalone question: <Rephrased question here>
    \`\`\`
    Your answer:`;

    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });

    const vectorStore = await HNSWLib.fromTexts(
      [
        "Mitochondria are the powerhouse of the cell",
        "Foo is red",
        "Bar is red",
        "Buildings are made out of brick",
        "Mitochondria are made of lipids",
      ],
      [{ id: 2 }, { id: 1 }, { id: 3 }, { id: 4 }, { id: 5 }],
      new OpenAIEmbeddings()
    );

    const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStore.asRetriever(),
      {
        memory: new BufferMemory({
          memoryKey: "chat_history",
          returnMessages: true,
        }),
        questionGeneratorChainOptions: {
          template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
        },
      }
    );

    const res = await chain.call({
      question:
        "I have a friend called Bob. He's 28 years old. He'd like to know what the powerhouse of the cell is?",
    });

    const res2 = await chain.call({
      question: "How old is Bob?",
    });

    return { res, res2 };
  }
}
