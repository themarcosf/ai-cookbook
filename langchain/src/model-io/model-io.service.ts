import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";

import {
  PromptTemplate,
  ChatPromptTemplate,
  FewShotPromptTemplate,
  PipelinePromptTemplate,
  LengthBasedExampleSelector,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
  SemanticSimilarityExampleSelector,
} from "langchain/prompts";
import {
  RegexParser,
  OutputFixingParser,
  CombiningOutputParser,
  StructuredOutputParser,
  CustomListOutputParser,
  CommaSeparatedListOutputParser,
} from "langchain/output_parsers";
import {
  BytesOutputParser,
  StringOutputParser,
} from "langchain/schema/output_parser";
import { z } from "zod";
import { LLMChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { Serialized } from "langchain/load/serializable";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RunnableSequence } from "langchain/schema/runnable";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { LLMResult, HumanMessage, SystemMessage } from "langchain/schema";

@Injectable()
export class ModelIoService {
  llm = new OpenAI({
    openAIApiKey: this.configService.get("OPENAI_API_KEY"),
    temperature: 0.9,
  });

  constructor(private readonly configService: ConfigService) {}

  async simpleExample() {
    // If a template is passed in, the input variables are inferred automatically from the template
    const prompt = PromptTemplate.fromTemplate(
      `You are a naming consultant for new companies.
       What is a good name for a company that makes {product}?`
    );

    const formattedPrompt = await prompt.format({
      product: "colorful socks",
    });

    return formattedPrompt;
  }

  async createPromptTemplate1() {
    // An example prompt with no input variables
    const noInputPrompt = new PromptTemplate({
      inputVariables: [],
      template: "Tell me a joke.",
    });
    const formattedNoInputPrompt = await noInputPrompt.format({});

    // An example prompt with one input variable
    const oneInputPrompt = new PromptTemplate({
      inputVariables: ["adjective"],
      template: "Tell me a {adjective} joke.",
    });
    const formattedOneInputPrompt = await oneInputPrompt.format({
      adjective: "funny",
    });

    // An example prompt with multiple input variables
    const multipleInputPrompt = new PromptTemplate({
      inputVariables: ["adjective", "content"],
      template: "Tell me a {adjective} joke about {content}.",
    });
    const formattedMultipleInputPrompt = await multipleInputPrompt.format({
      adjective: "funny",
      content: "chickens",
    });

    return {
      formattedNoInputPrompt,
      formattedOneInputPrompt,
      formattedMultipleInputPrompt,
    };
  }

  async createPromptTemplate2() {
    const template = "Tell me a {adjective} joke about {content}.";

    const promptTemplate = PromptTemplate.fromTemplate(template);

    const formattedPromptTemplate = await promptTemplate.format({
      adjective: "funny",
      content: "chickens",
    });

    return {
      inputVariables: promptTemplate.inputVariables,
      formattedPromptTemplate,
    };
  }

  async createChatPromptTemplate() {
    const systemTemplate =
      "You are a helpful assistant that translates {input_language} to {output_language}.";
    const humanTemplate = "{text}";

    const chatPrompt = ChatPromptTemplate.fromMessages([
      ["system", systemTemplate],
      ["human", humanTemplate],
    ]);

    // Format the messages
    const formattedChatPrompt = await chatPrompt.formatMessages({
      input_language: "English",
      output_language: "French",
      text: "I love programming.",
    });

    return formattedChatPrompt;
  }

  async partialWithStrings1() {
    const prompt = new PromptTemplate({
      template: "{foo}{bar}",
      inputVariables: ["foo", "bar"],
    });

    const partialPrompt = await prompt.partial({
      foo: "foo",
    });

    const formattedPrompt = await partialPrompt.format({
      bar: "baz",
    });

    return formattedPrompt;
  }

  async partialWithStrings2() {
    const prompt = new PromptTemplate({
      template: "{foo}{bar}",
      inputVariables: ["bar"],
      partialVariables: {
        foo: "foo",
      },
    });

    const formattedPrompt = await prompt.format({
      bar: "baz",
    });

    return formattedPrompt;
  }

  async partialWithFunctions1() {
    const getCurrentDate = () => {
      return new Date().toISOString();
    };

    const prompt = new PromptTemplate({
      template: "Tell me a {adjective} joke about the day {date}",
      inputVariables: ["adjective", "date"],
    });

    const partialPrompt = await prompt.partial({
      date: getCurrentDate,
    });

    const formattedPrompt = await partialPrompt.format({
      adjective: "funny",
    });

    return formattedPrompt;
  }

  async partialWithFunctions2() {
    const prompt = new PromptTemplate({
      template: "Tell me a {adjective} joke about the day {date}",
      inputVariables: ["adjective"],
      partialVariables: {
        date: () => new Date().toISOString(),
      },
    });

    const formattedPrompt = await prompt.format({
      adjective: "funny",
    });

    return formattedPrompt;
  }

  async pipeline() {
    const fullPrompt = PromptTemplate.fromTemplate(
      `{introduction}{example}{start}`
    );

    const introductionPrompt = PromptTemplate.fromTemplate(
      `You are impersonating {person}.`
    );

    const examplePrompt =
      PromptTemplate.fromTemplate(`Here's an example of an interaction:
                                   Q: {example_q}
                                   A: {example_a}`);

    const startPrompt = PromptTemplate.fromTemplate(`Now, do this for real!
                                                     Q: {input}
                                                     A:`);

    const composedPrompt = new PipelinePromptTemplate({
      pipelinePrompts: [
        {
          name: "introduction",
          prompt: introductionPrompt,
        },
        {
          name: "example",
          prompt: examplePrompt,
        },
        {
          name: "start",
          prompt: startPrompt,
        },
      ],
      finalPrompt: fullPrompt,
    });

    const formattedPrompt = await composedPrompt.format({
      person: "Elon Musk",
      example_q: `What's your favorite car?`,
      example_a: "Tesla",
      input: `What's your favorite social media site?`,
    });

    return formattedPrompt;
  }

  async selectByLength() {
    // Create a prompt template that will be used to format the examples
    const examplePrompt = new PromptTemplate({
      inputVariables: ["input", "output"],
      template: "Input: {input}\nOutput: {output}",
    });

    // Create a LengthBasedExampleSelector that will be used to select the examples
    const exampleSelector = await LengthBasedExampleSelector.fromExamples(
      [
        { input: "happy", output: "sad" },
        { input: "tall", output: "short" },
        { input: "energetic", output: "lethargic" },
        { input: "sunny", output: "gloomy" },
        { input: "windy", output: "calm" },
      ],
      {
        examplePrompt,
        maxLength: 25,
      }
    );

    // Create a FewShotPromptTemplate that will use the example selector
    const dynamicPrompt = new FewShotPromptTemplate({
      // We provide an ExampleSelector instead of examples
      exampleSelector,
      examplePrompt,
      prefix: "Give the antonym of every input",
      suffix: "Input: {adjective}\nOutput:",
      inputVariables: ["adjective"],
    });

    // An example with small input, so it selects all examples
    const smallInput = await dynamicPrompt.format({ adjective: "big" });

    // An example with long input, so it selects only one example
    const longString =
      "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else";
    const longInput = await dynamicPrompt.format({ adjective: longString });

    return {
      smallInput,
      longInput,
    };
  }

  async selectBySimilarity() {
    // Create a prompt template that will be used to format the examples
    const examplePrompt = new PromptTemplate({
      inputVariables: ["input", "output"],
      template: "Input: {input}\nOutput: {output}",
    });

    // Create a SemanticSimilarityExampleSelector that will be used to select the examples
    const exampleSelector =
      await SemanticSimilarityExampleSelector.fromExamples(
        [
          { input: "happy", output: "sad" },
          { input: "tall", output: "short" },
          { input: "energetic", output: "lethargic" },
          { input: "sunny", output: "gloomy" },
          { input: "windy", output: "calm" },
        ],
        new OpenAIEmbeddings(),
        HNSWLib,
        { k: 1 }
      );

    // Create a FewShotPromptTemplate that will use the example selector
    const dynamicPrompt = new FewShotPromptTemplate({
      // We provide an ExampleSelector instead of examples
      exampleSelector,
      examplePrompt,
      prefix: "Give the antonym of every input",
      suffix: "Input: {adjective}\nOutput:",
      inputVariables: ["adjective"],
    });

    // Input is about the weather, so should select eg. the sunny/gloomy example
    const weatherInput = await dynamicPrompt.format({ adjective: "rainy" });

    // Input is a measurement, so should select the tall/short example
    const measureInput = await dynamicPrompt.format({ adjective: "large" });

    return {
      weatherInput,
      measureInput,
    };
  }

  async callMethod1() {
    return await this.llm.call("Tell me a joke");
  }

  async callMethod2() {
    const model = new OpenAI({
      // customize openai model that's used
      modelName: "text-ada-001",

      // `max_tokens` supports a magic -1 param where the max token length for the
      // specified modelName is calculated and included in the request to OpenAI
      // as the `max_tokens` param
      maxTokens: -1,

      // use `modelKwargs` to pass params directly to the openai call
      // note that they use snake_case instead of camelCase
      modelKwargs: {
        user: "me",
      },

      // for additional logging for debugging purposes
      verbose: true,
    });

    const resA = await model.call(
      "What would be a good company name for a company that makes colorful socks?"
    );
    return { resA };
  }

  async generateMethod() {
    const llmResult = await this.llm.generate(
      ["Tell me a joke", "Tell me a poem"],
      ["Tell me a joke", "Tell me a poem"]
    );

    return {
      length: llmResult.generations.length,
      firstGen: llmResult.generations[0],
      secondGen: llmResult.generations[1],
      output: llmResult.llmOutput,
    };
  }

  async cancellingRequests() {
    const controller = new AbortController();

    // Call `controller.abort()` somewhere to cancel the request.

    const res = await this.llm.call(
      "What would be a good company name for a company that makes colorful socks?",
      { signal: controller.signal }
    );

    return { res, signal: controller.signal };
  }

  async caching() {
    // To make the caching really obvious, lets use a slower model
    const model = new OpenAI({
      modelName: "text-davinci-002",
      cache: true,
      n: 2,
      bestOf: 2,
    });

    // The first time, it is not yet in cache, so it should take longer
    let startTime = Date.now();
    const res = await model.predict("Tell me a joke");
    const endTime = Date.now() - startTime;

    // The second time it is, so it goes faster
    startTime = Date.now();
    const res2 = await model.predict("Tell me a joke");
    const endTime2 = Date.now() - startTime;

    return { res, endTime, res2, endTime2 };
  }

  async streaming() {
    // To enable streaming, we pass in `streaming: true` to the LLM constructor.
    // Additionally, we pass in a handler for the `handleLLMNewToken` event.
    const model = new OpenAI({
      maxTokens: 25,
      streaming: true,
    });

    const response = [];
    await model.call("Tell me a joke.", {
      callbacks: [
        {
          handleLLMNewToken(token: string) {
            response.push({ token });
          },
        },
      ],
    });

    return { response };
  }

  async subscribeEvents() {
    const responseObj = {};

    // We can pass in a list of CallbackHandlers to the LLM constructor
    // to get callbacks for various events
    const model = new OpenAI({
      callbacks: [
        {
          handleLLMStart: async (llm: Serialized, prompts: string[]) => {
            responseObj["llm"] = JSON.stringify(llm, null, 2);
            responseObj["prompts"] = JSON.stringify(prompts, null, 2);
          },
          handleLLMEnd: async (output: LLMResult) => {
            responseObj["output"] = JSON.stringify(output, null, 2);
          },
          handleLLMError: async (err: Error) => {
            responseObj["error"] = err;
          },
        },
      ],
    });

    await model.call(
      "What would be a good company name a company that makes colorful socks?"
    );

    return responseObj;
  }

  async addingTimeout() {
    try {
      return await this.llm.call(
        "What would be a good company name a company that makes colorful socks?",
        { timeout: 100 }
      );
    } catch (err) {
      return err.message;
    }
  }

  async chatCallMethod1() {
    const chat = new ChatOpenAI();
    // Pass in a list of messages to `call` to start a conversation
    // In this simple example, we only pass in one message
    const response = await chat.call([
      new HumanMessage(
        "What is a good name for a company that makes colorful socks?"
      ),
    ]);

    return response;
  }

  async chatCallMethod2() {
    const chat = new ChatOpenAI();

    const response2 = await chat.call([
      new SystemMessage(
        "You are a helpful assistant that translates English to French."
      ),
      new HumanMessage("Translate: I love programming."),
    ]);

    return response2;
  }

  async chatGenerateMethod() {
    const chat = new ChatOpenAI();

    const response3 = await chat.generate([
      [
        new SystemMessage(
          "You are a helpful assistant that translates English to French."
        ),
        new HumanMessage(
          "Translate this sentence from English to French. I love programming."
        ),
      ],
      [
        new SystemMessage(
          "You are a helpful assistant that translates English to French."
        ),
        new HumanMessage(
          "Translate this sentence from English to French. I love artificial intelligence."
        ),
      ],
    ]);

    return response3;
  }

  async cancellingChatRequests() {
    const model = new ChatOpenAI({ temperature: 1 });
    const controller = new AbortController();

    // Call `controller.abort()` somewhere to cancel the request.

    const res = await model.call(
      [
        new HumanMessage(
          "What is a good name for a company that makes colorful socks?"
        ),
      ],
      { signal: controller.signal }
    );

    return res;
  }

  async openAiFunctionCalling() {
    const extractionFunctionSchema = {
      name: "extractor",
      description: "Extracts fields from the input.",
      parameters: {
        type: "object",
        properties: {
          tone: {
            type: "string",
            enum: ["positive", "negative"],
            description: "The overall tone of the input",
          },
          word_count: {
            type: "number",
            description: "The number of words in the input",
          },
          chat_response: {
            type: "string",
            description: "A response to the human's input",
          },
        },
        required: ["tone", "word_count", "chat_response"],
      },
    };

    // Bind function arguments to the model.
    // All subsequent invoke calls will use the bound parameters.
    // "functions.parameters" must be formatted as JSON Schema
    // Omit "function_call" if you want the model to choose a function to call.
    const model = new ChatOpenAI({
      modelName: "gpt-4",
    }).bind({
      functions: [extractionFunctionSchema],
      function_call: { name: "extractor" },
    });

    const result = await model.invoke([
      new HumanMessage("What a beautiful day!"),
    ]);

    return result;

    // Alternatively, you can pass function call arguments as an additional argument as a one-off:
    /*
    const model = new ChatOpenAI({
      modelName: "gpt-4",
    });
    
    const result = await model.call([
      new HumanMessage("What a beautiful day!")
    ], {
      functions: [extractionFunctionSchema],
      function_call: {name: "extractor"}
    });
    */
  }

  async chatModelCaching() {
    const model = new ChatOpenAI({
      modelName: "gpt-4",
      cache: true,
      n: 2,
    });

    // The first time, it is not yet in cache, so it should take longer
    let startTime = Date.now();
    const res = await model.predict("Tell me a joke");
    const endTime = Date.now() - startTime;

    // The second time it is, so it goes faster
    startTime = Date.now();
    const res2 = await model.predict("Tell me a joke");
    const endTime2 = Date.now() - startTime;

    return { res, endTime, res2, endTime2 };
  }

  async llmChain() {
    const template =
      "You are a helpful assistant that translates {input_language} to {output_language}.";
    const humanTemplate = "{text}";

    const chatPrompt = ChatPromptTemplate.fromMessages([
      ["system", template],
      ["human", humanTemplate],
    ]);

    const chat = new ChatOpenAI({
      temperature: 0,
    });

    const chain = new LLMChain({
      llm: chat,
      prompt: chatPrompt,
    });

    const result = await chain.call({
      input_language: "English",
      output_language: "French",
      text: "I love programming",
    });
  }

  async prompts() {
    const template =
      "You are a helpful assistant that translates {input_language} to {output_language}.";
    const systemMessagePrompt =
      SystemMessagePromptTemplate.fromTemplate(template);
    const humanTemplate = "{text}";
    const humanMessagePrompt =
      HumanMessagePromptTemplate.fromTemplate(humanTemplate);

    const chatPrompt = ChatPromptTemplate.fromMessages([
      systemMessagePrompt,
      humanMessagePrompt,
    ]);

    const chat = new ChatOpenAI({
      temperature: 0,
    });

    const chain = new LLMChain({
      llm: chat,
      prompt: chatPrompt,
    });

    const result = await chain.call({
      input_language: "English",
      output_language: "French",
      text: "I love programming",
    });
  }

  async chatModelStreaming() {
    const chat = new ChatOpenAI({
      maxTokens: 25,
      streaming: true,
    });

    const response = await chat.call([new HumanMessage("Tell me a joke.")], {
      callbacks: [
        {
          handleLLMNewToken(token: string) {
            console.log({ token });
          },
        },
      ],
    });

    return response;
  }

  async chatModelSubscribeEvents() {
    // We can pass in a list of CallbackHandlers to the LLM constructor to get callbacks for various events.
    const model = new ChatOpenAI({
      callbacks: [
        {
          handleLLMStart: async (llm: Serialized, prompts: string[]) => {
            console.log(JSON.stringify(llm, null, 2));
            console.log(JSON.stringify(prompts, null, 2));
          },
          handleLLMEnd: async (output: LLMResult) => {
            console.log(JSON.stringify(output, null, 2));
          },
          handleLLMError: async (err: Error) => {
            console.error(err);
          },
        },
      ],
    });

    const response = await model.call([
      new HumanMessage(
        "What is a good name for a company that makes colorful socks?"
      ),
    ]);

    return response;
  }

  async chatModelAddingTimeout() {
    const chat = new ChatOpenAI({ temperature: 1 });

    try {
      const response = await chat.call(
        [
          new HumanMessage(
            "What is a good name for a company that makes colorful socks?"
          ),
        ],
        { timeout: 100 }
      );

      return response;
    } catch (err) {
      return err.message;
    }
  }

  async structuredOutputParser() {
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
      answer: "answer to the user's question",
      source: "source used to answer the user's question, should be a website.",
    });

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Answer the users question as best as possible.\n{format_instructions}\n{question}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      question: "What is the capital of France?",
      format_instructions: parser.getFormatInstructions(),
    });

    return { format_instructions: parser.getFormatInstructions(), response };
  }

  async structuredOutputParserZodSchema() {
    // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const parser = StructuredOutputParser.fromZodSchema(
      z.object({
        answer: z.string().describe("answer to the user's question"),
        sources: z
          .array(z.string())
          .describe("sources used to answer the question, should be websites."),
      })
    );

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Answer the users question as best as possible.\n{format_instructions}\n{question}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      question: "What is the capital of France?",
      format_instructions: parser.getFormatInstructions(),
    });

    return { format_instructions: parser.getFormatInstructions(), response };
  }

  async outputParserLlmChain() {
    const outputParser = StructuredOutputParser.fromZodSchema(
      z
        .array(
          z.object({
            fields: z.object({
              Name: z.string().describe("The name of the country"),
              Capital: z.string().describe("The country's capital"),
            }),
          })
        )
        .describe("An array of Airtable records, each representing a country")
    );

    const chatModel = new ChatOpenAI({
      modelName: "gpt-4", // Or gpt-3.5-turbo
      temperature: 0, // For best results with the output fixing parser
    });

    const outputFixingParser = OutputFixingParser.fromLLM(
      chatModel,
      outputParser
    );

    // Don't forget to include formatting instructions in the prompt!
    const prompt = new PromptTemplate({
      template: `Answer the user's question as best you can:\n{format_instructions}\n{query}`,
      inputVariables: ["query"],
      partialVariables: {
        format_instructions: outputFixingParser.getFormatInstructions(),
      },
    });

    const answerFormattingChain = new LLMChain({
      llm: chatModel,
      prompt,
      outputKey: "records", // For readability - otherwise the chain output will default to a property named "text"
      outputParser: outputFixingParser,
    });

    const result = await answerFormattingChain.call({
      query: "List 5 countries.",
    });

    return JSON.stringify(result.records, null, 2);
  }

  async bytesOutputParser() {
    const chain = RunnableSequence.from([
      new ChatOpenAI({ temperature: 0 }),
      new BytesOutputParser(),
    ]);

    const stream = await chain.stream("Hello there!");

    const decoder = new TextDecoder();

    for await (const chunk of stream) {
      if (chunk) {
        console.log("chunk: ", chunk);
        console.log("decoded: ", decoder.decode(chunk));
      }
    }

    return stream;
  }

  async combiningOutputParser() {
    const answerParser = StructuredOutputParser.fromNamesAndDescriptions({
      answer: "answer to the user's question",
      source: "source used to answer the user's question, should be a website.",
    });

    const confidenceParser = new RegexParser(
      /Confidence: (A|B|C), Explanation: (.*)/,
      ["confidence", "explanation"],
      "noConfidence"
    );

    const parser = new CombiningOutputParser(answerParser, confidenceParser);

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Answer the users question as best as possible.\n{format_instructions}\n{question}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      question: "What is the capital of France?",
      format_instructions: parser.getFormatInstructions(),
    });

    return response;
  }

  async listParser() {
    const parser = new CommaSeparatedListOutputParser();

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "List five {subject}.\n{format_instructions}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      subject: "ice cream flavors",
      format_instructions: parser.getFormatInstructions(),
    });

    return response;
  }

  async customListParser() {
    const parser = new CustomListOutputParser({ length: 3, separator: "\n" });

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Provide a list of {subject}.\n{format_instructions}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      subject: "great science fiction books (book, author)",
      format_instructions: parser.getFormatInstructions(),
    });

    return response;
  }

  async autoFixingParser() {
    const parser = StructuredOutputParser.fromZodSchema(
      z.object({
        answer: z.string().describe("answer to the user's question"),
        sources: z
          .array(z.string())
          .describe("sources used to answer the question, should be websites."),
      })
    );

    /** This is a bad output because sources is a string, not a list */
    const badOutput = `\`\`\`json
    {
      "answer": "foo",
      "sources": "foo.com"
    }
    \`\`\``;

    const fixParser = OutputFixingParser.fromLLM(
      new ChatOpenAI({ temperature: 0 }),
      parser
    );
    const output = await fixParser.parse(badOutput);
    return output;
  }

  async stringOutputParser() {
    const parser = new StringOutputParser();

    const model = new ChatOpenAI({ temperature: 0 });

    const stream = await model.pipe(parser).stream("Hello there!");

    for await (const chunk of stream) {
      console.log(chunk);
    }

    return stream;
  }

  async structuredOutputParser2() {
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
      answer: "answer to the user's question",
      source: "source used to answer the user's question, should be a website.",
    });

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Answer the users question as best as possible.\n{format_instructions}\n{question}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      question: "What is the capital of France?",
      format_instructions: parser.getFormatInstructions(),
    });

    return { format_instructions: parser.getFormatInstructions(), response };
  }

  async structuredOutputParserZodSchema2() {
    const parser = StructuredOutputParser.fromZodSchema(
      z.object({
        answer: z.string().describe("answer to the user's question"),
        sources: z
          .array(z.string())
          .describe("sources used to answer the question, should be websites."),
      })
    );

    const chain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        "Answer the users question as best as possible.\n{format_instructions}\n{question}"
      ),
      new OpenAI({ temperature: 0 }),
      parser,
    ]);

    const response = await chain.invoke({
      question: "What is the capital of France?",
      format_instructions: parser.getFormatInstructions(),
    });

    return { format_instructions: parser.getFormatInstructions(), response };
  }
}
