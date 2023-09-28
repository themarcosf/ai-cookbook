import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";

import {
  PromptTemplate,
  ChatPromptTemplate,
  FewShotPromptTemplate,
  PipelinePromptTemplate,
  LengthBasedExampleSelector,
  SemanticSimilarityExampleSelector,
} from "langchain/prompts";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";

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
}
