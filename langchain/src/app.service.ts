import { Injectable } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";

import { OpenAI } from "langchain/llms/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BaseOutputParser } from "langchain/schema/output_parser";
import { PromptTemplate, ChatPromptTemplate } from "langchain/prompts";

/**
 * Parse the output of an LLM call to a comma-separated list.
 */
class CommaSeparatedListOutputParser extends BaseOutputParser<string[]> {
  async parse(text: string): Promise<string[]> {
    return text.split(",").map((item) => item.trim());
  }

  getFormatInstructions(opt: any) {
    return "";
  }

  lc_namespace = [""];
}

@Injectable()
export class AppService {
  constructor(private readonly configService: ConfigService) {}

  async introductoryExample() {
    const llm = new OpenAI({
      openAIApiKey: this.configService.get("OPENAI_API_KEY"),
      temperature: 0.9,
    });

    // const chatModel = new ChatOpenAI(llm);
    const chatModel = new ChatOpenAI();

    const text =
      "What would be a good company name for a company that makes colorful socks?";

    const llmResult = await llm.predict(text);
    const chatModelResult = await chatModel.predict(text);

    return {
      llmResult,
      chatModelResult,
    };
  }

  async promptTemplate() {
    const prompt = PromptTemplate.fromTemplate(
      "What is a good name for a company that makes {product}?"
    );

    const formattedPrompt = await prompt.format({
      product: "colorful socks",
    });

    return { formattedPrompt };
  }

  async chatPromptTemplate() {
    const template =
      "You are a helpful assistant that translates {input_language} to {output_language}.";
    const humanTemplate = "{text}";

    const chatPrompt = ChatPromptTemplate.fromMessages([
      ["system", template],
      ["human", humanTemplate],
    ]);

    const formattedChatPrompt = await chatPrompt.formatMessages({
      input_language: "English",
      output_language: "French",
      text: "I love programming.",
    });

    return { formattedChatPrompt };
  }

  async outputParser() {
    const parser = new CommaSeparatedListOutputParser();

    const result = await parser.parse("a, b, c");

    return { result };
  }

  async promptTemplate_llm_outputParser() {
    const template = `You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more.`;

    const humanTemplate = "{text}";

    /**
     * Chat prompt for generating comma-separated lists. It combines the system
     * template and the human template.
     */
    const chatPrompt = ChatPromptTemplate.fromMessages([
      ["system", template],
      ["human", humanTemplate],
    ]);

    // const model = new ChatOpenAI({});
    const model = new ChatOpenAI();
    const parser = new CommaSeparatedListOutputParser();

    const chain = chatPrompt.pipe(model).pipe(parser);

    const result = await chain.invoke({
      text: "colors",
    });

    return { result };
  }
}
