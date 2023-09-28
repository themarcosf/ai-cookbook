import { AppService } from "./app.service";
import { Controller, Get } from "@nestjs/common";

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get("introductory-example")
  introductoryExample() {
    return this.appService.introductoryExample();
  }

  @Get("prompt-template")
  promptTemplate() {
    return this.appService.promptTemplate();
  }

  @Get("chat-prompt-template")
  chatPromptTemplate() {
    return this.appService.chatPromptTemplate();
  }

  @Get("output-parser")
  outputParser() {
    return this.appService.outputParser();
  }

  @Get("promptTemplate-llm-outputParser")
  promptTemplate_llm_outputParser() {
    return this.appService.promptTemplate_llm_outputParser();
  }
}
