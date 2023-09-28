import { Controller, Get } from "@nestjs/common";
import { ModelIoService } from "./model-io.service";

@Controller("model-io")
export class ModelIoController {
  constructor(private readonly modelIoService: ModelIoService) {}

  @Get("prompts/prompt-templates/simple-example")
  simpleExample() {
    return this.modelIoService.simpleExample();
  }

  @Get("prompts/prompt-templates/create-prompt-template-1")
  createPromptTemplate1() {
    return this.modelIoService.createPromptTemplate1();
  }

  @Get("prompts/prompt-templates/create-prompt-template-2")
  createPromptTemplate2() {
    return this.modelIoService.createPromptTemplate2();
  }

  @Get("prompts/prompt-templates/chat-prompt-template")
  createChatPromptTemplate() {
    return this.modelIoService.createChatPromptTemplate();
  }

  @Get(
    "prompts/prompt-templates/partial-prompt-templates/partial-with-strings-1"
  )
  partialWithStrings1() {
    return this.modelIoService.partialWithStrings1();
  }

  @Get(
    "prompts/prompt-templates/partial-prompt-templates/partial-with-strings-2"
  )
  partialWithStrings2() {
    return this.modelIoService.partialWithStrings2();
  }

  @Get(
    "prompts/prompt-templates/partial-prompt-templates/partial-with-functions-1"
  )
  partialWithFunctions1() {
    return this.modelIoService.partialWithFunctions1();
  }

  @Get(
    "prompts/prompt-templates/partial-prompt-templates/partial-with-functions-2"
  )
  partialWithFunctions2() {
    return this.modelIoService.partialWithFunctions2();
  }

  @Get("prompts/prompt-templates/composition/pipeline")
  pipeline() {
    return this.modelIoService.pipeline();
  }

  @Get("prompts/example-selectors/select-by-length")
  selectByLength() {
    return this.modelIoService.selectByLength();
  }

  @Get("prompts/example-selectors/select-by-similarity")
  selectBySimilarity() {
    return this.modelIoService.selectBySimilarity();
  }

  @Get("language-models/llm/call-method-1")
  callMethod1() {
    return this.modelIoService.callMethod1();
  }

  @Get("language-models/llm/call-method-2")
  callMethod2() {
    return this.modelIoService.callMethod2();
  }

  @Get("language-models/llm/generate-method")
  generateMethod() {
    return this.modelIoService.generateMethod();
  }

  @Get("language-models/llm/how-to/cancelling-requests")
  cancellingRequests() {
    return this.modelIoService.cancellingRequests();
  }

  @Get("language-models/llm/how-to/caching")
  caching() {
    return this.modelIoService.caching();
  }

  @Get("language-models/llm/how-to/streaming")
  streaming() {
    return this.modelIoService.streaming();
  }
}
