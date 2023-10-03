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

  @Get("language-models/llm/how-to/subscribing-to-events")
  subscribeEvents() {
    return this.modelIoService.subscribeEvents();
  }

  @Get("language-models/llm/how-to/adding-timeout")
  addingTimeout() {
    return this.modelIoService.addingTimeout();
  }

  @Get("language-models/chat-models/call-method-1")
  chatCallMethod1() {
    return this.modelIoService.chatCallMethod1();
  }

  @Get("language-models/chat-models/call-method-2")
  chatCallMethod2() {
    return this.modelIoService.chatCallMethod2();
  }

  @Get("language-models/chat-models/generate-method")
  chatGenerateMethod() {
    return this.modelIoService.chatGenerateMethod();
  }

  @Get("language-models/chat-models/how-to/cancelling-requests")
  cancellingChatRequests() {
    return this.modelIoService.cancellingChatRequests();
  }

  @Get("language-models/chat-models/how-to/openai-function-calling")
  openAiFunctionCalling() {
    return this.modelIoService.openAiFunctionCalling();
  }

  @Get("language-models/chat-models/how-to/caching")
  chatModelCaching() {
    return this.modelIoService.chatModelCaching();
  }

  @Get("language-models/chat-models/how-to/llm-chain")
  llmChain() {
    return this.modelIoService.llmChain();
  }

  @Get("language-models/chat-models/how-to/prompts")
  prompts() {
    return this.modelIoService.prompts();
  }

  @Get("language-models/chat-models/how-to/streaming")
  chatModelStreaming() {
    return this.modelIoService.chatModelStreaming();
  }

  @Get("language-models/chat-models/how-to/subscribe-events")
  chatModelSubscribeEvents() {
    return this.modelIoService.chatModelSubscribeEvents();
  }

  @Get("language-models/chat-models/how-to/adding-timeout")
  chatModelAddingTimeout() {
    return this.modelIoService.chatModelAddingTimeout();
  }

  @Get("output-parsers/structured-output-parser")
  structuredOutputParser() {
    return this.modelIoService.structuredOutputParser();
  }

  @Get("output-parsers/structured-output-parser-zod-schema")
  structuredOutputParserZodSchema() {
    return this.modelIoService.structuredOutputParserZodSchema();
  }

  @Get("output-parsers/llm-chain")
  outputParserLlmChain() {
    return this.modelIoService.outputParserLlmChain();
  }

  @Get("output-parsers/bytes-output-parser")
  bytesOutputParser() {
    return this.modelIoService.bytesOutputParser();
  }

  @Get("output-parsers/combining-output-parser")
  combiningOutputParser() {
    return this.modelIoService.combiningOutputParser();
  }

  @Get("output-parsers/list-parser")
  listParser() {
    return this.modelIoService.listParser();
  }

  @Get("output-parsers/custom-list-parser")
  customListParser() {
    return this.modelIoService.customListParser();
  }

  @Get("output-parsers/auto-fixing-parser")
  autoFixingParser() {
    return this.modelIoService.autoFixingParser();
  }

  @Get("output-parsers/string-output-parser")
  stringOutputParser() {
    return this.modelIoService.stringOutputParser();
  }

  @Get("output-parsers/structured-output-parser")
  structuredOutputParser2() {
    return this.modelIoService.structuredOutputParser2();
  }

  @Get("output-parsers/structured-output-parser-zod-schema")
  structuredOutputParserZodSchema2() {
    return this.modelIoService.structuredOutputParserZodSchema2();
  }
}
