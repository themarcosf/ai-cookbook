import { Controller, Get } from "@nestjs/common";
import { ChainsService } from "./chains.service";

@Controller("chains")
export class ChainsController {
  constructor(private readonly chainsService: ChainsService) {}

  @Get("basic-example")
  basicExample() {
    return this.chainsService.basicExample();
  }

  @Get("basic-example-chat-model")
  basicExampleChatModel() {
    return this.chainsService.basicExampleChatModel();
  }

  @Get("debugging-chains")
  debuggingChains() {
    return this.chainsService.debuggingChains();
  }

  @Get("add-state-memory")
  addStateMemory() {
    return this.chainsService.addStateMemory();
  }

  @Get("use-stream-callback")
  useStreamCallback() {
    return this.chainsService.useStreamCallback();
  }

  @Get("abort-request")
  abortRequest() {
    return this.chainsService.abortRequest();
  }

  @Get("sequential/simple-sequential-chain")
  simpleSequentialChain() {
    return this.chainsService.simpleSequentialChain();
  }

  @Get("sequential/sequential-chain")
  sequentialChain() {
    return this.chainsService.sequentialChain();
  }

  @Get("documents/stuff")
  documentsStuff() {
    return this.chainsService.documentsStuff();
  }

  @Get("documents/refine")
  documentsRefine() {
    return this.chainsService.documentsRefine();
  }

  @Get("documents/refine-customized")
  documentsRefineCustomized() {
    return this.chainsService.documentsRefineCustomized();
  }

  @Get("documents/map-reduce")
  mapReduce() {
    return this.chainsService.mapReduce();
  }

  @Get("popular/api-chains")
  apiChains() {
    return this.chainsService.apiChains();
  }

  @Get("popular/retrieval-qa")
  retrievalQA() {
    return this.chainsService.retrievalQA();
  }

  @Get("popular/retrieval-qa-custom-chain")
  retrievalQACustomChain() {
    return this.chainsService.retrievalQACustomChain();
  }

  @Get("popular/retrieval-qa-custom-prompt")
  retrievalQACustomPrompt() {
    return this.chainsService.retrievalQACustomPrompt();
  }

  @Get("popular/retrieval-qa-return-source-documents")
  retrievalQAReturnSourceDocuments() {
    return this.chainsService.retrievalQAReturnSourceDocuments();
  }

  @Get("popular/conversational-retrieval-qa")
  conversationalRetrievalQA() {
    return this.chainsService.conversationalRetrievalQA();
  }

  @Get("popular/conversational-retrieval-qa-builtin-memory")
  conversationalRetrievalQABuiltinMemory() {
    return this.chainsService.conversationalRetrievalQABuiltinMemory();
  }

  @Get("popular/conversational-retrieval-streaming")
  conversationalRetrievalStreaming() {
    return this.chainsService.conversationalRetrievalStreaming();
  }

  @Get("popular/conversational-retrieval-externally-managed-memory")
  conversationalRetrievalExternallyManagedMemory() {
    return this.chainsService.conversationalRetrievalExternallyManagedMemory();
  }

  @Get("popular/conversational-retrieval-prompt-customization")
  conversationalRetrievalPromptCustomization() {
    return this.chainsService.conversationalRetrievalPromptCustomization();
  }
}
