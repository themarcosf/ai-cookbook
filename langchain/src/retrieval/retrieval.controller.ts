import { Controller, Get } from "@nestjs/common";
import { RetrievalService } from "./retrieval.service";

@Controller("retrieval")
export class RetrievalController {
  constructor(private readonly retrievalService: RetrievalService) {}

  @Get("document-loaders/basic-example")
  documentLoaders() {
    return this.retrievalService.documentLoaders();
  }

  @Get("document-loaders/how-to/creating-document")
  creatingDocument() {
    return this.retrievalService.creatingDocument();
  }

  @Get("document-loaders/how-to/csv-all-columns")
  csvAllColumns() {
    return this.retrievalService.csvAllColumns();
  }

  @Get("document-loaders/how-to/csv-single-column")
  csvSingleColumn() {
    return this.retrievalService.csvSingleColumn();
  }

  @Get("document-loaders/how-to/json-no-pointer")
  jsonNoPointer() {
    return this.retrievalService.jsonNoPointer();
  }

  @Get("document-loaders/how-to/json-pointer")
  jsonPointer() {
    return this.retrievalService.jsonPointer();
  }

  @Get("document-loaders/how-to/file-directory")
  fileDirectory() {
    return this.retrievalService.fileDirectory();
  }

  @Get("document-transformers/text-splitter-raw-text")
  textSplitterRawText() {
    return this.retrievalService.textSplitterRawText();
  }

  @Get("document-transformers/text-splitter-document")
  textSplitterDocument() {
    return this.retrievalService.textSplitterDocument();
  }

  @Get("document-transformers/text-splitter-customized")
  textSplitterCustomized() {
    return this.retrievalService.textSplitterCustomized();
  }

  @Get("document-transformers/metadata-tagger")
  metadataTagger() {
    return this.retrievalService.metadataTagger();
  }

  @Get("document-transformers/metadata-tagger-customized")
  metadataTaggerCustomized() {
    return this.retrievalService.metadataTaggerCustomized();
  }

  @Get("document-transformers/text-spliters/code-split-javascript")
  codeSplitJavascript() {
    return this.retrievalService.codeSplitJavascript();
  }

  @Get("document-transformers/text-spliters/contextual-chunk-headers")
  contextualChunkHeaders() {
    return this.retrievalService.contextualChunkHeaders();
  }

  @Get("text-embedding/basic-example")
  basicExample() {
    return this.retrievalService.basicExample();
  }

  @Get("vector-stores/new-index-from-text")
  newIndexFromText() {
    return this.retrievalService.newIndexFromText();
  }

  @Get("vector-stores/new-index-from-loader")
  newIndexFromLoader() {
    return this.retrievalService.newIndexFromLoader();
  }

  @Get("retrievers/basic-example")
  retrieversBasicExample() {
    return this.retrievalService.retrieversBasicExample();
  }

  @Get("retrievers/contextual-compression")
  contextualCompression() {
    return this.retrievalService.contextualCompression();
  }

  @Get("retrievers/multiquery-retriever")
  multiqueryRetriever() {
    return this.retrievalService.multiqueryRetriever();
  }

  @Get("retrievers/multiquery-retriever-customized")
  multiqueryRetrieverCustomized() {
    return this.retrievalService.multiqueryRetrieverCustomized();
  }

  @Get("retrievers/multivector-retriever-smaller-chunks")
  multivectorRetrieverSmallerChunks() {
    return this.retrievalService.multivectorRetrieverSmallerChunks();
  }

  @Get("retrievers/multivector-retriever-summary")
  multivectorRetrieverSummary() {
    return this.retrievalService.multivectorRetrieverSummary();
  }

  @Get("retrievers/multivector-retriever-hypothetical-queries")
  multivectorRetrieverHypotheticalQueries() {
    return this.retrievalService.multivectorRetrieverHypotheticalQueries();
  }

  @Get("retrievers/parent-document-retriever")
  parentDocumentRetriever() {
    return this.retrievalService.parentDocumentRetriever();
  }

  @Get("retrievers/self-querying")
  selfQuerying() {
    return this.retrievalService.selfQuerying();
  }

  @Get("retrievers/similarity-score-threshold")
  similarityScoreThreshold() {
    return this.retrievalService.similarityScoreThreshold();
  }

  @Get("retrievers/time-weighted-vector-store")
  timeWeightedVectorStore() {
    return this.retrievalService.timeWeightedVectorStore();
  }
}
