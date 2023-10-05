import { Module } from '@nestjs/common';
import { RetrievalService } from './retrieval.service';
import { RetrievalController } from './retrieval.controller';

@Module({
  controllers: [RetrievalController],
  providers: [RetrievalService],
})
export class RetrievalModule {}
