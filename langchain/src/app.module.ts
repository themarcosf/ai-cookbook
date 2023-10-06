import { Module } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";

import { AppService } from "./app.service";
import { AppController } from "./app.controller";
import { ModelIoModule } from "./model-io/model-io.module";
import { RetrievalModule } from "./retrieval/retrieval.module";
import { ChainsModule } from './chains/chains.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
    ModelIoModule,
    RetrievalModule,
    ChainsModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
