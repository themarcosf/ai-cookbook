import { Module } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";

import { AppService } from "./app.service";
import { AppController } from "./app.controller";
import { ModelIoModule } from "./model-io/model-io.module";
import { RetrievalModule } from "./retrieval/retrieval.module";

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
    ModelIoModule,
    RetrievalModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
