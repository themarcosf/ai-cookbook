import { Module } from "@nestjs/common";
import { AppController } from "./app.controller";
import { AppService } from "./app.service";
import { ConfigModule } from "@nestjs/config";
import { ModelIoModule } from './model-io/model-io.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
    ModelIoModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
