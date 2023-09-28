import { Module } from '@nestjs/common';
import { ModelIoService } from './model-io.service';
import { ModelIoController } from './model-io.controller';

@Module({
  controllers: [ModelIoController],
  providers: [ModelIoService],
})
export class ModelIoModule {}
