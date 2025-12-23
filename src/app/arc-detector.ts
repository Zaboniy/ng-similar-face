import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
ort.env.wasm.wasmPaths = '/assets/ort/dist/';

@Injectable({
    providedIn: 'root',
})
export class ArcDetector {
    cachedModel: ort.InferenceSession | null = null;
    convertImageToFloat32Array(img: HTMLImageElement): Float32Array {
        const canvas = document.createElement('canvas');
        canvas.width = 112;
        canvas.height = 112;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, 112, 112);

        const { data } = ctx.getImageData(0, 0, 112, 112); // RGBA
        const float32Data = new Float32Array(112 * 112 * 3);

        for (let i = 0; i < 112 * 112; i++) {
            // NHWC Layout: RGB values stay together
            float32Data[i * 3 + 0] = (data[i * 4 + 0] - 127.5) / 128.0; // R
            float32Data[i * 3 + 1] = (data[i * 4 + 1] - 127.5) / 128.0; // G
            float32Data[i * 3 + 2] = (data[i * 4 + 2] - 127.5) / 128.0; // B
        }
        return float32Data;
    }

    convertImageToFloat32ArrayForResnet(img: HTMLImageElement): Float32Array {
        const canvas = document.createElement('canvas');
        canvas.width = 112;
        canvas.height = 112;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, 112, 112);

        const { data } = ctx.getImageData(0, 0, 112, 112); // RGBA
        const float32Data = new Float32Array(3 * 112 * 112);

        // NCHW Layout: Convert to channels-first format for ResNet
        for (let i = 0; i < 112 * 112; i++) {
            float32Data[i] = (data[i * 4 + 0] - 127.5) / 128.0; // R channel
            float32Data[112 * 112 + i] = (data[i * 4 + 1] - 127.5) / 128.0; // G channel
            float32Data[2 * 112 * 112 + i] = (data[i * 4 + 2] - 127.5) / 128.0; // B channel
        }
        return float32Data;
    }

    async getModelBuffer(url: string): Promise<ArrayBuffer> {
        const response = await fetch(url);
        return await response.arrayBuffer();
    }

    async getSession(): Promise<ort.InferenceSession> {
        if (this.cachedModel) {
            return this.cachedModel;
        }
        const modelBuffer = await this.getModelBuffer('assets/ort/arc.onnx');
        this.cachedModel = await ort.InferenceSession.create(modelBuffer);
        return this.cachedModel;
    }


    async runArcFaceInference(img: HTMLImageElement) {
        const float32Data = this.convertImageToFloat32Array(img);
        const session = await this.getSession();

        // 2. Prepare input tensor from your pre-processed image data
        // imageData should be a Float32Array of size 1*3*112*112
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 112, 112, 3]);
        const feeds = { [session.inputNames[0]]: inputTensor };
        try {
            // Convert image to Float32Array

            const results = await session.run(feeds);

            // 4. Get the output embedding (e.g., 'output' or another name depending on the model)
            const outputName = session.outputNames[0];
            const embedding = results[outputName].data; // This is your 512-dim embedding

            return embedding
        } catch (e) {
            const float32Data = this.convertImageToFloat32ArrayForResnet(img);
            const session = await this.getSession();
            const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 112, 112]);
            const feeds = { [session.inputNames[0]]: inputTensor };
            const results = await session.run(feeds);
            // 4. Get the output embedding (e.g., 'output' or another name depending on the model)
            const outputName = session.outputNames[0];
            const embedding = results[outputName].data; // This is your 512-dim embedding

            return embedding
        }
    }

}
