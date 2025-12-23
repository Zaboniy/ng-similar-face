import { Injectable } from '@angular/core';

declare const faceapi: any;

@Injectable({
  providedIn: 'root',
})
export class FaceApiDetector {
  private readonly MODEL_URL = '/assets/models';
  private modelsLoaded = false;
  private loadingPromise: Promise<void> | null = null;

  async loadModels(): Promise<void> {
    if (this.modelsLoaded) {
      return;
    }

    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    this.loadingPromise = (async () => {
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(this.MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(this.MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(this.MODEL_URL),
          faceapi.nets.tinyFaceDetector.loadFromUri(this.MODEL_URL),
        ]);
        this.modelsLoaded = true;
      } catch (error) {
        this.loadingPromise = null;
        throw error;
      }
    })();

    return this.loadingPromise;
  }

  isModelsLoaded(): boolean {
    return this.modelsLoaded;
  }

  async detectSingleFace(img: HTMLImageElement): Promise<any | undefined> {
    if (!this.modelsLoaded) {
      await this.loadModels();
    }
    const detection = await faceapi.detectSingleFace(
      img,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 320,
        scoreThreshold: 0.5
      })
    );

    if (!detection) return;
    const faceCanvasArray = await faceapi.extractFaces(img, [detection.box]);
    const faceCanvas = faceCanvasArray[0];

    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 112;
    resizedCanvas.height = 112;
    const ctx = resizedCanvas.getContext('2d');
    if (ctx) ctx.drawImage(faceCanvas, 0, 0, 112, 112);

    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.src = resizedCanvas.toDataURL('image/jpeg', 0.95);
    });
  }

  async getImageEmbedding(img: HTMLImageElement): Promise<Float32Array | null> {
    if (!this.modelsLoaded) {
      await this.loadModels();
    }

    const result = await faceapi.detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (result && result.descriptor) {
      return result.descriptor;
    }
    
    return null;
  }
}
