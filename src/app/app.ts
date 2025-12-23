import { Component, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ImageUploadComponent } from './image-upload/image-upload.component';
import { ModelUploaderComponent } from './model-uploader/model-uploader.component';
import { ArcDetector } from './arc-detector';
import { FaceApiDetector } from './face-api-detector';
import { cosineSimilarity, describeSimilarity, SimilarityDescription } from './utils';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, ImageUploadComponent, ModelUploaderComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  private arcDetector = inject(ArcDetector);
  private faceApiDetector = inject(FaceApiDetector);
  
  protected readonly title = signal('ng-similar-face');
  firstImageEmbedding = signal<Float32Array | null>(null);
  secondImageEmbedding = signal<Float32Array | null>(null);
  similarityLabel = signal<SimilarityDescription | null>(null);

  async onFirstImageUploaded(img: HTMLImageElement): Promise<void> {
    try {
      let embedding: Float32Array | null = null;
      
      // Use ARC detector if custom model is loaded, otherwise use face-api
      if (this.arcDetector.cachedModel) {
        const result = await this.arcDetector.runArcFaceInference(img);
        if (result instanceof Float32Array) {
          embedding = result;
        }
      } else {
        embedding = await this.faceApiDetector.getImageEmbedding(img);
      }
      
      if (embedding) {
        this.firstImageEmbedding.set(embedding);
        this.checkSimilarity();
      }
    } catch (error) {
      console.error('Error generating first image embedding:', error);
    }
  }

  async onSecondImageUploaded(img: HTMLImageElement): Promise<void> {
    try {
      let embedding: Float32Array | null = null;
      
      // Use ARC detector if custom model is loaded, otherwise use face-api
      if (this.arcDetector.cachedModel) {
        const result = await this.arcDetector.runArcFaceInference(img);
        if (result instanceof Float32Array) {
          embedding = result;
        }
      } else {
        embedding = await this.faceApiDetector.getImageEmbedding(img);
      }
      
      if (embedding) {
        this.secondImageEmbedding.set(embedding);
        this.checkSimilarity();
      }
    } catch (error) {
      console.error('Error generating second image embedding:', error);
    }
  }

  private checkSimilarity(): void {
    if (this.firstImageEmbedding() && this.secondImageEmbedding()) {
      const cosine = cosineSimilarity(this.firstImageEmbedding()!, this.secondImageEmbedding()!);
      const label = describeSimilarity(cosine);
      this.similarityLabel.set(label);
      console.log('label:', {label});
    } else {
      this.similarityLabel.set(null);
    }
  }
}
