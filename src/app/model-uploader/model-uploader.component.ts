import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ArcDetector } from '../arc-detector';
import * as ort from 'onnxruntime-web';

@Component({
  selector: 'app-model-uploader',
  imports: [CommonModule],
  templateUrl: './model-uploader.component.html',
  styleUrl: './model-uploader.component.css',
})
export class ModelUploaderComponent {
  private arcDetector = inject(ArcDetector);
  
  modelLoaded = signal<boolean>(false);
  modelName = signal<string | null>(null);
  isUploading = signal<boolean>(false);
  errorMessage = signal<string | null>(null);

  async onModelSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    
    if (input.files && input.files[0]) {
      const file = input.files[0];
      
      // Validate file type
      if (!file.name.endsWith('.onnx')) {
        this.errorMessage.set('Please upload a valid ONNX model file (.onnx)');
        return;
      }
      
      this.isUploading.set(true);
      this.errorMessage.set(null);
      
      try {
        // Read the file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();
        const session = await ort.InferenceSession.create(arrayBuffer);
        this.arcDetector.cachedModel = session;
        this.modelLoaded.set(true);
        this.modelName.set(file.name);
      } catch (error) {
        this.errorMessage.set('Failed to load model. Please ensure it is a valid ONNX model.');
        this.modelLoaded.set(false);
        this.modelName.set(null);
      } finally {
        this.isUploading.set(false);
      }
    }
  }

  clearModel(): void {
    this.arcDetector.cachedModel = null;
    this.modelLoaded.set(false);
    this.modelName.set(null);
    this.errorMessage.set(null);
  }
}
