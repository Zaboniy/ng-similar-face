import { Component, signal, inject, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FaceApiDetector } from '../face-api-detector';
import { ArcDetector } from '../arc-detector';

let nextId = 0;

@Component({
  selector: 'app-image-upload',
  imports: [CommonModule],
  templateUrl: './image-upload.component.html',
})
export class ImageUploadComponent {
  private faceApiDetector = inject(FaceApiDetector);

  imageUploaded = output<HTMLImageElement>();
  
  readonly componentId = `image-upload-${nextId++}`;
  uploadedImage = signal<string | null>(null);
  isProcessing = signal<boolean>(false);
  
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    
    if (input.files && input.files[0]) {
      const file = input.files[0];
      
      const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!allowedTypes.includes(file.type)) {
        alert('Please upload a valid image file (.jpg, .jpeg, or .png)');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = async (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          const imageDataUrl = e.target.result as string;
          await this.validateAndSetImage(imageDataUrl);
        }
      };
      reader.readAsDataURL(file);
    }
  }
  
  private async validateAndSetImage(imageDataUrl: string): Promise<void> {
    this.isProcessing.set(true);
    
    try {
      const img = new Image();
      img.src = imageDataUrl;
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
      const detection = await this.faceApiDetector.detectSingleFace(img);
      if (detection) {
        this.uploadedImage.set(detection.src);
        this.imageUploaded.emit(detection);
      } else{
        alert('No face detected. Please upload an image with a single face.');
      }
    } catch (error) {
      console.error('Error validating image:', error);
      alert('Error processing the image. Please try again.');
    } finally {
      this.isProcessing.set(false);
    }
  }
  
  clearImage(): void {
    this.uploadedImage.set(null);
  }
}