
export type SimilarityLabel =
  | "Opposite"
  | "Not Similar"
  | "Somewhat Similar"
  | "Similar"
  | "Highly Similar"
  | "Exactly Same";

export interface Threshold {
  /** Upper bound (inclusive) for this bucket, in cosine similarity units. */
  max: number;
  /** Label to emit when the value falls within this bucket. */
  label: string;
}


  export function l2Normalize(vector: Float32Array): Float32Array {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return vector; // Prevent division by zero
    return vector.map(v => v / norm);
  }

  /**
   * Calculates Cosine Similarity.
   * If vectors are already L2-normalized, this is equivalent to the Dot Product.
   */
   export function cosineSimilarity(vec1: Float32Array, vec2: Float32Array): number {
    // 1. Normalize both vectors
    const norm1 = l2Normalize(vec1);
    const norm2 = l2Normalize(vec2);

    // 2. Calculate Dot Product (since magnitude is now 1.0)
    let similarity = 0;
    for (let i = 0; i < norm1.length; i++) {
      similarity += norm1[i] * norm2[i];
    }

    return similarity;
  }

/**
 * Clamp helper to keep cosine similarity in [-1, 1].
 */
export function clampCosine(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value < -1) return -1;
  if (value > 1) return 1;
  return value;
}

/**
 * Convert cosine similarity in [-1, 1] to a 0–100 percentage by shifting & scaling.
 * -1 → 0%, 0 → 50%, 1 → 100%.
 */
export function cosineToPercentageShiftScale(cosine: number): number {
  if (Number.isNaN(cosine)) return NaN;
  const c = clampCosine(cosine);
  return ((c + 1) / 2) * 100;
}

/**
 * Optional alternative: angle-based percentage.
 * Uses θ = arccos(cosine), maps 0 rad (identical) → 100%, π rad (opposite) → 0%.
 */
export function cosineToPercentageAngle(cosine: number): number {
  if (Number.isNaN(cosine)) return NaN;
  const c = clampCosine(cosine);
  const theta = Math.acos(c); // radians, in [0, π]
  return (1 - theta / Math.PI) * 100;
}

/**
 * Map cosine similarity to a label using configurable thresholds.
 * Special-case: returns "Same" when cosine is (approximately) 1.0.
 */
export function similarityLabel(
  cosine: number,
): string {
  const defaultThresholds: Threshold[] = [
    { max: 0.0, label: "Opposite" },         // [-1.0 .. 0.0]
    { max: 0.3, label: "Not Similar" },      // (0.0 .. 0.3]
    { max: 0.6, label: "Somewhat Similar" }, // (0.3 .. 0.6]
    { max: 0.8, label: "Similar" },          // (0.6 .. 0.8]
    { max: 0.9, label: "Highly Similar" },   // (0.8 .. 0.9]
    { max: 1.0, label: "Exactly Same" },   // (0.9 .. 1.0]
  ];

  const invalidLabel =  "Invalid Similarity";
  const thresholds = defaultThresholds;
  const eps = 1e-12;

  if (Number.isNaN(cosine)) return invalidLabel;

  // Clamp to [-1, 1] so minor floating error doesn’t break boundaries.
  const c = clampCosine(cosine);

  // Exact (epsilon-based) "Same" check
  if (Math.abs(1 - c) <= eps) {
    return "Same";
  }

  // Enforce that the last bucket must cover up to 1.0 for safety.
  const lastMax = thresholds[thresholds.length - 1]?.max ?? 1.0;
  if (lastMax < 1.0) {
    return invalidLabel;
  }

  for (let i = 0; i < thresholds.length; i++) {
    const { max, label } = thresholds[i];
    if (c <= max) {
      return label;
    }
  }

  return invalidLabel;
}

/**
 * Convenience function that returns both percentage(s) and label.
 */
export interface SimilarityDescription {
    input: number;
    percentageShiftScale: number; // 0–100
    percentageAngle: number;      // 0–100
    label: string;
}

export function describeSimilarity(
    cosine: number,
): SimilarityDescription {
    return {
        input: cosine,
        percentageShiftScale: cosineToPercentageShiftScale(cosine),
        percentageAngle: cosineToPercentageAngle(cosine),
        label: similarityLabel(cosine),
    };
}
