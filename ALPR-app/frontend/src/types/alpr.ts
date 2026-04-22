export interface RtoMetadata {
  state: string;
  district_code: string;
}

export interface AlprResultData {
  restoration_msg: string;
  detection_source: string;
  original_image: string | null;
  nafnet_image: string | null;
  darkir_image: string | null;
  dehaze_image: string | null;
  derain_image: string | null;
  detection_used: string | null;
  annotated_image: string | null;
  plate_crop: string | null;
  plate_upscaled: string | null;
  extracted_text: string;
  confidence: number;
  rto_metadata: RtoMetadata;
}

export interface WorkerResponse {
  task_id: string;
  status: string;
  data?: AlprResultData;
  error?: string;
}

export interface VehicleLookupResponse {
  valid: boolean;
  data?: {
    owner?: string;
    make?: string;
    model?: string;
    engine?: string;
    fuel?: string;
    location?: string;
    registration_date?: string;
    year?: string;
    insurance?: string;
  };
  error?: string;
}

export type PipelineStatus = 'IDLE' | 'UPLOADING' | 'PENDING' | 'SUCCESS' | 'ERROR';
