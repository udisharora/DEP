import type { VehicleLookupResponse, WorkerResponse } from '../types/alpr';

export async function startAlprTask(file: File): Promise<{ task_id: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/process-image', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function getTaskStatus(taskId: string): Promise<WorkerResponse> {
  const response = await fetch(`/api/status/${taskId}`);

  if (!response.ok) {
    throw new Error('Polling failed');
  }

  return response.json();
}

export async function getVehicleInfo(plateText: string): Promise<VehicleLookupResponse> {
  const response = await fetch(`/api/vehicle/${plateText}`);

  if (!response.ok) {
    throw new Error('Failed to fetch vehicle data');
  }

  return response.json();
}
