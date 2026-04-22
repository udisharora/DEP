import React, { useState, useEffect } from 'react';
import styles from './App.module.css';

interface RtoMetadata {
  state: string;
  district_code: string;
}

interface WorkerResponse {
  task_id: string;
  status: string;
  data?: {
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
  };
  error?: string;
}

function App() {
  // State to hold the user's selected image file
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // State to store the Celery background task ID returned from the backend
  const [taskId, setTaskId] = useState<string | null>(null);
  // Track the current workflow status (IDLE, UPLOADING, PENDING, SUCCESS, ERROR)
  const [status, setStatus] = useState<string>('IDLE');
  // Hold the final parsed ALPR response payload
  const [resultData, setResultData] = useState<WorkerResponse['data'] | null>(null);
  // Store fetched vehicle metadata (if requested by the user)
  const [vehicleData, setVehicleData] = useState<any>(null);
  // Boolean to toggle loading UI specifically for the vehicle data fetch
  const [loadingVehicle, setLoadingVehicle] = useState(false);
  // Store any error messages that occur during the pipeline
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Handle file input changes
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Check if the user actually chose a file
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      // Reset all application state variables when a new file is chosen
      setTaskId(null);
      setStatus('IDLE');
      setResultData(null);
      setVehicleData(null);
      setErrorMsg(null);
    }
  };

  // Trigger the ML pipeline
  const startProcessing = async () => {
    if (!selectedFile) return;
    setStatus('UPLOADING');
    setErrorMsg(null);
    // Bundle the file into a FormData object for multipart/form-data POST request
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Send the file to the FastAPI gateway backend
      const response = await fetch('/api/process-image', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      // Parse the resulting task ID from the JSON
      const data = await response.json();
      setTaskId(data.task_id);
      // Transition state to PENDING to trigger polling mechanism
      setStatus('PENDING');
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to upload image');
      setStatus('ERROR');
    }
  };

  // useEffect hook to handle background polling of the Celery task ID
  useEffect(() => {
    let intervalId: number;

    const pollStatus = async () => {
      // Only poll if there is a task ID and the process is still pending
      if (!taskId || status !== 'PENDING') return;

      try {
        // Query the status endpoint using the task_id
        const response = await fetch(`/api/status/${taskId}`);
        if (!response.ok) throw new Error('Polling failed');
        const data: WorkerResponse = await response.json();

        // If the Celery task is finished successfully
        if (data.status === 'SUCCESS') {
          setStatus('SUCCESS');
          setResultData(data.data); // Store the payload displaying the ALPR results
        // If the Celery worker crashed or returned an exception
        } else if (data.status === 'FAILURE') {
          setStatus('ERROR');
          setErrorMsg(data.error || 'Worker failed processing');
        }
      } catch (err) {
        console.error(err);
      }
    };

    // If currently 'PENDING', execute pollStatus every 2 seconds
    if (status === 'PENDING') {
      intervalId = window.setInterval(pollStatus, 2000);
    }

    // Cleanup phase: clear the interval whenever dependencies change or component unmounts
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [taskId, status]);

  const fetchVehicleInfo = async () => {
    if (!resultData?.extracted_text) return;
    setLoadingVehicle(true);
    try {
      const response = await fetch(`/api/vehicle/${resultData.extracted_text}`);
      if (!response.ok) throw new Error('Failed to fetch vehicle data');
      const data = await response.json();
      setVehicleData(data);
    } catch (err: any) {
      alert(err.message || 'Error fetching vehicle info');
    } finally {
      setLoadingVehicle(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Auto-Adaptive License Plate Recognition</h1>
        <p>A detection-first ALPR pipeline demonstrating microservices architecture.</p>
      </header>
      
      <main className={styles.main}>
        <section className={styles.uploadSection}>
          <input type="file" accept="image/jpeg, image/png, image/jpg" onChange={handleFileChange} />
          <button onClick={startProcessing} disabled={!selectedFile || status === 'PENDING' || status === 'UPLOADING'}>
            {status === 'PENDING' ? 'Processing...' : status === 'UPLOADING' ? 'Uploading...' : 'Run ALPR Pipeline'}
          </button>
        </section>

        {errorMsg && <div className={styles.errorAlert}>{errorMsg}</div>}

        {status === 'PENDING' && (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Running ML Pipeline in the background... Please wait.</p>
          </div>
        )}

        {status === 'SUCCESS' && resultData && (
          <div className={styles.resultsContainer}>
            <div className={styles.infoBox}>
              <strong>Detection Preparation:</strong> {resultData.restoration_msg}
            </div>

            <section className={styles.sequentialSection}>
              <div className={styles.imageBlock}>
                <h3>Classifier Result</h3>
                <div className={styles.moduleStatus}>{resultData.restoration_msg}</div>
              </div>
              <div className={styles.imageBlock}>
                <h3>Restoration Applied</h3>
                {resultData.nafnet_image && (
                  <>
                    <img src={resultData.nafnet_image} alt="NAFNet Output" />
                    <div className={styles.moduleStatus}>NAFNet Applied</div>
                  </>
                )}
                {resultData.darkir_image && (
                  <>
                    <img src={resultData.darkir_image} alt="DarkIR Output" />
                    <div className={styles.moduleStatus}>DarkIR Applied</div>
                  </>
                )}
                {resultData.dehaze_image && (
                  <>
                    <img src={resultData.dehaze_image} alt="DeHaze Output" />
                    <div className={styles.moduleStatus}>DeHaze Applied</div>
                  </>
                )}
                {resultData.derain_image && (
                  <>
                    <img src={resultData.derain_image} alt="DeRain Output" />
                    <div className={styles.moduleStatus}>DeRain Applied</div>
                  </>
                )}
                {!resultData.nafnet_image && !resultData.darkir_image && !resultData.dehaze_image && !resultData.derain_image && (
                  <div className={styles.moduleStatus}>No restoration applied</div>
                )}
              </div>
              <div className={styles.imageBlock}>
                <h3>Detection Used</h3>
                {resultData.detection_used ? (
                  <img src={resultData.detection_used} alt="Detection Used" />
                ) : (
                  <div className={styles.moduleStatus}>Detection Used: Not available</div>
                )}
              </div>
            </section>

            <section className={styles.detectionSection}>
               <h2>License Plate Detection</h2>
               {resultData.annotated_image ? (
                 <div className={styles.detectionContent}>
                    <div className={styles.annotatedBox}>
                        <p>Detected on: <strong>{resultData.detection_source}</strong></p>
                        <img src={resultData.annotated_image} alt="Annotated" />
                    </div>
                    
                    <div className={styles.sequentialSection}>
                        <h3>License Plate Crop</h3>
                        {resultData.plate_crop && <img src={resultData.plate_crop} alt="Crop" className={styles.smallImg} />}
                        
                        <h3>Deblurred & Upscaled</h3>
                        {resultData.plate_upscaled && <img src={resultData.plate_upscaled} alt="Upscaled" className={styles.smallImg} />}
                    </div>
                        
                    <div className={styles.successBanner}>
                        Final Extracted Text: <code>{resultData.extracted_text}</code> (Conf: {resultData.confidence.toFixed(2)})
                    </div>
                        
                    {resultData.rto_metadata.state !== 'Unknown' && (
                       <div className={styles.infoBanner}>
                         📍 Registered In: {resultData.rto_metadata.state} (District: {resultData.rto_metadata.district_code})
                       </div>
                    )}
                        
                    <hr className={styles.divider} />
                    
                    <div className={styles.vehicleDetailsBox}>
                      <h3>Vehicle Details</h3>
                      <button className={styles.fetchButton} onClick={fetchVehicleInfo} disabled={loadingVehicle}>
                        {loadingVehicle ? 'Fetching...' : 'Fetch RegCheck API Data'}
                      </button>
                      
                      {vehicleData && vehicleData.valid && (
                        <div className={styles.vehicleList}>
                           <p><strong>Owner:</strong> {vehicleData.data.owner || 'N/A'}</p>
                           <p><strong>Make/Model:</strong> {vehicleData.data.make} {vehicleData.data.model}</p>
                           <p><strong>Engine/Fuel:</strong> {vehicleData.data.engine}cc | {vehicleData.data.fuel}</p>
                           <p><strong>Location:</strong> {vehicleData.data.location || 'N/A'}</p>
                           <p><strong>Registration Date:</strong> {vehicleData.data.registration_date || vehicleData.data.year}</p>
                           <p><strong>Insurance Expiry:</strong> {vehicleData.data.insurance || 'N/A'}</p>
                        </div>
                      )}
                      
                      {vehicleData && !vehicleData.valid && (
                          <div className={styles.errorAlert}>
                              Could not fetch vehicle data via RegCheck: {vehicleData.error || 'Unknown Error'}
                          </div>
                      )}
                    </div>
                 </div>
               ) : (
                  <div className={styles.warningAlert}>
                      No license plates detected on either the deblurred image, the DarkIR fallback, the DeHaze fallback, or the DeRain fallback image.
                  </div>
               )}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
