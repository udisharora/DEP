import React, { useState } from 'react';
import { UploadCloud, Loader2, CheckCircle2 } from 'lucide-react';
import './Scanner.css';

const Scanner = () => {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [vehicleData, setVehicleData] = useState(null);
  const [isFetchingVehicle, setIsFetchingVehicle] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (uploadedFile) => {
    if (!uploadedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }
    setFile(uploadedFile);
    setError(null);
    setResults(null);
    setVehicleData(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
    };
    reader.readAsDataURL(uploadedFile);
  };

  const runPipeline = async () => {
    if (!file) return;

    setIsLoading(true);
    setResults(null);
    setError(null);
    setVehicleData(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Pipeline execution failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError('An error occurred during ALPR analysis. Check server logs.');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchVehicleData = async (plateText) => {
    setIsFetchingVehicle(true);
    try {
      const response = await fetch(`http://localhost:8000/vehicle_info/${encodeURIComponent(plateText)}`);
      if (!response.ok) throw new Error('Failed to fetch vehicle metadata');
      const data = await response.json();
      setVehicleData(data);
    } catch (err) {
      console.error(err);
      alert(err.message);
    } finally {
      setIsFetchingVehicle(false);
    }
  };

  return (
    <div className="container scanner-container animate-fade-in">
      <header className="scanner-header">
        <h1>Intelligence Pipeline</h1>
        <p>Drop a vehicle image below to initialize the multi-stage ALPR sequence.</p>
      </header>

      {!results && (
        <div 
          className="upload-area glass-panel"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {imagePreview ? (
            <div className="preview-container">
              <img src={imagePreview} alt="Preview" className="image-preview" />
              <div className="preview-actions">
                <button className="btn-secondary" onClick={() => { setFile(null); setImagePreview(null); }}>Clear</button>
                <button className="btn-primary" onClick={runPipeline} disabled={isLoading}>
                  {isLoading ? <><Loader2 className="spinner" /> Processing...</> : 'Run ALPR Pipeline'}
                </button>
              </div>
            </div>
          ) : (
            <div className="upload-prompt">
              <UploadCloud size={48} className="upload-icon" />
              <h2>Drag & Drop vehicle image</h2>
              <p>or click to browse local files (JPG, PNG)</p>
              <input 
                type="file" 
                className="file-input" 
                accept="image/*" 
                onChange={handleFileInput}
              />
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="error-card glass-panel">
          {error}
        </div>
      )}

      {/* Results Section */}
      {results && (
        <div className="results-section animate-fade-in">
          <div className="results-header">
            <button className="btn-secondary" onClick={() => setResults(null)}>Analyze New Image</button>
            <div className="badge-status">
              <CheckCircle2 size={16} /> Pipeline Complete ({results.restoration_msg})
            </div>
          </div>

          {/* Restoration Grid */}
          <h3>Restoration Nodes</h3>
          <div className="images-grid">
            <div className="image-card glass-panel">
              <img src={`data:image/jpeg;base64,${results.images.original}`} alt="Original" />
              <div className="tag">1. Original</div>
            </div>
            <div className="image-card glass-panel">
              <img src={`data:image/jpeg;base64,${results.images.deblurred}`} alt="Deblurred" />
              <div className="tag">2. NAFNet Deblur</div>
            </div>
            <div className="image-card glass-panel">
              <img src={`data:image/jpeg;base64,${results.images.darkir}`} alt="DarkIR" />
              <div className="tag">3. DarkIR</div>
            </div>
            <div className="image-card glass-panel">
              <img src={`data:image/jpeg;base64,${results.images.dehaze}`} alt="Dehaze" />
              <div className="tag">4. DeHaze</div>
            </div>
            <div className="image-card glass-panel">
              <img src={`data:image/jpeg;base64,${results.images.derain}`} alt="Derain" />
              <div className="tag">5. DeRain</div>
            </div>
            <div className="image-card glass-panel highlight-border">
              <img src={`data:image/jpeg;base64,${results.images.detection_used}`} alt="Detection Used" />
              <div className="tag accent-bg">6. Detection Source</div>
            </div>
          </div>

          {/* Extraction Analysis */}
          {results.plate_detected ? (
            <div className="extraction-analysis">
              <div className="plate-card glass-panel">
                <h3>Final Extraction</h3>
                <div className="plate-display">
                  <h2 className="plate-text">{results.plate_results.text}</h2>
                  <div className="conf-bar">
                    <div className="conf-fill" style={{ width: `${results.plate_results.confidence * 100}%` }}></div>
                  </div>
                  <span>{(results.plate_results.confidence * 100).toFixed(1)}% Confidence</span>
                </div>
                
                <div className="plate-crop-row">
                  <div>
                    <span className="small-label">Detection Bounding Box</span>
                    <img src={`data:image/jpeg;base64,${results.images.annotated}`} alt="Detected plate box" className="roi-img" />
                  </div>
                  <div>
                    <span className="small-label">Super-Resolved Crop</span>
                    <img src={`data:image/jpeg;base64,${results.plate_results.best_sr_image}`} alt="Deblurred plate" className="roi-img" />
                  </div>
                </div>
              </div>

              {!vehicleData && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '16px' }}>
                  <button 
                    className="btn-primary" 
                    onClick={() => fetchVehicleData(results.plate_results.text)}
                    disabled={isFetchingVehicle}
                  >
                    {isFetchingVehicle ? <Loader2 className="spinner" /> : 'Fetch RegCheck Data'}
                  </button>
                </div>
              )}

              {vehicleData && vehicleData.valid && (
                <div className="api-card glass-panel" style={{ marginTop: '24px' }}>
                  <h3>RegCheck Fetch</h3>
                  <div className="info-grid">
                    <div>
                      <span className="info-label">Make & Model</span>
                      <span className="info-value">{vehicleData.data.make} {vehicleData.data.model}</span>
                    </div>
                    <div>
                      <span className="info-label">Owner</span>
                      <span className="info-value">{vehicleData.data.owner}</span>
                    </div>
                    <div>
                      <span className="info-label">Location</span>
                      <span className="info-value">{vehicleData.data.location}</span>
                    </div>
                    <div>
                      <span className="info-label">Engine/Fuel</span>
                      <span className="info-value">{vehicleData.data.engine}cc {vehicleData.data.fuel}</span>
                    </div>
                  </div>
                </div>
              )}
              
              {vehicleData && !vehicleData.valid && (
                <div className="error-card glass-panel" style={{ marginTop: '24px' }}>
                  Could not fetch vehicle metadata via RegCheck.
                </div>
              )}
            </div>
          ) : (
            <div className="error-card glass-panel">
              No license plate could be detected in any of the restoration states.
            </div>
          )}

        </div>
      )}
    </div>
  );
};

export default Scanner;
