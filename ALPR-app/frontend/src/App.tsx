import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import styles from './App.module.css';
import { UploadWorkspace } from './components/UploadWorkspace';
import { ResultsDashboard } from './components/ResultsDashboard';
import { getTaskStatus, getVehicleInfo, startAlprTask } from './lib/api';
import type { AlprResultData, PipelineStatus, VehicleLookupResponse } from './types/alpr';

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<PipelineStatus>('IDLE');
  const [resultData, setResultData] = useState<AlprResultData | null>(null);
  const [vehicleData, setVehicleData] = useState<VehicleLookupResponse | null>(null);
  const [loadingVehicle, setLoadingVehicle] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const resultsRef = useRef<HTMLElement | null>(null);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setTaskId(null);
      setStatus('IDLE');
      setResultData(null);
      setVehicleData(null);
      setErrorMsg(null);
    }
  };

  const startProcessing = async () => {
    if (!selectedFile) return;
    setStatus('UPLOADING');
    setErrorMsg(null);
    try {
      const data = await startAlprTask(selectedFile);
      setTaskId(data.task_id);
      setStatus('PENDING');
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to upload image');
      setStatus('ERROR');
    }
  };

  useEffect(() => {
    let intervalId: number;

    const pollStatus = async () => {
      if (!taskId || status !== 'PENDING') return;
      try {
        const data = await getTaskStatus(taskId);
        if (data.status === 'SUCCESS') {
          setStatus('SUCCESS');
          setResultData(data.data || null);
          setTimeout(() => {
            resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 100);
        } else if (data.status === 'FAILURE') {
          setStatus('ERROR');
          setErrorMsg(data.error || 'Worker failed processing');
        }
      } catch (err) {
        console.error(err);
      }
    };

    if (status === 'PENDING') {
      intervalId = window.setInterval(pollStatus, 2000);
    }

    return () => { if (intervalId) clearInterval(intervalId); };
  }, [taskId, status]);

  const fetchVehicleInfo = async () => {
    if (!resultData?.extracted_text) return;
    setLoadingVehicle(true);
    try {
      const data = await getVehicleInfo(resultData.extracted_text);
      setVehicleData(data);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Error fetching vehicle info');
    } finally {
      setLoadingVehicle(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.gridBg} aria-hidden="true" />
      <div className={styles.orb1} aria-hidden="true" />
      <div className={styles.orb2} aria-hidden="true" />

      <nav className={styles.navbar}>
        <a href="#" className={styles.navBrand}>
          <span className={styles.navLogo}>LP</span>
          ALPR<span className={styles.navAccent}>Vision</span>
        </a>
        <button 
          className={styles.themeToggle} 
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </nav>

      <main className={styles.mainContainer}>
        {/* Spread Layout Hero */}
        <section className={styles.heroSection}>
          <span className={styles.watermark} aria-hidden="true">
            alprvision
          </span>

          <div className={styles.heroContent}>
            <div className={styles.heroBadge}>
              <span className={styles.heroBadgeDot} />
              Powered by YOLO + TrOCR + NAFNet
            </div>

            <h1 className={styles.heroTitle}>
              License Plate Recognition <span className={styles.heroHighlight}>made Instant</span> & Accurate
            </h1>

            <p className={styles.heroDesc}>
              Upload a vehicle image. Our pipeline automatically detects conditions,
              restores quality, and extracts the license plate with high accuracy.
            </p>
          </div>

          <div className={styles.uploadContainer}>
            <UploadWorkspace
              status={status}
              selectedFileName={selectedFile?.name || null}
              onFileChange={handleFileChange}
              onStart={startProcessing}
              errorMsg={errorMsg}
            />
          </div>
        </section>

        {/* Results */}
        {status === 'SUCCESS' && resultData && (
          <section
            ref={resultsRef as React.RefObject<HTMLElement>}
            className={styles.resultsSection}
          >
            <ResultsDashboard
              resultData={resultData}
              vehicleData={vehicleData}
              loadingVehicle={loadingVehicle}
              onFetchVehicleInfo={fetchVehicleInfo}
            />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
