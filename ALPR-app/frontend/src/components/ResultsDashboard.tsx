import styles from '../App.module.css';
import type { AlprResultData, VehicleLookupResponse } from '../types/alpr';

interface ResultsDashboardProps {
  resultData: AlprResultData;
  vehicleData: VehicleLookupResponse | null;
  loadingVehicle: boolean;
  onFetchVehicleInfo: () => void;
}

function getRestorationImage(resultData: AlprResultData): { image: string | null; label: string } {
  if (resultData.nafnet_image)  return { image: resultData.nafnet_image,  label: 'NAFNet Applied' };
  if (resultData.darkir_image)  return { image: resultData.darkir_image,  label: 'DarkIR Applied' };
  if (resultData.dehaze_image)  return { image: resultData.dehaze_image,  label: 'DeHaze Applied' };
  if (resultData.derain_image)  return { image: resultData.derain_image,  label: 'DeRain Applied' };
  return { image: null, label: 'No restoration applied' };
}

export function ResultsDashboard({
  resultData,
  vehicleData,
  loadingVehicle,
  onFetchVehicleInfo,
}: ResultsDashboardProps) {
  const restoration = getRestorationImage(resultData);
  const conf = resultData.confidence;
  const confClass = conf >= 0.8 ? styles.confidenceHigh : styles.confidenceMed;

  return (
    <>
      {/* ── Header ────────────────────────────────── */}
      <div className={styles.resultsHeader}>
        <div className={styles.resultsBadge}>✓ Pipeline Complete</div>
        <h2 className={styles.resultsTitle}>Scan Results</h2>
      </div>

      {/* ── OCR Hero ──────────────────────────────── */}
      <div className={styles.ocrHero}>
        <div className={styles.ocrPlate}>{resultData.extracted_text || '—'}</div>
        <div className={styles.ocrMeta}>
          <div className={styles.ocrMetaRow}>
            <span className={styles.ocrMetaLabel}>Confidence</span>
            <span className={`${styles.confidenceBadge} ${confClass}`}>
              {(conf * 100).toFixed(1)}%
            </span>
          </div>
          <div className={styles.ocrMetaRow}>
            <span className={styles.ocrMetaLabel}>Source</span>
            <span>{resultData.detection_source}</span>
          </div>
          {resultData.rto_metadata.state !== 'Unknown' && (
            <div className={styles.ocrMetaRow}>
              <span className={styles.ocrMetaLabel}>Registered</span>
              <span>
                {resultData.rto_metadata.state} · {resultData.rto_metadata.district_code}
              </span>
            </div>
          )}
          <div className={styles.ocrMetaRow}>
            <span className={styles.ocrMetaLabel}>Restoration</span>
            <span>{restoration.label}</span>
          </div>
        </div>
      </div>

      {/* ── Pipeline Grid ─────────────────────────── */}
      <div className={styles.pipelineGrid}>
        {/* Left column */}
        <div className={styles.col}>
          <article className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardIcon}>🖼</span>
              <span className={styles.cardTitle}>Original Image</span>
            </div>
            {resultData.original_image ? (
              <img src={resultData.original_image} alt="Original Upload" />
            ) : (
              <p className={styles.mutedText}>Original image preview not available.</p>
            )}
          </article>

          <article className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardIcon}>✨</span>
              <span className={styles.cardTitle}>Restoration Output</span>
            </div>
            {restoration.image ? (
              <>
                <img src={restoration.image} alt="Restoration Output" />
                <span className={styles.imageTag}>✓ {restoration.label}</span>
              </>
            ) : (
              <p className={styles.mutedText}>{restoration.label}</p>
            )}
          </article>
        </div>

        {/* Right column */}
        <div className={styles.col}>
          <article className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardIcon}>🔬</span>
              <span className={styles.cardTitle}>Classifier Result</span>
            </div>
            <p className={styles.classifierMsg}>{resultData.restoration_msg}</p>
          </article>

          <article className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardIcon}>🎯</span>
              <span className={styles.cardTitle}>Image Used for Final Detection</span>
            </div>
            {resultData.detection_used ? (
              <img src={resultData.detection_used} alt="Image used for final detection" />
            ) : (
              <p className={styles.mutedText}>Detection image unavailable.</p>
            )}
          </article>
        </div>
      </div>

      {/* ── Bottom Grid ───────────────────────────── */}
      <div className={styles.bottomGrid}>
        <article className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.cardIcon}>📌</span>
            <span className={styles.cardTitle}>Annotated Detection</span>
          </div>
          {resultData.annotated_image ? (
            <img src={resultData.annotated_image} alt="Annotated detection" />
          ) : (
            <p className={styles.warningText}>
              No license plate detected from restoration and fallback route.
            </p>
          )}
        </article>

        <article className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.cardIcon}>🔍</span>
            <span className={styles.cardTitle}>Plate Crop &amp; Upscaled</span>
          </div>
          <div className={styles.plateRow}>
            <div>
              <p className={styles.plateRowLabel}>Detected Crop</p>
              {resultData.plate_crop ? (
                <img src={resultData.plate_crop} alt="License plate crop" />
              ) : (
                <p className={styles.mutedText}>No crop available.</p>
              )}
            </div>
            <div>
              <p className={styles.plateRowLabel}>Deblurred &amp; Upscaled</p>
              {resultData.plate_upscaled ? (
                <img src={resultData.plate_upscaled} alt="Upscaled plate" />
              ) : (
                <p className={styles.mutedText}>No upscaled image.</p>
              )}
            </div>
          </div>
        </article>
      </div>

      {/* ── Vehicle Details ───────────────────────── */}
      <article className={styles.vehicleCard}>
        <div className={styles.cardHeader}>
          <span className={styles.cardIcon}>🚗</span>
          <span className={styles.cardTitle}>Vehicle Details — RegCheck API</span>
        </div>

        <button
          className={styles.fetchBtn}
          onClick={onFetchVehicleInfo}
          disabled={loadingVehicle}
        >
          {loadingVehicle ? '⟳ Fetching…' : '⚡ Fetch RegCheck Data'}
        </button>

        {vehicleData && vehicleData.valid && vehicleData.data && (
          <div className={styles.vehicleGrid}>
            {[
              { label: 'Owner',             value: vehicleData.data.owner },
              { label: 'Make / Model',      value: `${vehicleData.data.make ?? ''} ${vehicleData.data.model ?? ''}`.trim() },
              { label: 'Engine / Fuel',     value: `${vehicleData.data.engine ?? '—'}cc · ${vehicleData.data.fuel ?? '—'}` },
              { label: 'Location',          value: vehicleData.data.location },
              { label: 'Registration Date', value: vehicleData.data.registration_date ?? vehicleData.data.year },
              { label: 'Insurance Expiry',  value: vehicleData.data.insurance },
            ].map(({ label, value }) => (
              <div key={label} className={styles.vehicleItem}>
                <span className={styles.vehicleItemLabel}>{label}</span>
                <span className={styles.vehicleItemValue}>{value || 'N/A'}</span>
              </div>
            ))}
          </div>
        )}

        {vehicleData && !vehicleData.valid && (
          <div className={styles.vehicleError}>
            ⚠ Could not fetch vehicle data: {vehicleData.error || 'Unknown error'}
          </div>
        )}
      </article>
    </>
  );
}
