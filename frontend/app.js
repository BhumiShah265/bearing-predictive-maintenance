/* app.js - Chart.js & FastAPI Endpoint Integration */

let timeChart = null;
let fftChart = null;

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  setupDragAndDrop();

  // Load default normal sample on startup
  loadSample('normal');
});

function initCharts() {
  const ctxTime = document.getElementById('timeChart').getContext('2d');
  timeChart = new Chart(ctxTime, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Raw Vibration Waveform (g)',
        data: [],
        borderColor: '#06b6d4',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Time (seconds)', color: '#9ca3af' },
          ticks: { color: '#9ca3af', maxTicksLimit: 8 },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          title: { display: true, text: 'Acceleration (g)', color: '#9ca3af' },
          ticks: { color: '#9ca3af' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#f3f4f6' } }
      }
    }
  });

  const ctxFFT = document.getElementById('fftChart').getContext('2d');
  fftChart = new Chart(ctxFFT, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'FFT Magnitude Spectrum |X(f)|',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.15)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Frequency (Hz)', color: '#9ca3af' },
          ticks: { color: '#9ca3af', maxTicksLimit: 10 },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          title: { display: true, text: 'Magnitude', color: '#9ca3af' },
          ticks: { color: '#9ca3af' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#f3f4f6' } }
      }
    }
  });
}

function setupDragAndDrop() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });
}

async function loadSample(faultType) {
  try {
    const response = await fetch(`/api/sample/${faultType}`);
    if (!response.ok) throw new Error('Sample fetch failed');
    const data = await response.json();
    renderAnalysisResult(data);
  } catch (err) {
    console.error(err);
    alert(`Failed to load ${faultType} sample.`);
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/predict-file', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      const errJson = await response.json();
      throw new Error(errJson.detail || 'File prediction failed');
    }
    const data = await response.json();
    renderAnalysisResult(data);
  } catch (err) {
    console.error(err);
    alert(`File upload error: ${err.message}`);
  }
}

function renderAnalysisResult(data) {
  // 1. Diagnosis Header
  const faultResult = document.getElementById('faultResult');
  const confidenceVal = document.getElementById('confidenceVal');
  
  faultResult.textContent = data.predicted_condition;
  confidenceVal.textContent = `Confidence: ${data.confidence_percentage}`;

  if (data.predicted_condition === 'Normal') {
    faultResult.className = 'fault-display normal';
  } else {
    faultResult.className = 'fault-display fault';
  }

  // 2. Probabilities Breakdown
  const probContainer = document.getElementById('probContainer');
  probContainer.innerHTML = '';
  for (const [cls, prob] of Object.entries(data.class_probabilities)) {
    const percent = (prob * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <span class="prob-label">${cls}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width: ${percent}%"></div>
      </div>
      <span>${percent}%</span>
    `;
    probContainer.appendChild(item);
  }

  // 3. Physical Features Grid
  const feats = data.signal_features;
  document.getElementById('featRms').textContent = feats.rms.toFixed(4);
  document.getElementById('featKurt').textContent = feats.kurtosis.toFixed(2);
  document.getElementById('featCrest').textContent = feats.crest_factor.toFixed(2);
  document.getElementById('featP2P').textContent = feats.peak_to_peak.toFixed(4);
  document.getElementById('featDomFreq').textContent = `${feats.dominant_frequency.toFixed(1)} Hz`;
  document.getElementById('featCentroid').textContent = `${feats.spectral_centroid.toFixed(1)} Hz`;

  // 4. Update Time Chart
  timeChart.data.labels = data.time_series_data.time;
  timeChart.data.datasets[0].data = data.time_series_data.amplitude;
  timeChart.update();

  // 5. Update FFT Chart
  fftChart.data.labels = data.fft_data.frequencies;
  fftChart.data.datasets[0].data = data.fft_data.amplitudes;
  fftChart.update();

  // 6. Render Spectrogram Placeholder Matrix Summary
  const spec = data.spectrogram_data;
  const specContainer = document.getElementById('specContainer');
  specContainer.innerHTML = `
    <div style="font-size: 0.85rem; color: #9ca3af; text-align: center; padding-top: 1rem;">
      <p><strong>STFT Time-Frequency Matrix Loaded:</strong> ${spec.frequencies.length} Frequency Bins × ${spec.times.length} Time Windows</p>
      <p style="margin-top: 0.5rem;">Peak Spectral Energy Concentration: <strong>${Math.max(...spec.z_values.flat()).toFixed(2)} dB</strong></p>
    </div>
  `;
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  event.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}
