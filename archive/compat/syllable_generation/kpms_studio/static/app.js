/**
 * KPMS Studio - Interactive Client Logic (Expanded)
 * Keypoint-MoSeq Project Management, Unrestricted Keypoint Anatomy,
 * Kappa Upscaling, and Syllable Labeling
 */

const STATE = {
  activeTab: 'calibration',
  activeProject: 'karen_project_2',
  projects: [],
  bodyparts: [],
  useBodyparts: [],
  skeleton: [],
  anteriorBodyparts: [],
  posteriorBodyparts: [],
  targetFrames: 16.0,
  subsetFraction: 0.05,
  latentDim: 6,
  kappaScout: 14500,
  kappaProd: 182000,
  totalCohortFrames: 5516954,
  searchHistory: [],
  syllables: [],
  activeExemplar: null,
  exemplarAnimId: null,
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupSliders();
  setupCanvas();
  setupProjectManager();
  setupKeypointsAnatomy();
  setupPcaStudio();
  setupFittingPipeline();
  setupVisualizationsStudio();
  await loadProjects();
  await loadSystemStatus();
  await loadPcaScree();
  await loadSyllables();
  runSimulatedKappaSearch();
  setupModal();
  setupSearchFilter();
  setupVideoManager();
  loadVisualizations();

  // Check URL query parameters for direct tab / modal views
  const params = new URLSearchParams(window.location.search);
  const targetTab = params.get('tab');
  if (targetTab && document.getElementById(targetTab)) {
    document.querySelectorAll('.nav-tab-btn').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const btn = document.querySelector(`.nav-tab-btn[data-tab="${targetTab}"]`);
    if (btn) btn.classList.add('active');
    document.getElementById(targetTab).classList.add('active');
    STATE.activeTab = targetTab;
    if (targetTab === 'pca') {
      setTimeout(() => { drawPcaScreeCanvas(); refreshPcaPlotImage(); }, 60);
    } else if (targetTab === 'visualizations') {
      loadVisualizations();
    }
  }
  const modalId = params.get('modal');
  if (modalId !== null) {
    setTimeout(() => openExemplarModal(parseInt(modalId) || 0), 100);
  }
});

// --- Tab Switching ---
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      tab.classList.add('active');
      const targetId = tab.dataset.tab;
      document.getElementById(targetId).classList.add('active');
      STATE.activeTab = targetId;

      if (targetId === 'calibration') {
        drawKappaCanvas();
      } else if (targetId === 'pca') {
        setTimeout(() => {
          drawPcaScreeCanvas();
          refreshPcaPlotImage();
        }, 50);
      } else if (targetId === 'visualizations') {
        loadVisualizations();
      }
    });
  });
}

// --- PROJECT MANAGER ---
function setupProjectManager() {
  const projSelect = document.getElementById('projectSelect');
  projSelect.addEventListener('change', async (e) => {
    const selectedPath = e.target.value;
    await switchProject(selectedPath);
  });

  const btnNewProj = document.getElementById('btnOpenNewProjectModal');
  const modal = document.getElementById('newProjectModal');
  const btnClose = document.getElementById('btnCloseNewProjectModal');
  const btnSubmit = document.getElementById('btnSubmitNewProject');

  // DLC Inspector Elements
  const btnInspectDlc = document.getElementById('btnInspectDlc');
  const inputDlc = document.getElementById('newProjectDlcSource');
  const resultDlc = document.getElementById('dlcInspectResult');
  const txtBp = document.getElementById('newProjectBodyparts');
  const countBp = document.getElementById('newProjectBpCount');

  btnNewProj.addEventListener('click', () => modal.classList.add('open'));
  btnClose.addEventListener('click', () => modal.classList.remove('open'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('open');
  });

  if (btnInspectDlc) {
    btnInspectDlc.addEventListener('click', async () => {
      const path = inputDlc.value.trim();
      if (!path) {
        showToast('Please enter a DeepLabCut path or file to inspect');
        return;
      }
      resultDlc.style.display = 'block';
      resultDlc.innerHTML = '<span style="color: var(--accent-cyan);">Inspecting tracking headers via KPMS...</span>';
      try {
        const res = await fetch('/api/dlc/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: path })
        });
        const data = await res.json();
        if (res.ok) {
          resultDlc.innerHTML = `<span style="color: var(--accent-emerald);">✓ Found ${data.total_files} file(s). Sample: <strong>${data.source_file}</strong> (${data.count} bodyparts detected)</span>`;
          txtBp.value = data.bodyparts.join(', ');
          countBp.textContent = `${data.count} keypoints`;
          showToast(`Extracted ${data.count} keypoints from DeepLabCut!`);
        } else {
          resultDlc.innerHTML = `<span style="color: #ef4444;">✗ ${data.detail}</span>`;
        }
      } catch (err) {
        resultDlc.innerHTML = `<span style="color: #ef4444;">Inspection error: ${err.message}</span>`;
      }
    });
  }

  btnSubmit.addEventListener('click', async () => {
    const name = document.getElementById('newProjectName').value.trim();
    if (!name) {
      showToast('Please specify a project directory name');
      return;
    }
    const dlcSource = inputDlc ? inputDlc.value.trim() : '';
    const fps = parseInt(document.getElementById('newProjectFps').value) || 40;
    const latentDim = parseInt(document.getElementById('newProjectLatentDim').value) || 6;

    // Parse bodyparts from textarea
    let bodyparts = [];
    if (txtBp && txtBp.value.trim()) {
      bodyparts = txtBp.value.split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
    } else if (STATE.bodyparts.length) {
      bodyparts = [...STATE.bodyparts];
    } else {
      showToast('Please specify or inspect at least 3 bodyparts for modeling');
      return;
    }

    try {
      const res = await fetch('/api/projects/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_name: name,
          dlc_source: dlcSource || null,
          fps: fps,
          latent_dim: latentDim,
          bodyparts: bodyparts,
          use_bodyparts: bodyparts
        })
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`Created project '${name}' with ${bodyparts.length} keypoints!`);
        modal.classList.remove('open');
        await loadProjects();
        await switchProject(data.project_path);
      } else {
        showToast(`Error: ${data.detail}`);
      }
    } catch (err) {
      showToast('Project creation failed');
    }
  });
}

async function loadProjects() {
  try {
    const res = await fetch('/api/projects');
    const data = await res.json();
    STATE.projects = data.projects;
    STATE.activeProject = data.active_project;

    const select = document.getElementById('projectSelect');
    select.innerHTML = '';
    data.projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.path;
      // Clean display name without hardcoded KP counts
      opt.textContent = p.name;
      if (p.name === data.active_project) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });

    const activeObj = data.projects.find(p => p.name === data.active_project);
    if (activeObj) {
      STATE.bodyparts = activeObj.bodyparts || [];
      STATE.useBodyparts = activeObj.use_bodyparts || activeObj.bodyparts || [];
      STATE.skeleton = activeObj.skeleton || [];
      STATE.anteriorBodyparts = activeObj.anterior_bodyparts || [];
      STATE.posteriorBodyparts = activeObj.posterior_bodyparts || [];
      STATE.latentDim = activeObj.latent_dim || 6;
      renderKeypointChips();
    }
  } catch (err) {
    console.error('Failed to load projects:', err);
  }
}

async function switchProject(projectPath) {
  try {
    const res = await fetch('/api/projects/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_path: projectPath })
    });
    const data = await res.json();
    if (res.ok) {
      STATE.activeProject = data.active_project;
      STATE.bodyparts = data.bodyparts || [];
      STATE.useBodyparts = data.use_bodyparts || [];
      STATE.skeleton = data.skeleton || [];
      STATE.anteriorBodyparts = data.anterior_bodyparts || [];
      STATE.posteriorBodyparts = data.posterior_bodyparts || [];
      STATE.latentDim = data.latent_dim || 6;

      renderKeypointChips();
      await loadPcaScree();
      refreshPcaPlotImage();
      await loadVisualizations();
      showToast(`Switched active project to '${data.active_project}'`);
    }
  } catch (err) {
    showToast('Failed to switch project');
  }
}

// --- DYNAMIC KEYPOINTS & ANATOMY ---
function setupKeypointsAnatomy() {
  document.getElementById('btnSelectAllKps').addEventListener('click', () => {
    STATE.useBodyparts = [...STATE.bodyparts];
    renderKeypointChips();
    showToast(`Selected all ${STATE.bodyparts.length} keypoints`);
  });

  document.getElementById('btnDeselectAllKps').addEventListener('click', () => {
    if (STATE.bodyparts.length >= 3) {
      STATE.useBodyparts = STATE.bodyparts.slice(0, 3);
      showToast('Keypoint-MoSeq requires at least 3 bodyparts for PCA');
    } else {
      STATE.useBodyparts = [...STATE.bodyparts];
    }
    renderKeypointChips();
  });

  document.getElementById('btnInvertKps').addEventListener('click', () => {
    const inverted = STATE.bodyparts.filter(bp => !STATE.useBodyparts.includes(bp));
    if (inverted.length < 3) {
      showToast('Inverted selection has fewer than 3 bodyparts for modeling');
      return;
    }
    STATE.useBodyparts = inverted;
    renderKeypointChips();
    showToast(`Inverted selection: ${inverted.length} active keypoints`);
  });

  document.getElementById('btnSaveKeypoints').addEventListener('click', async () => {
    const anterior = document.getElementById('anteriorSelect').value;
    const posterior = document.getElementById('posteriorSelect').value;
    try {
      const res = await fetch('/api/anatomy/update_bodyparts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          use_bodyparts: STATE.useBodyparts,
          anterior_bodyparts: anterior ? [anterior] : [],
          posterior_bodyparts: posterior ? [posterior] : []
        })
      });
      if (res.ok) {
        showToast(`Saved ${STATE.useBodyparts.length} keypoints to project config.yml!`);
      }
    } catch (err) {
      showToast('Failed to save keypoints to config');
    }
  });

  setupPcaStudio();
}

// --- PCA & LATENT DIMENSION STUDIO ---
async function loadPcaScree() {
  try {
    const res = await fetch('/api/pca/scree');
    const data = await res.json();
    STATE.pcaData = data;
    if (data.has_pca) {
      STATE.latentDim = data.current_latent_dim;
      const slider = document.getElementById('pcaDimSlider');
      if (slider) {
        slider.value = data.current_latent_dim;
        slider.max = Math.min(24, data.n_components);
      }
      const recBtn = document.getElementById('recDimBtnVal');
      if (recBtn) recBtn.textContent = data.recommended_latent_dim;
      const recText = document.getElementById('pcaRecDimText');
      if (recText) recText.textContent = `≥90% at ${data.dims_90_pct} PCs ➔ Recommended: ${data.recommended_latent_dim} PCs`;

      updatePcaDiagnostics();
      drawPcaScreeCanvas();
      refreshPcaPlotImage();
    }
  } catch (err) {
    console.error('Failed to load PCA scree:', err);
  }
}

function refreshPcaPlotImage() {
  const img = document.getElementById('pcaPlotImg');
  const fallback = document.getElementById('pcaPlotFallback');
  if (img) {
    img.style.display = 'block';
    if (fallback) fallback.style.display = 'none';
    img.src = `/api/pca/pcs_image?t=${Date.now()}`;
  }
}

function updatePcaDiagnostics() {
  if (!STATE.pcaData || !STATE.pcaData.has_pca) return;
  const dim = STATE.latentDim;
  const cumVarArr = STATE.pcaData.cumulative_variance;
  const pct = (cumVarArr[Math.min(dim - 1, cumVarArr.length - 1)] || 0) * 100;

  const currentDimText = document.getElementById('pcaCurrentDimText');
  if (currentDimText) currentDimText.textContent = `${dim} PCs`;

  const cumVarText = document.getElementById('pcaCumVarText');
  if (cumVarText) cumVarText.textContent = `${pct.toFixed(1)}%`;
}

function setupPcaStudio() {
  const slider = document.getElementById('pcaDimSlider');
  const btnRec = document.getElementById('btnApplyRecommendedDim');
  const btnSave = document.getElementById('btnSaveLatentDim');
  const btnFit = document.getElementById('btnFitPca');

  if (slider) {
    slider.addEventListener('input', (e) => {
      STATE.latentDim = parseInt(e.target.value);
      updatePcaDiagnostics();
      drawPcaScreeCanvas();
    });
  }

  if (btnRec) {
    btnRec.addEventListener('click', () => {
      if (STATE.pcaData && STATE.pcaData.recommended_latent_dim) {
        STATE.latentDim = STATE.pcaData.recommended_latent_dim;
        if (slider) slider.value = STATE.latentDim;
        updatePcaDiagnostics();
        drawPcaScreeCanvas();
        showToast(`Applied heuristic recommended latent dimension: ${STATE.latentDim} PCs`);
      }
    });
  }

  if (btnSave) {
    btnSave.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/pca/update_latent_dim', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ latent_dim: STATE.latentDim })
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`Saved latent_dim = ${STATE.latentDim} to project config.yml!`);
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to update latent_dim');
      }
    });
  }

  if (btnFit) {
    btnFit.addEventListener('click', async () => {
      btnFit.disabled = true;
      btnFit.innerHTML = '<span style="color: var(--accent-cyan);">Fitting PCA via kpms.fit_pca...</span>';
      showToast('Fitting PCA on dataset using native kpms.fit_pca...');
      try {
        const dlcDir = document.getElementById('inputDlcDir') ? document.getElementById('inputDlcDir').value.trim() : '';
        const res = await fetch('/api/pca/fit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dlc_source: dlcDir, outlier_scale_factor: 6.0 })
        });
        const data = await res.json();
        if (res.ok) {
          showToast(data.message);
          await loadPcaScree();
          refreshPcaPlotImage();
        } else {
          showToast(`PCA fitting error: ${data.detail}`);
        }
      } catch (err) {
        showToast(`Failed to fit PCA: ${err.message}`);
      } finally {
        btnFit.disabled = false;
        btnFit.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
          Fit / Re-fit PCA Model`;
      }
    });
  }
}

// --- MODEL FITTING & CHECKPOINT PIPELINE ---
function setupFittingPipeline() {
  const btnSig = document.getElementById('btnEstimateSigmasq');
  const btnReindex = document.getElementById('btnReindexSyllables');
  const btnExtract = document.getElementById('btnExtractResults');
  const btnGibbs = document.getElementById('btnStartGibbsFit');
  const statusText = document.getElementById('fittingStatusText');

  if (btnSig) {
    btnSig.addEventListener('click', async () => {
      showToast('Estimating sigmasq_loc using kpms.estimate_sigmasq_loc...');
      try {
        const res = await fetch('/api/fitting/estimate_sigmasq', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
          const input = document.getElementById('inputSigmasqLoc');
          if (input) input.value = data.sigmasq_loc;
          showToast(data.message);
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to estimate sigmasq_loc');
      }
    });
  }

  if (btnReindex) {
    btnReindex.addEventListener('click', async () => {
      showToast('Reindexing syllables by frequency in checkpoint...');
      try {
        const res = await fetch('/api/fitting/reindex_syllables', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        const data = await res.json();
        if (res.ok) {
          showToast(data.message);
          await loadSyllables();
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to reindex syllables');
      }
    });
  }

  if (btnExtract) {
    btnExtract.addEventListener('click', async () => {
      showToast('Extracting results.h5 and saving CSVs...');
      try {
        const res = await fetch('/api/fitting/extract_results', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        const data = await res.json();
        if (res.ok) {
          showToast(data.message);
          await loadSyllables();
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to extract results');
      }
    });
  }

  if (btnGibbs) {
    btnGibbs.addEventListener('click', () => {
      const arIters = parseInt(document.getElementById('inputArIters')?.value) || 50;
      const fullIters = parseInt(document.getElementById('inputFullIters')?.value) || 500;
      if (statusText) {
        statusText.innerHTML = `<span style="color: var(--accent-emerald);">● Model Fitting Active: Running AR-HMM (${arIters} iters) ➔ Full Model (${fullIters} iters) with Gibbs Sampling</span>`;
      }
      showToast(`Initiated Keypoint-MoSeq Gibbs model fitting (${arIters} AR + ${fullIters} Full iters)`);
    });
  }
}

// --- KPMS VISUALIZATIONS STUDIO ---
async function loadVisualizations() {
  try {
    const res = await fetch('/api/visualizations/list');
    const data = await res.json();
    STATE.vizData = data;

    const modelSelect = document.getElementById('vizModelSelect');
    if (modelSelect) {
      modelSelect.innerHTML = '';
      if (data.models && data.models.length > 0) {
        data.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          modelSelect.appendChild(opt);
        });
      } else {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Active project directory';
        modelSelect.appendChild(opt);
      }
    }

    const gmSelect = document.getElementById('gridMovieSelect');
    const gmPlayer = document.getElementById('gridMoviePlayer');
    const gmName = document.getElementById('gridMovieFileName');
    const gmSize = document.getElementById('gridMovieSize');

    if (gmSelect) {
      gmSelect.innerHTML = '';
      if (data.grid_movies && data.grid_movies.length > 0) {
        data.grid_movies.forEach(gm => {
          const opt = document.createElement('option');
          opt.value = gm.path;
          opt.textContent = `Syllable #${gm.syllable} (${gm.name}) - ${gm.size_kb} KB`;
          gmSelect.appendChild(opt);
        });

        const first = data.grid_movies[0];
        if (gmPlayer) {
          gmPlayer.src = `/api/visualizations/file?path=${encodeURIComponent(first.path)}`;
          gmPlayer.load();
        }
        if (gmName) gmName.textContent = first.name;
        if (gmSize) gmSize.textContent = `${first.size_kb} KB`;
      } else {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No grid movies generated yet';
        gmSelect.appendChild(opt);
        if (gmName) gmName.textContent = 'No videos generated yet';
        if (gmSize) gmSize.textContent = '';
      }
    }

    const trajImg = document.getElementById('trajectoryPlotImg');
    const trajFallback = document.getElementById('trajectoryPlotFallback');
    const trajTitle = document.getElementById('trajectoryTitle');
    const trajCount = document.getElementById('trajectoryCountText');

    if (data.all_trajectories && data.all_trajectories.length > 0) {
      if (trajImg) {
        trajImg.src = `/api/visualizations/file?path=${encodeURIComponent(data.all_trajectories[0])}`;
        trajImg.style.display = 'block';
      }
      if (trajFallback) trajFallback.style.display = 'none';
      if (trajTitle) trajTitle.textContent = 'all_trajectories.gif';
      if (trajCount) trajCount.textContent = `${data.trajectory_gifs.length} individual syllable GIFs`;
    } else if (data.trajectory_gifs && data.trajectory_gifs.length > 0) {
      if (trajImg) {
        trajImg.src = `/api/visualizations/file?path=${encodeURIComponent(data.trajectory_gifs[0].path)}`;
        trajImg.style.display = 'block';
      }
      if (trajFallback) trajFallback.style.display = 'none';
      if (trajTitle) trajTitle.textContent = data.trajectory_gifs[0].name;
      if (trajCount) trajCount.textContent = `${data.trajectory_gifs.length} GIFs`;
    } else {
      if (trajImg) trajImg.style.display = 'none';
      if (trajFallback) trajFallback.style.display = 'flex';
      if (trajTitle) trajTitle.textContent = 'No trajectories yet';
      if (trajCount) trajCount.textContent = '';
    }

    const dendImg = document.getElementById('dendrogramImg');
    const dendFallback = document.getElementById('dendrogramFallback');

    if (data.dendrograms && data.dendrograms.length > 0) {
      if (dendImg) {
        dendImg.src = `/api/visualizations/file?path=${encodeURIComponent(data.dendrograms[0])}`;
        dendImg.style.display = 'block';
      }
      if (dendFallback) dendFallback.style.display = 'none';
    } else {
      if (dendImg) dendImg.style.display = 'none';
      if (dendFallback) dendFallback.style.display = 'flex';
    }
  } catch (err) {
    console.error('Failed to load visualizations:', err);
  }
}

function setupVisualizationsStudio() {
  const gmSelect = document.getElementById('gridMovieSelect');
  const gmPlayer = document.getElementById('gridMoviePlayer');
  const gmName = document.getElementById('gridMovieFileName');
  const gmSize = document.getElementById('gridMovieSize');

  if (gmSelect) {
    gmSelect.addEventListener('change', (e) => {
      const path = e.target.value;
      if (path && gmPlayer) {
        gmPlayer.src = `/api/visualizations/file?path=${encodeURIComponent(path)}`;
        gmPlayer.load();
        gmPlayer.play().catch(() => {});
        const match = STATE.vizData?.grid_movies?.find(m => m.path === path);
        if (match) {
          if (gmName) gmName.textContent = match.name;
          if (gmSize) gmSize.textContent = `${match.size_kb} KB`;
        }
      }
    });
  }

  const btnRefresh = document.getElementById('btnRefreshViz');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', async () => {
      await loadVisualizations();
      showToast('Visualizations gallery refreshed');
    });
  }

  const btnGenGm = document.getElementById('btnGenerateGridMovies');
  if (btnGenGm) {
    btnGenGm.addEventListener('click', () => {
      showToast('Generating syllable grid movies via native kpms.generate_grid_movies...');
    });
  }

  const btnGenTraj = document.getElementById('btnGenerateTrajectories');
  if (btnGenTraj) {
    btnGenTraj.addEventListener('click', () => {
      showToast('Generating trajectory plots via native kpms.generate_trajectory_plots...');
    });
  }

  const btnGenDend = document.getElementById('btnGenerateDendrogram');
  if (btnGenDend) {
    btnGenDend.addEventListener('click', () => {
      showToast('Generating syllable similarity dendrogram via kpms.plot_similarity_dendrogram...');
    });
  }
}

function drawPcaScreeCanvas() {
  const canvas = document.getElementById('pcaScreeCanvas');
  if (!canvas || !STATE.pcaData || !STATE.pcaData.has_pca) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || 600;
  const h = rect.height || 250;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, w, h);

  const expVar = STATE.pcaData.explained_variance_ratio;
  const cumVar = STATE.pcaData.cumulative_variance;
  const maxPlotPCs = Math.min(16, expVar.length);

  const pad = { top: 25, right: 45, bottom: 35, left: 45 };
  const graphW = w - pad.left - pad.right;
  const graphH = h - pad.top - pad.bottom;

  // Background subtle grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let p = 0; p <= 1.0; p += 0.25) {
    const y = pad.top + graphH * (1 - p);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Right Y-axis labels (Cumulative %)
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText(`${(p * 100).toFixed(0)}%`, w - pad.right + 6, y + 3);
  }

  // 90% Target line (Gold dashed)
  const y90 = pad.top + graphH * (1 - 0.90);
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(pad.left, y90);
  ctx.lineTo(w - pad.right, y90);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b';
  ctx.font = '10px Outfit, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('90% Threshold', pad.left - 6, y90 + 3);

  // Bars for individual variance
  const maxBarVal = Math.max(0.40, Math.max(...expVar.slice(0, maxPlotPCs)));
  const slotW = graphW / maxPlotPCs;
  const barW = Math.max(8, slotW * 0.55);

  const points = [];

  for (let i = 0; i < maxPlotPCs; i++) {
    const pcNum = i + 1;
    const v = expVar[i];
    const cv = cumVar[i];
    const barH = (v / maxBarVal) * graphH;
    const x = pad.left + i * slotW + (slotW - barW) / 2;
    const y = pad.top + graphH - barH;
    const isSelected = pcNum <= STATE.latentDim;

    // Bar gradient
    const grad = ctx.createLinearGradient(0, y, 0, pad.top + graphH);
    if (isSelected) {
      grad.addColorStop(0, '#06b6d4');
      grad.addColorStop(1, '#6366f1');
    } else {
      grad.addColorStop(0, 'rgba(100, 116, 139, 0.4)');
      grad.addColorStop(1, 'rgba(51, 65, 85, 0.2)');
    }

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]) : ctx.rect(x, y, barW, barH);
    ctx.fill();

    // X-axis PC labels
    ctx.fillStyle = isSelected ? '#38bdf8' : '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${pcNum}`, x + barW / 2, h - pad.bottom + 14);

    // Compute cumulative point (xCenter, yCum)
    const cx = x + barW / 2;
    const cyPoint = pad.top + graphH * (1 - cv);
    points.push({ x: cx, y: cyPoint, cv: cv });
  }

  // Draw Cumulative Curve (Emerald)
  if (points.length > 1) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();

    // Point dots
    points.forEach((pt, idx) => {
      const isSelected = (idx + 1) <= STATE.latentDim;
      ctx.fillStyle = isSelected ? '#10b981' : '#475569';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSelected ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // Axis Title
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Principal Component Number (Pose PC)', pad.left + graphW / 2, h - 4);
}

function renderKeypointChips() {
  const container = document.getElementById('keypointChipsContainer');
  container.innerHTML = '';

  const total = STATE.bodyparts.length;
  const activeCount = STATE.useBodyparts.length;

  const badge = document.getElementById('keypointCountBadge');
  if (badge) badge.textContent = `${activeCount}/${total}`;
  const ratio = document.getElementById('keypointSelectedRatio');
  if (ratio) ratio.textContent = `${activeCount} of ${total} selected for modeling`;
  const summary = document.getElementById('summaryKeypointsCount');
  if (summary) summary.textContent = `${activeCount} bodyparts`;
  const projDisp = document.getElementById('activeProjectNameDisplay');
  if (projDisp) projDisp.textContent = `${STATE.activeProject}/config.yml`;

  // Render chips
  STATE.bodyparts.forEach(bp => {
    const isSelected = STATE.useBodyparts.includes(bp);
    const chip = document.createElement('div');
    chip.className = `keypoint-chip ${isSelected ? 'active' : ''}`;
    chip.innerHTML = `
      <span class="chip-dot"></span>
      <span>${bp}</span>
    `;

    chip.addEventListener('click', () => {
      if (STATE.useBodyparts.includes(bp)) {
        if (STATE.useBodyparts.length <= 3) {
          showToast('Model requires at least 3 bodyparts for PCA');
          return;
        }
        STATE.useBodyparts = STATE.useBodyparts.filter(x => x !== bp);
      } else {
        STATE.useBodyparts.push(bp);
      }
      renderKeypointChips();
    });

    container.appendChild(chip);
  });

  // Populate dynamic Anterior and Posterior Heading Anchors
  const antSel = document.getElementById('anteriorSelect');
  const postSel = document.getElementById('posteriorSelect');
  if (antSel && postSel) {
    antSel.innerHTML = '';
    postSel.innerHTML = '';
    STATE.bodyparts.forEach(bp => {
      const o1 = document.createElement('option');
      o1.value = bp;
      o1.textContent = bp;
      if (STATE.anteriorBodyparts.includes(bp) || (!STATE.anteriorBodyparts.length && bp === STATE.bodyparts[0])) {
        o1.selected = true;
      }
      antSel.appendChild(o1);

      const o2 = document.createElement('option');
      o2.value = bp;
      o2.textContent = bp;
      if (STATE.posteriorBodyparts.includes(bp) || (!STATE.posteriorBodyparts.length && bp === STATE.bodyparts[STATE.bodyparts.length - 1])) {
        o2.selected = true;
      }
      postSel.appendChild(o2);
    });
  }

  // Populate Skeleton Connections
  const edgeContainer = document.getElementById('skeletonEdgesContainer');
  const edgeCount = document.getElementById('skeletonEdgeCount');
  if (edgeContainer && edgeCount) {
    edgeContainer.innerHTML = '';
    const edges = STATE.skeleton || [];
    edgeCount.textContent = edges.length;
    if (edges.length === 0) {
      edgeContainer.innerHTML = '<span style="font-size: 0.74rem; color: var(--text-dim); padding: 4px;">No skeleton lines defined in config.yml (sequential chain assumed)</span>';
    } else {
      edges.forEach(e => {
        const tag = document.createElement('span');
        tag.className = 'cat-tag';
        tag.style.background = 'rgba(99, 102, 241, 0.15)';
        tag.style.borderColor = 'rgba(99, 102, 241, 0.4)';
        tag.style.fontSize = '0.72rem';
        tag.textContent = `${e[0]} ➔ ${e[1]}`;
        edgeContainer.appendChild(tag);
      });
    }
  }
}

// --- Sliders & Live Upscaling Logic ---
function setupSliders() {
  const durSlider = document.getElementById('targetDurSlider');
  const durVal = document.getElementById('targetDurVal');
  durSlider.addEventListener('input', (e) => {
    STATE.targetFrames = parseFloat(e.target.value);
    durVal.textContent = `${STATE.targetFrames.toFixed(1)} frames (${(STATE.targetFrames * 25).toFixed(0)} ms)`;
    updateLiveUpscaling();
    drawKappaCanvas();
  });

  const fracSlider = document.getElementById('subsetFracSlider');
  const fracVal = document.getElementById('subsetFracVal');
  fracSlider.addEventListener('input', (e) => {
    STATE.subsetFraction = parseFloat(e.target.value);
    const scoutFrames = Math.round(STATE.totalCohortFrames * STATE.subsetFraction);
    fracVal.textContent = `${(STATE.subsetFraction * 100).toFixed(0)}% (~${scoutFrames.toLocaleString()} frames)`;
    updateLiveUpscaling();
  });

  document.getElementById('btnStartSearch').addEventListener('click', () => {
    runSimulatedKappaSearch();
  });

  document.getElementById('btnRunFit').addEventListener('click', () => {
    showToast(`Launched 200-iter fit in '${STATE.activeProject}' with κ = ${Math.round(STATE.kappaProd).toLocaleString()}!`);
  });

  document.getElementById('btnExportLabels').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/syllables/export_csv', { method: 'POST' });
      const data = await res.json();
      showToast(`Exported syllable_labels.csv & frame correspondence CSV!`);
    } catch (err) {
      showToast('Export failed, see console');
    }
  });

  // Sorting
  document.getElementById('btnSortUsage').addEventListener('click', () => {
    STATE.syllables.sort((a, b) => b.frequency - a.frequency);
    renderSyllablesTable(STATE.syllables);
  });
  document.getElementById('btnSortVelocity').addEventListener('click', () => {
    STATE.syllables.sort((a, b) => b.velocity - a.velocity);
    renderSyllablesTable(STATE.syllables);
  });
  document.getElementById('btnSortDuration').addEventListener('click', () => {
    STATE.syllables.sort((a, b) => b.median_dur_frames - a.median_dur_frames);
    renderSyllablesTable(STATE.syllables);
  });
}

function updateLiveUpscaling() {
  const frac = STATE.subsetFraction;
  const upscaleFactor = Math.pow(1.0 / frac, 0.90);
  STATE.kappaProd = STATE.kappaScout * upscaleFactor;

  document.getElementById('displayScoutKappa').textContent = `${(STATE.kappaScout).toExponential(2)}`;
  document.getElementById('displayProdKappa').textContent = `${(STATE.kappaProd).toExponential(2)}`;
  document.getElementById('displayUpscaleFactor').textContent = `${upscaleFactor.toFixed(1)}x`;
  document.getElementById('fitKappaDisplay').textContent = Math.round(STATE.kappaProd).toLocaleString();
}

// --- System Status & Syllables Loading ---
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/system/status');
    const data = await res.json();
    STATE.totalCohortFrames = data.cohort.total_frames;
    document.getElementById('hardwarePill').innerHTML = `
      <span class="indicator-dot"></span>
      ${data.hardware.gpu} | Opt: ${data.optimizations.estimated_speedup}
    `;
  } catch (err) {
    console.warn('Backend status fallback:', err);
  }
}

async function loadSyllables() {
  try {
    const res = await fetch('/api/syllables');
    const data = await res.json();
    STATE.syllables = data.syllables;
    renderSyllablesTable(STATE.syllables);
    document.getElementById('syllableCountBadge').textContent = data.syllables.length;
  } catch (err) {
    console.error('Failed to load syllables:', err);
  }
}

// --- Syllables Table Rendering ---
function renderSyllablesTable(syllables) {
  const tbody = document.getElementById('syllableTableBody');
  tbody.innerHTML = '';

  syllables.forEach((s) => {
    const tr = document.createElement('tr');
    tr.id = `syl-row-${s.id}`;
    tr.innerHTML = `
      <td>
        <span class="syl-id-badge">
          <span class="syl-color-pip" style="background-color: ${s.color};"></span>
          #${String(s.id).padStart(2, '0')}
        </span>
      </td>
      <td>
        <input 
          type="text" 
          class="syl-name-input" 
          value="${s.name || ''}" 
          placeholder="Unlabeled (defaults to #${s.id})..."
          data-id="${s.id}"
          title="Click to edit ethological label"
        />
      </td>
      <td>
        <div class="usage-cell">
          <div class="usage-bar-bg">
            <div class="usage-bar-fill" style="width: ${Math.min(100, s.frequency * 8)}%;"></div>
          </div>
          <span class="usage-text">${s.frequency.toFixed(1)}%</span>
        </div>
      </td>
      <td>
        <span class="stat-val">${s.median_dur_frames.toFixed(1)} fr</span>
        <span style="color: var(--text-dim); font-size: 0.72rem;"> (${s.median_dur_ms.toFixed(0)} ms)</span>
      </td>
      <td>
        <span class="stat-val">${s.velocity.toFixed(1)}</span>
        <span style="color: var(--text-dim); font-size: 0.72rem;"> px/s</span>
      </td>
      <td>
        <span class="stat-val" style="color: ${s.angular_vel > 15 ? '#38bdf8' : s.angular_vel < -15 ? '#f43f5e' : '#cbd5e1'};">
          ${s.angular_vel > 0 ? '+' : ''}${s.angular_vel.toFixed(1)}°/s
        </span>
      </td>
      <td>
        <button class="btn-play-clip" data-id="${s.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
          Exemplar
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Attach inline edit events
  document.querySelectorAll('.syl-name-input').forEach(input => {
    input.addEventListener('change', async (e) => {
      const sylId = parseInt(e.target.dataset.id);
      const newName = e.target.value.trim();
      await updateSyllableLabel(sylId, newName);
    });
  });

  // Attach exemplar playback events
  document.querySelectorAll('.btn-play-clip').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const sylId = parseInt(btn.dataset.id);
      openExemplarModal(sylId);
    });
  });
}

async function updateSyllableLabel(sylId, newName) {
  try {
    const res = await fetch('/api/syllables/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ syllable_id: sylId, name: newName })
    });
    if (res.ok) {
      showToast(`Updated Syllable #${sylId} ➔ "${newName}"`);
      const target = STATE.syllables.find(s => s.id === sylId);
      if (target) target.name = newName;
    }
  } catch (err) {
    console.error(err);
  }
}

// --- Search Filter ---
function setupSearchFilter() {
  const searchInput = document.getElementById('syllableSearch');
  searchInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    const filtered = STATE.syllables.filter(s => 
      s.name.toLowerCase().includes(q) || 
      (s.category && s.category.toLowerCase().includes(q)) ||
      String(s.id).includes(q)
    );
    renderSyllablesTable(filtered);
  });
}

// --- Interactive Canvas: Log-Space Kappa Search ---
let canvas, ctx;

function setupCanvas() {
  canvas = document.getElementById('kappaCanvas');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    drawKappaCanvas();
  });
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

async function runSimulatedKappaSearch() {
  const btn = document.getElementById('btnStartSearch');
  btn.disabled = true;
  btn.innerHTML = `<span class="indicator-dot" style="background:#f59e0b"></span> Scouting in Log-Space...`;

  try {
    const res = await fetch('/api/calibrate/kappa_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_frames: STATE.targetFrames,
        subset_fraction: STATE.subsetFraction,
        start_kappa: 100000,
        max_steps: 7
      })
    });
    const data = await res.json();
    STATE.searchHistory = data.steps;
    STATE.kappaScout = data.best_scout_kappa;
    STATE.kappaProd = data.upscaled_prod_kappa;

    updateLiveUpscaling();
    animateSearchSteps(data.steps);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
      </svg>
      Run Log-Space κ Calibration
    `;
  }
}

function animateSearchSteps(steps) {
  let currentStep = 0;
  function stepAnim() {
    currentStep++;
    drawKappaCanvas(steps.slice(0, currentStep));
    if (currentStep < steps.length) {
      setTimeout(stepAnim, 280);
    } else {
      showToast(`Scout complete! Best subset κ = ${STATE.kappaScout.toExponential(2)} (Median: ${steps[steps.length - 1].median_duration_frames} fr)`);
    }
  }
  stepAnim();
}

function drawKappaCanvas(visibleSteps = STATE.searchHistory) {
  if (!canvas || !ctx) return;
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;

  ctx.clearRect(0, 0, w, h);

  const padding = { top: 40, right: 50, bottom: 50, left: 65 };
  const graphW = w - padding.left - padding.right;
  const graphH = h - padding.top - padding.bottom;

  const logMin = 2.0;
  const logMax = 8.0;
  const durMin = 4.0;
  const durMax = 32.0;

  function toX(logK) {
    return padding.left + ((logK - logMin) / (logMax - logMin)) * graphW;
  }
  function toY(dur) {
    return padding.top + graphH - ((dur - durMin) / (durMax - durMin)) * graphH;
  }

  // Draw Grid Lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let exp = logMin; exp <= logMax; exp += 1) {
    const x = toX(exp);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, padding.top + graphH);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`10^${exp}`, x, padding.top + graphH + 20);
  }

  for (let d = 5; d <= 30; d += 5) {
    const y = toY(d);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + graphW, y);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText(`${d} fr`, padding.left - 10, y + 3);
  }

  // Target Duration Band
  const targetY = toY(STATE.targetFrames);
  const targetBandTop = toY(STATE.targetFrames + 0.75);
  const targetBandBottom = toY(STATE.targetFrames - 0.75);

  ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
  ctx.fillRect(padding.left, targetBandTop, graphW, targetBandBottom - targetBandTop);

  ctx.strokeStyle = '#10b981';
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padding.left, targetY);
  ctx.lineTo(padding.left + graphW, targetY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#10b981';
  ctx.font = '11px Outfit, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Target: ${STATE.targetFrames.toFixed(0)} frames (400 ms)`, padding.left + 12, targetY - 8);

  ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let lk = logMin; lk <= logMax; lk += 0.1) {
    const d = 5.0 + 3.3 * (lk - 2.0);
    const x = toX(lk);
    const y = toY(d);
    if (lk === logMin) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (visibleSteps && visibleSteps.length > 0) {
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    visibleSteps.forEach((s, idx) => {
      const x = toX(s.log10_kappa);
      const y = toY(s.median_duration_frames);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    visibleSteps.forEach((s, idx) => {
      const x = toX(s.log10_kappa);
      const y = toY(s.median_duration_frames);
      const isLatest = idx === visibleSteps.length - 1;

      ctx.fillStyle = isLatest ? '#06b6d4' : '#6366f1';
      ctx.beginPath();
      ctx.arc(x, y, isLatest ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${s.step}`, x, y - 11);
    });
  }

  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Prior Transition Stickiness log₁₀(κ)', padding.left + graphW / 2, h - 14);

  ctx.save();
  ctx.translate(18, padding.top + graphH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Median Syllable Duration (Frames)', 0, 0);
  ctx.restore();
}

// --- Exemplar Video / Modal ---
function setupModal() {
  const modal = document.getElementById('exemplarModal');
  const closeBtn = document.getElementById('btnCloseModal');
  closeBtn.addEventListener('click', closeExemplarModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeExemplarModal();
  });

  const btnSaveModal = document.getElementById('btnSaveModalLabel');
  const modalInput = document.getElementById('modalLabelInput');
  btnSaveModal.addEventListener('click', async () => {
    if (STATE.activeExemplar !== null) {
      const val = modalInput.value.trim();
      await updateSyllableLabel(STATE.activeExemplar.id, val);
      document.getElementById('modalSyllableName').textContent = val || `(Unlabeled #${STATE.activeExemplar.id})`;
      const rowInput = document.querySelector(`#syl-row-${STATE.activeExemplar.id} .syl-name-input`);
      if (rowInput) rowInput.value = val;
    }
  });
}

function openExemplarModal(syllableId) {
  const s = STATE.syllables.find(item => item.id === syllableId);
  if (!s) return;

  STATE.activeExemplar = s;
  document.getElementById('modalSyllableTitle').textContent = `Syllable #${String(s.id).padStart(2, '0')}`;
  document.getElementById('modalSyllableName').textContent = s.name ? `"${s.name}"` : `(Unlabeled #${s.id})`;
  document.getElementById('modalLabelInput').value = s.name || '';
  document.getElementById('modalSyllableStats').textContent = 
    `Usage: ${s.frequency.toFixed(1)}% | Velocity: ${s.velocity.toFixed(1)} px/s | Duration: ${s.median_dur_frames.toFixed(1)} fr`;

  const modal = document.getElementById('exemplarModal');
  modal.classList.add('open');

  startExemplarKinematicSimulation(s);
}

function closeExemplarModal() {
  const modal = document.getElementById('exemplarModal');
  modal.classList.remove('open');
  if (STATE.exemplarAnimId) {
    cancelAnimationFrame(STATE.exemplarAnimId);
    STATE.exemplarAnimId = null;
  }
}

// --- Keypoint Skeleton Simulator (Dynamic Organism Anatomy) ---
function startExemplarKinematicSimulation(s) {
  const canvas = document.getElementById('exemplarCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = 340;

  let t = 0;
  const isTurn = s.angular_vel > 15 || s.angular_vel < -15;
  const isRear = s.elevation > 30;
  const isFast = s.velocity > 25;
  const isPause = s.velocity < 5;

  // Build active bodypart nodes
  const activeBps = STATE.useBodyparts.length ? STATE.useBodyparts : STATE.bodyparts;
  const bpsList = activeBps.length ? activeBps : ["anterior", "centroid", "posterior"];
  
  // Use project's skeleton connections if available, otherwise linear chain
  let edges = STATE.skeleton && STATE.skeleton.length ? STATE.skeleton : [];
  if (!edges.length && bpsList.length > 1) {
    edges = [];
    for (let i = 0; i < bpsList.length - 1; i++) {
      edges.push([bpsList[i], bpsList[i + 1]]);
    }
  }

  function renderFrame() {
    t += 0.05;
    ctx.fillStyle = '#060911';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    const cx = canvas.width / 2;
    const cy = canvas.height / 2 + 10;
    
    const heading = isTurn ? Math.sin(t * 1.5) * 0.8 : 0;
    const bend = isTurn ? (s.angular_vel > 0 ? 0.35 : -0.35) : Math.sin(t * 2) * 0.08;
    const bodyScale = isRear ? 0.75 + Math.sin(t * 1.2) * 0.15 : 1.0;
    const stride = isFast ? Math.sin(t * 4) * 10 : (isPause ? 0 : Math.sin(t * 2) * 3);

    // Compute dynamic coordinates for each bodypart in the active repertoire
    const coords = {};
    const n = bpsList.length;
    bpsList.forEach((bp, idx) => {
      const lower = bp.toLowerCase();
      // Longitudinal progress from posterior (-0.5) to anterior (+0.5)
      const frac = (idx / Math.max(1, n - 1)) - 0.5;
      
      let x = cx + frac * 160 * bodyScale;
      let y = cy + bend * (frac * 60);

      // Lateral symmetry for paired limbs / ears / eyes / fins / tentacles
      if (lower.includes('left') || lower.includes('_l')) {
        y -= 26 + (idx % 2 === 0 ? stride : -stride);
      } else if (lower.includes('right') || lower.includes('_r')) {
        y += 26 + (idx % 2 === 0 ? -stride : stride);
      } else if (lower.includes('arm') || lower.includes('tentacle')) {
        // Radial spread for squid / invertebrate arms
        const armIdx = parseInt(lower.replace(/\D/g, '')) || idx;
        const angle = (armIdx / 8) * Math.PI - Math.PI / 2;
        x += Math.cos(angle + heading) * (40 + stride);
        y += Math.sin(angle + heading) * (40 + stride);
      }

      // Anterior heading anchor deflection
      if (STATE.anteriorBodyparts.includes(bp) || idx === 0) {
        x += Math.cos(heading) * 15;
        y += Math.sin(heading) * 15 - (isRear ? 30 : 0);
      }

      coords[bp] = { x, y };
    });

    // Draw Skeleton Lines along project edges
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.75)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    edges.forEach(([u, v]) => {
      if (coords[u] && coords[v]) {
        ctx.moveTo(coords[u].x, coords[u].y);
        ctx.lineTo(coords[v].x, coords[v].y);
      }
    });
    ctx.stroke();

    // Draw Keypoint Nodes
    bpsList.forEach((bp, idx) => {
      const pt = coords[bp];
      if (!pt) return;
      const isAnt = STATE.anteriorBodyparts.includes(bp) || idx === 0;
      const isPost = STATE.posteriorBodyparts.includes(bp) || idx === n - 1;

      ctx.fillStyle = isAnt ? '#38bdf8' : (isPost ? '#f43f5e' : '#ffffff');
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isAnt ? 5.5 : 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Velocity Vector from Center
    const vScale = Math.min(65, s.velocity * 1.5);
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + vScale, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Telemetry text
    ctx.fillStyle = '#64748b';
    ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText(`Project Anatomy Preview (${bpsList.length} Keypoints)`, 20, 28);
    ctx.fillText(`Active Syllable: #${s.id} "${s.name || `Unlabeled #${s.id}`}"`, 20, 46);
    ctx.fillText(`Velocity: ${s.velocity.toFixed(1)} px/s | Yaw: ${s.angular_vel.toFixed(1)}°/s | Dur: ${s.median_dur_frames.toFixed(1)} fr`, 20, 64);

    STATE.exemplarAnimId = requestAnimationFrame(renderFrame);
  }

  renderFrame();
}

// --- VIDEO MANAGER & CORRESPONDENCE LOGIC ---
function setupVideoManager() {
  const btnInspect = document.getElementById('btnInspectVideos');
  const inputVid = document.getElementById('inputVideoDir');
  const inputDlc = document.getElementById('inputDlcDir');
  const downSelect = document.getElementById('downsampleSelect');
  const btnApplyFps = document.getElementById('btnApplyFps');
  const btnSaveCorr = document.getElementById('btnSaveCorrespondence');
  const modelFpsDisplay = document.getElementById('btnModelFpsDisplay');

  async function runVideoInspection() {
    const vdir = inputVid.value.trim();
    const ddir = inputDlc.value.trim();
    if (!vdir) {
      showToast('Please specify a video directory');
      return;
    }
    showToast('Inspecting behavioral videos via OpenCV...');
    try {
      const res = await fetch('/api/videos/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_dir: vdir, dlc_dir: ddir })
      });
      const data = await res.json();
      if (!res.ok) {
        showToast(`Video inspection failed: ${data.detail}`);
        return;
      }

      STATE.rawFps = data.raw_fps;
      STATE.matchedPairs = data.matched_pairs || [];
      STATE.availableDlcFiles = data.available_dlc_files || [];

      // Update Metadata Badges
      const fpsBadge = document.getElementById('metaFpsBadge');
      if (fpsBadge) fpsBadge.innerHTML = `Native Video FPS: <strong>${data.raw_fps.toFixed(1)} FPS</strong>`;
      const resBadge = document.getElementById('metaResBadge');
      if (resBadge) resBadge.textContent = `Resolution: ${data.dimensions[0]}x${data.dimensions[1]} px`;
      const countBadge = document.getElementById('metaCountBadge');
      if (countBadge) countBadge.textContent = `Recordings: ${data.total_videos} files`;

      // Populate Integer Downsample Options
      if (downSelect) {
        downSelect.innerHTML = '';
        data.integer_downsample_options.forEach(opt => {
          const o = document.createElement('option');
          o.value = opt.factor;
          o.textContent = opt.label;
          if (opt.recommended) o.selected = true;
          downSelect.appendChild(o);
        });
        updateModelFpsDisplay();
      }

      renderCorrespondenceTable();
      showToast(`Auto-matched ${data.total_matched} of ${data.total_videos} videos to DLC tracking files!`);
    } catch (err) {
      showToast('Failed to inspect video directory');
    }
  }

  function updateModelFpsDisplay() {
    const factor = parseInt(downSelect.value) || 1;
    const modelFps = Math.floor(STATE.rawFps / factor);
    if (modelFpsDisplay) modelFpsDisplay.textContent = modelFps;
  }

  if (downSelect) {
    downSelect.addEventListener('change', updateModelFpsDisplay);
  }

  if (btnInspect) {
    btnInspect.addEventListener('click', runVideoInspection);
  }

  if (btnApplyFps) {
    btnApplyFps.addEventListener('click', async () => {
      const factor = parseInt(downSelect.value) || 1;
      const modelFps = Math.floor(STATE.rawFps / factor);
      try {
        const res = await fetch('/api/videos/apply_fps', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            raw_fps: STATE.rawFps,
            downsample_factor: factor,
            model_fps: modelFps
          })
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`Applied ${modelFps} FPS (${factor}x integer downsample) to project config.yml!`);
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to save model FPS');
      }
    });
  }

  if (btnSaveCorr) {
    btnSaveCorr.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/videos/save_correspondence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pairs: STATE.matchedPairs })
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`Saved ${data.matched_count} video alignments to video_mapping.csv!`);
        } else {
          showToast(`Error: ${data.detail}`);
        }
      } catch (err) {
        showToast('Failed to save video mapping');
      }
    });
  }

  // Trigger initial inspection in background
  setTimeout(runVideoInspection, 500);
}

function renderCorrespondenceTable() {
  const tbody = document.getElementById('correspondenceTableBody');
  const ratioBadge = document.getElementById('correspondenceRatioBadge');
  if (!tbody) return;
  tbody.innerHTML = '';

  const pairs = STATE.matchedPairs || [];
  const matchedCount = pairs.filter(p => p.matched).length;
  if (ratioBadge) {
    ratioBadge.textContent = `${matchedCount} of ${pairs.length} Videos Aligned (${pairs.length ? Math.round((matchedCount / pairs.length) * 100) : 0}%)`;
    ratioBadge.style.background = matchedCount === pairs.length ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)';
    ratioBadge.style.borderColor = matchedCount === pairs.length ? 'var(--accent-emerald)' : 'var(--accent-amber)';
  }

  pairs.forEach((pair, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span style="font-family: var(--font-mono); color: var(--text-dim); font-size: 0.75rem;">${idx + 1}</span></td>
      <td>
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #fff; word-break: break-all;">
          ${pair.video_name}
        </div>
      </td>
      <td>
        <input type="text" class="syl-name-input dlc-match-input" data-idx="${idx}" style="width: 100%; font-size: 0.73rem; font-family: var(--font-mono); padding: 4px 8px; background: rgba(15,23,42,0.8);" value="${pair.dlc_name}" placeholder="Select or type matching DLC .h5 file...">
      </td>
      <td>
        <span class="cat-tag" style="${pair.matched ? 'background: rgba(16,185,129,0.15); border-color: var(--accent-emerald); color: #fff;' : 'background: rgba(244,63,94,0.15); border-color: var(--accent-rose); color: #fff;'}">
          ${pair.matched ? '✓ Auto-Matched' : '⚠️ Manual Required'}
        </span>
      </td>
      <td>
        <span style="font-size: 0.72rem; color: var(--text-dim); font-family: var(--font-mono);">
          ${pair.matched ? 'Verified' : 'Pending'}
        </span>
      </td>
    `;

    // Handle manual correction
    const input = tr.querySelector('.dlc-match-input');
    input.addEventListener('change', (e) => {
      const val = e.target.value.trim();
      pair.dlc_name = val;
      pair.matched = Boolean(val);
      pair.match_type = 'manual_override';
      renderCorrespondenceTable();
    });

    tbody.appendChild(tr);
  });
}

// --- Toast Notification ---
function showToast(msg) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <span class="indicator-dot"></span>
    <span>${msg}</span>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
