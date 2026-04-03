// Modal Functions
function openConfigModal() {
      $('#configModal').classList.add('active');
      loadConfigToForm();
}

function closeConfigModal() {
  $('#configModal').classList.remove('active');
}

function loadConfigToForm() {
  $('#numToCatMethod').value = config.numToCatMethod;
  $('#numToCatCuts').value = config.numToCatCuts;
  $('#seed').value = config.seed;
  $('#useBiasMitigation').checked = config.useBiasMitigation;
  $('#useAccuracyEnhancement').checked = config.useAccuracyEnhancement;
  $('#mainStep').value = config.mainStep;
  $('#mainClassifier').value = config.mainClassifier;
  $('#mainMaxIteration').value = config.mainMaxIteration;
  $('#mainTrainingRate').value = config.mainTrainingRate;
  $('#mainThresholdEpsilon').value = config.mainThresholdEpsilon;
  $('#mainThresholdEpsilonSlider').value = config.mainThresholdEpsilon;
  $('#epsilonValue').textContent = config.mainThresholdEpsilon.toFixed(2);
  $('#mainThresholdAccuracy').value = config.mainThresholdAccuracy;
  $('#mainAeImportanceMeasure').value = config.mainAeImportanceMeasure;
  $('#mainAeRebinMethod').value = config.mainAeRebinMethod;
  $('#mainAlphaO').value = config.mainAlphaO;
  $('#evalHOrder').value = config.evalHOrder;
  $('#evalSum').value = config.evalSum;
  $('#evalCat').value = config.evalCat;
  $('#evalNum').value = config.evalNum;
  $('#evalScale').value = config.evalScale;
  $('#evalDistMetric').value = config.evalDistMetric;
  $('#transformMethod').value = config.transformMethod;
  $('#transformMulti').value = config.transformMulti;
  $('#transformStream').value = config.transformStream;
  $('#streamConfigP').value = config.streamConfig.p;
  $('#streamConfigQ').value = config.streamConfig.q;
  $('#streamConfigEmin').value = config.streamConfig.emin;
  $('#streamConfigEmax').value = config.streamConfig.emax;
  $('#streamConfigOrder').value = config.streamConfig.order;
  $('#streamConfigLength').value = config.streamConfig.length;
  
  // Setup epsilon slider sync
  setupEpsilonSync();
}

function setupEpsilonSync() {
  const slider = $('#mainThresholdEpsilonSlider');
  const input = $('#mainThresholdEpsilon');
  const display = $('#epsilonValue');
  
  function updateSliderProgress(value) {
    const percent = (value / 1) * 100;
    slider.style.setProperty('--slider-value', percent + '%');
  }
  
  // Initialize
  updateSliderProgress(slider.value);
  
  slider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    input.value = value;
    display.textContent = value.toFixed(2);
    updateSliderProgress(value);
  });
  
  input.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value >= 0 && value <= 1) {
      slider.value = value;
      display.textContent = value.toFixed(2);
      updateSliderProgress(value);
    }
  });
}

function saveConfigFromForm() {
  config.numToCatMethod = $('#numToCatMethod').value;
  config.numToCatCuts = parseInt($('#numToCatCuts').value);
  config.seed = parseInt($('#seed').value);
  config.useBiasMitigation = $('#useBiasMitigation').checked;
  config.useAccuracyEnhancement = $('#useAccuracyEnhancement').checked;
  
  console.log('[DEBUG] Saved config - Use BM:', config.useBiasMitigation, 'Use AE:', config.useAccuracyEnhancement);
  config.mainStep = $('#mainStep').value;
  config.mainClassifier = $('#mainClassifier').value;
  config.mainMaxIteration = parseInt($('#mainMaxIteration').value);
  config.mainTrainingRate = parseFloat($('#mainTrainingRate').value);
  config.mainThresholdEpsilon = parseFloat($('#mainThresholdEpsilon').value);
  config.mainThresholdAccuracy = parseFloat($('#mainThresholdAccuracy').value);
  config.mainAeImportanceMeasure = $('#mainAeImportanceMeasure').value;
  config.mainAeRebinMethod = $('#mainAeRebinMethod').value;
  config.mainAlphaO = parseFloat($('#mainAlphaO').value);
  config.evalHOrder = $('#evalHOrder').value;
  config.evalSum = $('#evalSum').value;
  config.evalCat = $('#evalCat').value;
  config.evalNum = $('#evalNum').value;
  config.evalScale = $('#evalScale').value;
  config.evalDistMetric = $('#evalDistMetric').value;
  config.transformMethod = $('#transformMethod').value;
  config.transformMulti = $('#transformMulti').value;
  config.transformStream = $('#transformStream').value;
  config.streamConfig.p = parseInt($('#streamConfigP').value);
  config.streamConfig.q = parseInt($('#streamConfigQ').value);
  config.streamConfig.emin = parseFloat($('#streamConfigEmin').value);
  config.streamConfig.emax = parseFloat($('#streamConfigEmax').value);
  config.streamConfig.order = parseInt($('#streamConfigOrder').value);
  config.streamConfig.length = parseInt($('#streamConfigLength').value);
  
  // Get selected fairness metrics
  config.evalMetricFairness = Array.from($$('.checkbox-group input[type="checkbox"]'))
    .filter(cb => cb.checked && ['BNC','BPC','CUAE','EOpp','EO','FDRP','FORP','FNRB','FPRB','NPVP','OAE','PPVP','SP'].includes(cb.value))
    .map(cb => cb.value);

  // Get selected accuracy metrics
  config.evalMetricAccuracy = Array.from($$('.checkbox-group input[type="checkbox"]'))
    .filter(cb => cb.checked && ['ACC','F1','Recall','Precision'].includes(cb.value))
    .map(cb => cb.value);
}

// Update backend configuration
async function updateBackendConfig() {
  console.log('[DEBUG] Current config:', config);
  console.log('[DEBUG] Use Bias Mitigation:', config.useBiasMitigation);
  console.log('[DEBUG] Use Accuracy Enhancement:', config.useAccuracyEnhancement);
  
  const backendConfig = {
    'PARAMS_NUM_TO_CAT_METHOD': config.numToCatMethod,
    'PARAMS_NUM_TO_CAT_CUTS': config.numToCatCuts,
    'SEED': config.seed,
    'USE_BIAS_MITIGATION': config.useBiasMitigation,
    'USE_ACCURACY_ENHANCEMENT': config.useAccuracyEnhancement,
    'PARAMS_MAIN_STEP': config.mainStep,
    'PARAMS_MAIN_CLASSIFIER': config.mainClassifier,
    'PARAMS_MAIN_MAX_ITERATION': config.mainMaxIteration,
    'PARAMS_MAIN_TRAINING_RATE': config.mainTrainingRate,
    'PARAMS_MAIN_THRESHOLD_EPSILON': config.mainThresholdEpsilon,
    'PARAMS_MAIN_THRESHOLD_ACCURACY': config.mainThresholdAccuracy,
    'PARAMS_MAIN_AE_IMPORTANCE_MEASURE': config.mainAeImportanceMeasure,
    'PARAMS_MAIN_AE_REBIN_METHOD': config.mainAeRebinMethod,
    'PARAMS_MAIN_ALPHA_O': config.mainAlphaO,
    'PARAMS_EVAL_H_ORDER': config.evalHOrder,
    'PARAMS_EVAL_SUM': config.evalSum,
    'PARAMS_EVAL_CAT': config.evalCat,
    'PARAMS_EVAL_NUM': config.evalNum,
    'PARAMS_EVAL_SCALE': config.evalScale,
    'PARAMS_EVAL_DIST_METRIC': config.evalDistMetric,
    'PARAMS_EVAL_METRIC_FAIRNESS': config.evalMetricFairness,
    'PARAMS_EVAL_METRIC_ACCURACY': config.evalMetricAccuracy,
    'PARAMS_TRANSFORM': config.transformMethod,
    'PARAMS_TRANSFORM_MULTI': config.transformMulti,
    'PARAMS_TRANSFORM_STREAM': config.transformStream,
    'PARAMS_TRANSFORM_STREAM_CONFIG': config.streamConfig
  };
  
  await api.updateConfig(backendConfig);
}

$('#btnConfig').addEventListener('click', openConfigModal);
$('#btnCloseModal').addEventListener('click', closeConfigModal);
$('#btnCancelModal').addEventListener('click', closeConfigModal);
$('#modalOverlay').addEventListener('click', closeConfigModal);
$('#btnSaveConfig').addEventListener('click', ()=> {
  saveConfigFromForm();
  closeConfigModal();
  const bmStatus = config.useBiasMitigation ? 'Enabled' : 'Disabled';
  const aeStatus = config.useAccuracyEnhancement ? 'Enabled' : 'Disabled';
  alert(`Configuration saved successfully!\n\nBias Mitigation: ${bmStatus}\nAccuracy Enhancement: ${aeStatus}`);
});
