async function runFullProcess() {
    if (!state.datasetId) {
    alert('Please load a dataset first.');
        return;
    }
    
    // Get selected protected attributes from data explorer
    const protectedAttrs = getSelectedProtectedAttributes();
    if (protectedAttrs.length === 0) {
    alert('Please select at least one protected attribute in the Data Explorer first.');
    return;
    }

    const targetAttrs = getSelectedTargetAttributes();
    if (!targetAttrs){
        alert('Please select one target attribute in the Data Explorer first.');
        return;
    }
    
    startProcess();

    try {
        // Update backend config
        await updateBackendConfig();

        const initResult = await api.startDemoJob(state.runMode);
        if (initResult.status !== 'success') {
            throw new Error(initResult.message);
        }
        
        state.jobId = initResult.job_id;
        
        console.log(`Started poll with ID: ${state.jobId}`);
        pollJobProgress(protectedAttrs);
    } catch (err) {
        processError(err);
    }
}

async function runStepProcess() {
    if (!state.datasetId) {
    alert('Please load a dataset first.');
        return;
    }
    
    // Get selected protected attributes from data explorer
    const protectedAttrs = getSelectedProtectedAttributes();
    if (protectedAttrs.length === 0) {
    alert('Please select at least one protected attribute in the Data Explorer first.');
    return;
    }

    const targetAttrs = getSelectedTargetAttributes();
    if (!targetAttrs){
        alert('Please select one target attribute in the Data Explorer first.');
        return;
    }
    
    startProcess();

    try {
        // Update backend config
        await updateBackendConfig();
        
        if (state.currentStep === 0) {
            const initResult = await api.startDemoJob(state.runMode);
            if (initResult.status !== 'success') {
                throw new Error(initResult.message);
            }
        }
        
        state.jobId = initResult.job_id;
        await api.runDemoJobStep(state.jobId);

        pollJobProgress(protectedAttrs);
    } catch (err) {
        processError(err);
    }
}

function resetProcess() {
    // 清除轮询
    if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
    }
    
    // 停止时间更新
    resetTime();
    
    state.isRunning = false;
    state.currentStep = 0;
    state.history = [];
    state.jobId = null;
    /* state.startTime = null; */
    state.logPath = null;
    state.maxEpsilonSeries = [];
    state.epsilonThreshold = 0;
    state.canDownload = false;
    state.data = null;
    state.canViewDetail = false;  // 是否可以查看细节

    hideLogDownloadButton();
    updateProcessStatus('Ready');
    $('#btnOpenAttrModal').disabled = false;
    $('#btnOpenTargetModal').disabled = false;
    $('#runBtnText').textContent = 'Run';
    $('#btnRun').disabled = false;
    $('#btnRun').classList.remove('btn--download');
    
    const processContent = $('#processContent');
    processContent.innerHTML = `
    <div class="empty-state">
        <div class="empty-state__icon">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20v-6m0 0V4m0 10l4-4m-4 4l-4-4"/>
            <circle cx="12" cy="12" r="10" opacity="0.2"/>
        </svg>
        </div>
        <p class="empty-state__title">Ready to start debiasing</p>
        <p class="empty-state__text">Load your dataset and configure parameters to begin</p>
    </div>
    `;
}

function startProcess() {
    state.isRunning = true;
    startTimeUpdate(); // 启动时间更新
    updateProcessStatus('Running');
    $('#runBtnText').textContent = 'Running...';
    $('#btnRun').disabled = true;
    $('#btnOpenAttrModal').disabled = true;
    $('#btnOpenTargetModal').disabled = true;
}

function pauseProcess() {
    state.isRunning = false;
    stopTimeUpdate(); // 启动时间更新
    updateProcessStatus('Ready');
    $('#runBtnText').textContent = 'Run';
    $('#btnRun').disabled = false;
    $('#btnOpenAttrModal').disabled = true;
    $('#btnOpenTargetModal').disabled = true;
}

function processError(err){
    alert(`Error: ${err.message}`);
    console.error(err.stack);
    updateProcessStatus('Error');
    $('#runBtnText').textContent = 'Error';
    $('#btnRun').disabled = true;
    $('#btnOpenAttrModal').disabled = true;
    $('#btnOpenTargetModal').disabled = true;
    state.isRunning = false;
    stopTimeUpdate();
}

function completeProcess(log_path) {
    stopTimeUpdate();
    updateProcessStatus("Completed");
    if (log_path) {
        state.logPath = log_path;
        showLogDownloadButton();
    }
    showViewDetailsButton();
}

// Run Mode Selection
$$('input[name="runMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        resetProcess();
        state.runMode = e.target.value;
        $('#runBtnText').textContent = state.runMode === 'step' ? 'Next Step' : 'Run';
    });
});

// Process Controls
$('#btnRun').addEventListener('click', () => {
    // Check if in download mode
    if (state.canViewDetail) {
        transitionToResultsPage();
        return;
    }
    
    if (!state.currentData && !state.currentFile) {
        alert('Please load a dataset first.');
        return;
    }
    
    if (state.runMode === 'all') runFullProcess();
    if (state.runMode === 'step') runStepProcess();
});

$('#btnReset').addEventListener('click', resetProcess);
