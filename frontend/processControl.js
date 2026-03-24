async function runAllSteps() {
    if (!state.datasetId) {
    alert('Please load a dataset first.');
        return;
    }
    
    // Get selected protected attributes from data explorer
    const protectedAttrs = getSelectedProtectedAttributes();
    if (!protectedAttrs || protectedAttrs.length === 0) {
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
    
    // Initialize debiasing job with selected protected attributes
    console.log('[INFO] Starting debiasing with protected attributes:', protectedAttrs);
    if (!state.runDemo) {
        const initResult = await api.initDebias(state.datasetId, protectedAttrs,targetAttrs);
        if (initResult.status !== 'success') {
            throw new Error(initResult.message);
        }
        
        state.jobId = initResult.data.job_id;
        
        // Add initial metrics to history
        state.history.push({
            iteration: 0,
            metrics: initResult.data.init_metrics
        });
        
        // Start full process in background
        const startResult = await api.runFullProcess(state.jobId);
        
        if (startResult.status !== 'success') {
            throw new Error(startResult.message);
        }
        
        // 开始轮询进度（不在这里重置按钮，等轮询完成后再重置）
        pollJobProgress();
    }else {
        const initResult = await api.startDemoJob();
        if (initResult.status !== 'success') {
            throw new Error(initResult.message);
        }
        
        state.jobId = initResult.job_id;
        
        pollDemoJobProgress();
    }


    } catch (err) {
    alert(`Error: ${err.message}`);
    updateProcessStatus('Error');
    $('#runBtnText').textContent = 'Run';
    $('#btnRun').disabled = false;
    $('#btnOpenAttrModal').disabled = false;
    $('#btnOpenTargetModal').disabled = false;
    state.isRunning = false;
    }
}

async function runNextStep() {
    if (!state.datasetId) {
    alert('Please load a dataset first.');
        return;
    }
    
    try {
    // Initialize job if not started
    if (!state.jobId) {
        // Get selected protected attributes from data explorer
        const protectedAttrs = getSelectedProtectedAttributes();
        if (!protectedAttrs || protectedAttrs.length === 0) {
        alert('Please select at least one protected attribute in the Data Explorer first.');
        return;
        }
                
        const targetAttrs = getSelectedTargetAttributes();
        if (!targetAttrs) {
            alert('Please select one target attribute in the Data Explorer first.');
            return;
        }

        
       /*  state.startTime = Date.now(); */  // 记录开始时间
        await startProcess(); 
        await updateBackendConfig();
        
        console.log('[INFO] Starting debiasing with protected attributes:', protectedAttrs);
        const initResult = await api.initDebias(state.datasetId, protectedAttrs, targetAttrs);
        if (initResult.status !== 'success') {
            throw new Error(initResult.message);
        }
        
        state.jobId = initResult.data.job_id;
        state.history.push({
        iteration: 0,
        metrics: initResult.data.init_metrics
        });
        
    }else {
        startProcess();
    }
    

    // Execute one complete iteration (BM + AE + evaluate)
    const result = await api.stepIteration(state.jobId);

    const statusResult = await api.getJobStatus(state.jobId);
    
    if (statusResult.status === 'success') {
        const data = statusResult.data;
        
        console.log('[DEBUG] pollJobProgress - Full response data:', data);
        console.log('[DEBUG] pollJobProgress - data.max_epsilon_series:', data.max_epsilon_series);
        console.log('[DEBUG] pollJobProgress - data.epsilon_threshold:', data.epsilon_threshold);
        
        // 保存 max_epsilon_series 和 epsilon_threshold 到 state
        if (data.max_epsilon_series && Array.isArray(data.max_epsilon_series) && data.max_epsilon_series.length > 0) {
            state.maxEpsilonSeries = data.max_epsilon_series;
            console.log('[DEBUG] ✅ Saved max_epsilon_series to state:', state.maxEpsilonSeries);
        } else {
            console.log('[DEBUG] ❌ max_epsilon_series is empty or invalid, not saving to state');
        }
        if (data.epsilon_threshold !== undefined) {
        state.epsilonThreshold = data.epsilon_threshold;
        console.log('[DEBUG] epsilon_threshold:', data.epsilon_threshold);
        }
        
        // 更新进度显示
        updateProcessStatus(`Running (${data.current_iteration}/${data.max_iteration})`);
        
        // 更新history（只包含已完成的iterations）
        if (data.history && data.history.length > 0) {
        console.log('[DEBUG] Received history from backend:', data.history.length, 'items');
        console.log('[DEBUG] Current state.history:', state.history.length, 'items');
        
        // 合并新的history（包含初始值和所有迭代）
        // state.history[0] 是初始metrics，data.history 包含所有迭代
        if (state.history.length > 0 && state.history[0]) {
            // 如果已有初始值，合并后续的迭代
            if (data.history.length > state.history.length - 1) {
            state.history = [state.history[0], ...data.history];
            console.log('[DEBUG] Merged history with initial value, total:', state.history.length);
            }
        } else {
            // 如果没有初始值，直接使用data.history
            state.history = data.history;
            console.log('[DEBUG] Using backend history directly, total:', state.history.length);
        }
        
        // 检查 history 中的每个元素
        state.history.forEach((h, i) => {
            if (!h || !h.metrics) {
            console.warn(`[WARN] History item ${i} is invalid:`, h);
            }
        });
        
        // 实时更新图表
        showProcessStep(state.history.length - 1, true);
        }
        
        // 检查是否完成
        if (data.state === 'completed') {
            clearInterval(pollInterval);
            pollInterval = null;
            stopTimeUpdate(); // 停止时间更新
            
            // 更新最后一次的 history 数据
            if (data.history && data.history.length > 0) {
                if (state.history.length > 0 && state.history[0]) {
                state.history = [state.history[0], ...data.history];
                } else {
                state.history = data.history;
                }
                console.log('[DEBUG] Final history update on completion, total:', state.history.length);
            }
            
            // 显示最终结果
            if (state.history.length > 0) {
                showProcessStep(state.history.length - 1, true);
            }
            
            updateProcessStatus('Completed');
            
            // If log_path is available, show it
            if (data.log_path) {
                state.logPath = data.log_path;
                showLogDownloadButton();
            }
            
            console.log('[INFO] Process completed!', {
                terminated: data.terminated,
                reason: data.termination_reason,
                iterations: data.current_iteration,
                logPath: data.log_path
            });
            
            // 改变按钮为下载按钮
            $('#runBtnText').textContent = 'Download Transforms';
            $('#btnRun').disabled = false;
            $('#btnRun').classList.add('btn--download');
            state.isRunning = false;
            state.canDownload = true;
        } else if (data.state === 'failed') {
            clearInterval(pollInterval);
            pollInterval = null;
            stopTimeUpdate(); // 停止时间更新
            
            updateProcessStatus('Failed');
            
            let failMessage = `Process failed: ${data.error || 'Unknown error'}`;
            
            // If log_path is available, show it even for failed jobs
            if (data.log_path) {
                state.logPath = data.log_path;
                failMessage += `\n\nDebug log saved: ${data.log_path}`;
                showLogDownloadButton();
            }
            
            alert(failMessage);
            
            state.isRunning = false;
        }else {
            clearInterval(pollInterval);
            pollInterval = null;
            stopTimeUpdate();

            updateProcessStatus('Ready');

            $('#runBtnText').textContent = 'Run';
            $('#btnRun').disabled = false;
            
            state.isRunning = false;
        }
    }   
    } catch (err) {
    state.isRunning = false;
    stopTimeUpdate(); // 停止时间更新
    alert(`Error: ${err.message}`);
    updateProcessStatus('Error');
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

function completeProcess(status) {
    stopTimeUpdate();
}
// Run Mode Selection
$$('input[name="runMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
    state.runMode = e.target.value;
    $('#runBtnText').textContent = state.runMode === 'step' ? 'Next Step' : 'Run';
    });
});

// Process Controls
$('#btnRun').addEventListener('click', () => {
    // Check if in download mode
    if (state.canDownload) {
    downloadTransformationRules();
    return;
    }
    
    if (!state.currentData && !state.currentFile) {
    alert('Please load a dataset first.');
    return;
    }
    
    if (state.runMode === 'all') {
    runAllSteps();
    } else {
    runNextStep();
    }
});

$('#btnReset').addEventListener('click', resetProcess);
