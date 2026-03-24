// 轮询job进度
let pollInterval = null;

let demoPollInterval = null;
let isPollingDemo = false;

function pollDemoJobProgress() {
    if (demoPollInterval) clearInterval(demoPollInterval);

    demoPollInterval = setInterval(async () => {
        console.log("polling")
        if (isPollingDemo) return;
        isPollingDemo = true;

        try {
            const result = await api.getDemoJobStatus(state.jobId);

            if (result.current_iteration > state.currentStep) {
                /* 首次更新时计算epsilonThreshold的平均值，后续更新时保持不变 */
                if (state.currentStep === 0) {
                    const sexData = result?.data?.iterations?.[0]?.epsilon?.sex;

                    console.log('[DEBUG sexData]', sexData);

                    if (sexData) {
                        const values = Object.values(sexData);

                        state.epsilonThreshold =
                        values.reduce((a, b) => a + b, 0) / values.length;
                    } else {
                        console.error('sex data missing!', result);
                    }
                }
                
                console.log('[DEBUG poll]', result);
                console.log('[DEBUG job_status]', result?.job_status);
                
                state.currentStep = result.current_iteration;

                console.log(`updating figure, current_iteration: ${state.currentStep}`);
                renderProgress(result.data);
            }

            if (result.job_status === "completed") {
                clearInterval(demoPollInterval);
                demoPollInterval = null;

                completeProcess();
                updateProcessStatus("Completed");

    /*              if (result.log_path) {
                    state.logPath = result.log_path;
                    showLogDownloadButton();
                } */

                showDownloadButton();
            }
        } catch (err) {
            console.error("Polling failed:", err);
            clearInterval(demoPollInterval);
            demoPollInterval = null;
        } finally {
            isPollingDemo = false;
        }
    }, 5000);
}

async function pollJobProgress() {
    if (!state.jobId) return;
    
    // 清除之前的轮询
    if (pollInterval) {
    clearInterval(pollInterval);
    }
    
    // 启动实时时间更新
    startTimeUpdate();
    
    pollInterval = setInterval(async () => {
    try {
        const statusResult = await api.getJobStatus(state.jobId);
        
        if (statusResult.status === 'success') {
            const data = statusResult.data;
            
    /*             console.log('[DEBUG] pollJobProgress - Full response data:', data);
            console.log('[DEBUG] pollJobProgress - data.max_epsilon_series:', data.max_epsilon_series);
            console.log('[DEBUG] pollJobProgress - data.epsilon_threshold:', data.epsilon_threshold); */
            
            // 保存 max_epsilon_series 和 epsilon_threshold 到 state
            if (data.max_epsilon_series && Array.isArray(data.max_epsilon_series) && data.max_epsilon_series.length > 0) {
                state.maxEpsilonSeries = data.max_epsilon_series;
                /* console.log('[DEBUG] ✅ Saved max_epsilon_series to state:', state.maxEpsilonSeries); */
            } else {
                /* console.log('[DEBUG] ❌ max_epsilon_series is empty or invalid, not saving to state'); */
            }
            if (data.epsilon_threshold !== undefined) {
                state.epsilonThreshold = data.epsilon_threshold;
                /* console.log('[DEBUG] epsilon_threshold:', data.epsilon_threshold); */
            }
            
            // 更新进度显示
            updateProcessStatus(`Running (${data.current_iteration}/${data.max_iteration})`);
            
            // 更新history（只包含已完成的iterations）
            if (data.history && data.history.length > 0) {
/*                 console.log('[DEBUG] Received history from backend:', data.history.length, 'items');
                console.log('[DEBUG] Current state.history:', state.history.length, 'items'); */
                
                // 合并新的history（包含初始值和所有迭代）
                // state.history[0] 是初始metrics，data.history 包含所有迭代
                if (state.history.length > 0 && state.history[0]) {
                // 如果已有初始值，合并后续的迭代
                if (data.history.length > state.history.length - 1) {
                    state.history = [state.history[0], ...data.history];
                    /* console.log('[DEBUG] Merged history with initial value, total:', state.history.length); */
                }
                } else {
                // 如果没有初始值，直接使用data.history
                state.history = data.history;
                /* console.log('[DEBUG] Using backend history directly, total:', state.history.length); */
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
                completeProcess();
                
                // 更新最后一次的 history 数据
                if (data.history && data.history.length > 0) {
                if (state.history.length > 0 && state.history[0]) {
                    state.history = [state.history[0], ...data.history];
                } else {
                    state.history = data.history;
                }
                /* console.log('[DEBUG] Final history update on completion, total:', state.history.length); */
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
                
/*                 console.log('[INFO] Process completed!', {
                    terminated: data.terminated,
                    reason: data.termination_reason,
                    iterations: data.current_iteration,
                    logPath: data.log_path
                }); */
                
                // 改变按钮为下载按钮
                showDownloadButton();

                showProcessStep(state.history.length - 1, true);
            } 

            if (data.state === 'failed') {
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
                
                // 重置按钮
                $('#runBtnText').textContent = 'Run';
                $('#btnRun').disabled = true;
                $('#btnOpenAttrModal').disabled = true;
                state.isRunning = false;
            }
        }
    } catch (err) {
        console.error('Error polling job status:', err);
    }
    }, 5000); // 每500ms轮询一次
}