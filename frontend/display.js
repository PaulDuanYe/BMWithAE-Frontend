// Process Control Functions
/* const processSteps = [
    { name: 'Data Preprocessing', desc: 'Preparing and cleaning data' },
    { name: 'Feature Engineering', desc: 'Extracting and transforming features' },
    { name: 'Model Training', desc: 'Training classification model' },
    { name: 'Bias Detection', desc: 'Analyzing fairness metrics' },
    { name: 'Bias Mitigation', desc: 'Applying debiasing techniques' },
    { name: 'Accuracy Enhancement', desc: 'Optimizing model performance' },
    { name: 'Final Evaluation', desc: 'Computing final metrics' }
];
 */

function showProcessStep(stepIndex, realData = null) {
    const processContent = $('#processContent');
    
    // 获取当前选择的 metric（如果有的话），默认 Max_Epsilon
    if (!state.selectedMetric) {
    state.selectedMetric = 'Max_Epsilon';
    }
    const currentSelectedMetric = state.selectedMetric;
    console.log(`[DEBUG] Rendering chart for metric: ${currentSelectedMetric}, step: ${stepIndex}, realData: ${realData}`);
    
    // 确定当前步骤名称和描述
    let stepName = 'Processing';
    let stepDesc = 'Running debiasing process';
    let currentIteration = state.history.length;
    
    if (realData && state.history.length > 0) {
    const lastHistory = state.history[state.history.length - 1];
    // 根据 iteration 数据判断当前步骤
    if (config.useBiasMitigation && !config.useAccuracyEnhancement) {
        stepName = 'Bias Mitigation';
        stepDesc = 'Reducing bias in the model';
    } else if (!config.useBiasMitigation && config.useAccuracyEnhancement) {
        stepName = 'Accuracy Enhancement';
        stepDesc = 'Improving model accuracy';
    } else if (config.useBiasMitigation && config.useAccuracyEnhancement) {
        // 两者都启用时，显示综合步骤
        stepName = 'Bias Mitigation & Accuracy Enhancement';
        stepDesc = 'Optimizing fairness and accuracy';
    }
    }
    
    // Use real data if available, otherwise generate sample data
    let fairnessData, accuracyData, fairnessScore, accuracyScore;
    
    console.log('[DEBUG] showProcessStep - Before data processing:');
    console.log('  - currentSelectedMetric:', currentSelectedMetric);
    console.log('  - state.maxEpsilonSeries:', state.maxEpsilonSeries);
    console.log('  - state.epsilonThreshold:', state.epsilonThreshold);
    console.log('  - realData:', realData);
    console.log('  - state.history.length:', state.history.length);
    
    // 准备左侧图表数据（根据选择的 metric）
    let leftChartData = [];
    let leftChartLabel = '';
    let showThreshold = false;
    
    if (currentSelectedMetric === 'Max_Epsilon' && realData && state.maxEpsilonSeries.length > 0) {
    // Max Epsilon 模式
    leftChartData = state.maxEpsilonSeries.map((value, i) => ({
        iteration: i + 1,
        value: value
    }));
    leftChartLabel = 'Max Epsilon';
    showThreshold = true;  // 只有 Max Epsilon 显示阈值线
    console.log('[DEBUG] Left chart - Max epsilon data:', leftChartData);
    } else if (realData && state.history.length > 0) {
    // 其他 Fairness Metrics 模式
    leftChartData = state.history.map((h, i) => {
        let value = 0;
        if (h && h.metrics && h.metrics[currentSelectedMetric] !== undefined) {
        const metricValue = h.metrics[currentSelectedMetric];
        if (typeof metricValue === 'object' && metricValue !== null) {
            const values = Object.values(metricValue).filter(v => typeof v === 'number');
            value = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
        } else if (typeof metricValue === 'number') {
            value = metricValue;
        }
        }
        return { iteration: i + 1, value: value };
    });
    leftChartLabel = currentSelectedMetric;
    console.log('[DEBUG] Left chart - Fairness metric data:', leftChartData);
    }
    
    // Extract accuracy data（无论选择什么 metric 都要显示）
    // Skip iteration 0 (initial state) to align with epsilon series
    if (realData && state.history.length > 0) {
    accuracyData = state.history
        .filter((h, i) => i > 0)  // Skip iteration 0
        .map((h, i) => ({
        iteration: i + 1,
        value: h && h.metrics && h.metrics.ACC ? h.metrics.ACC : 0
        }));
    console.log('[DEBUG] Accuracy data extracted (excluding iteration 0):', accuracyData);
    } else {
    accuracyData = [];
    console.log('[DEBUG] No accuracy data - realData:', realData, 'history.length:', state.history.length);
    }
    
    if (realData && state.history.length > 0) {
    // Use real historical data - 显示用户选择的具体metric
    fairnessData = state.history.map((h, i) => {
        let value = 0;
        if (h && h.metrics && h.metrics[currentSelectedMetric] !== undefined) {
        const metricValue = h.metrics[currentSelectedMetric];
        // 如果是嵌套字典（如 {'SEX': 0.001}），取平均值
        if (typeof metricValue === 'object' && metricValue !== null) {
            const values = Object.values(metricValue).filter(v => typeof v === 'number');
            value = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
        } else if (typeof metricValue === 'number') {
            value = metricValue;
        }
        }
        return { iteration: i + 1, value: value };
    });
    
    // Safely access latest metrics
    const latestHistory = state.history[state.history.length - 1];
    const latestMetrics = latestHistory ? latestHistory.metrics : null;
    
    if (latestMetrics && latestMetrics[currentSelectedMetric] !== undefined) {
        const metricValue = latestMetrics[currentSelectedMetric];
        if (typeof metricValue === 'object' && metricValue !== null) {
        const values = Object.values(metricValue).filter(v => typeof v === 'number');
        fairnessScore = values.length > 0 
            ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(6)
            : '0.000000';
        } else {
        fairnessScore = metricValue.toFixed(6);
        }
    } else {
        fairnessScore = '0.000000';
    }
    accuracyScore = latestMetrics ? (latestMetrics.ACC || 0.8).toFixed(4) : '0.8000';
    } else {
    // Generate sample data
    fairnessData = Array.from({length: 10}, (_, i) => ({
        iteration: i + 1,
        value: 0.5 + Math.random() * 0.3 + (i * 0.02)
    }));
    accuracyData = Array.from({length: 10}, (_, i) => ({
        iteration: i + 1,
        value: 0.85 - Math.random() * 0.05 - (i * 0.005)
    }));
    fairnessScore = (Math.random() * 0.3 + 0.7).toFixed(6);
    accuracyScore = (Math.random() * 0.1 + 0.85).toFixed(4);
    }
    
    // Fairness metrics descriptions
    const metricDescriptions = {
    'BNC': 'Between Negative Classes',
    'BPC': 'Between Positive Classes',
    'CUAE': 'Conditional Use Accuracy Equality',
    'EOpp': 'Equal Opportunity',
    'EO': 'Equalized Odds',
    'FDRP': 'False Discovery Rate Parity',
    'FORP': 'False Omission Rate Parity',
    'FNRB': 'False Negative Rate Balance',
    'FPRB': 'False Positive Rate Balance',
    'NPVP': 'Negative Predictive Value Parity',
    'OAE': 'Overall Accuracy Equality',
    'PPVP': 'Positive Predictive Value Parity',
    'SP': 'Statistical Parity'
    };
    
    // 计算右侧 Accuracy 图表的 Y轴范围（仅使用 accuracyData）
    const accuracyValues = accuracyData.map(d => d.value);
    const accMaxValue = accuracyData.length > 0 ? Math.max(...accuracyValues) : 1;
    const accMinValue = accuracyData.length > 0 ? Math.min(...accuracyValues) : 0;
    const accRange = accMaxValue - accMinValue || 0.001;
    
    // 添加15%的padding
    const yMax = accMaxValue + accRange * 0.15;
    const yMin = Math.max(0, accMinValue - accRange * 0.15);
    const yRange = yMax - yMin || 0.001;
    
    // Y轴缩放函数（右侧 Accuracy 图表使用）
    const scaleY = (value) => {
    return 250 - ((value - yMin) / yRange) * 220;
    };
    
    // Y轴标签格式化函数（不使用科学计数法）
    const formatYLabel = (value) => {
    // 对于非常小的值使用科学计数法
    if (Math.abs(value) < 0.0001 && value !== 0) {
        return value.toExponential(2);
    } else if (Math.abs(value) < 0.01) {
        return value.toFixed(6);
    } else if (value >= 0 && value <= 1) {
        return value.toFixed(3);
    } else {
        return value.toFixed(2);
    }
    };
    
    // 计算左侧图表的 Y轴范围
    const leftChartValues = leftChartData.map(d => d.value);
    const leftMaxValue = leftChartData.length > 0 ? Math.max(...leftChartValues, showThreshold ? state.epsilonThreshold : 0) : 0.001;
    const leftMinValue = leftChartData.length > 0 ? Math.min(...leftChartValues, showThreshold ? state.epsilonThreshold : 0) : 0;
    const leftRange = leftMaxValue - leftMinValue || 0.0001;
    const leftYMax = leftMaxValue + leftRange * 0.15;
    const leftYMin = Math.max(0, leftMinValue - leftRange * 0.15);
    const leftYRange = leftYMax - leftYMin || 0.0001;
    const scaleLeftY = (value) => 250 - ((value - leftYMin) / leftYRange) * 220;
    
    // 计算总时间
    const totalTime = state.startTime ? ((Date.now() - state.startTime) / 1000).toFixed(2) : '0.00';
    
    processContent.innerHTML = `
    <div class="process-step-display">
        <div class="step-visualization" style="padding: 8px 24px; background: transparent;">
        <div class="charts-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            
            <!-- Left Chart: Fairness Metrics -->
            <div class="chart-section" style="background: transparent; border: none; padding: 0; padding-right: 20px; border-right: 1px solid #e5e7eb;">
            <div class="chart-header" style="margin-top: 16px;">
                <h4 class="chart-title">Metrics</h4>
                <select class="metric-selector" id="metricSelector">
                <option value="Max_Epsilon" ${currentSelectedMetric === 'Max_Epsilon' ? 'selected' : ''}>Max Epsilon</option>
                <option value="Overall_Fairness" ${currentSelectedMetric === 'Overall_Fairness' ? 'selected' : ''}>Overall Fairness Score</option>
                <optgroup label="Individual Metrics">
                    ${config.evalMetricFairness.map(metric => 
                    `<option value="${metric}" ${currentSelectedMetric === metric ? 'selected' : ''}>${metric} - ${metricDescriptions[metric]}</option>`
                    ).join('')}
                </optgroup>
                </select>
            </div>
            <div class="line-chart" id="leftChart">
                <svg viewBox="0 0 400 280" class="chart-svg" preserveAspectRatio="xMidYMid meet">
                <!-- Grid lines -->
                ${Array.from({length: 6}, (_, i) => {
                    const yPos = 30 + i * 44;
                    const yValue = leftYMax - (i / 5) * leftYRange;
                    return `
                    <line x1="60" y1="${yPos}" x2="380" y2="${yPos}" stroke="#e5e7eb" stroke-width="1" opacity="0.5"/>
                    <text x="55" y="${yPos + 4}" text-anchor="end" fill="#64748b" font-size="11">${formatYLabel(yValue)}</text>
                    `;
                }).join('')}
                
                <!-- Threshold line (仅 Max Epsilon 显示) -->
                ${showThreshold && state.epsilonThreshold > 0 ? `
                    <line x1="60" y1="${scaleLeftY(state.epsilonThreshold)}" x2="380" y2="${scaleLeftY(state.epsilonThreshold)}" 
                        stroke="#ef4444" stroke-width="2" stroke-dasharray="5,5" opacity="0.8"/>
                    <text x="375" y="${scaleLeftY(state.epsilonThreshold) - 5}" text-anchor="end" fill="#ef4444" font-size="10" font-weight="bold">
                    Threshold
                    </text>
                ` : ''}
                
                <!-- Data line -->
                ${leftChartData.length > 0 ? `
                    <polyline fill="none" stroke="#2563eb" stroke-width="3"
                    points="${leftChartData.map((d, i) => 
                        `${60 + (i / Math.max(leftChartData.length - 1, 1)) * 320},${scaleLeftY(d.value)}`
                    ).join(' ')}" />
                    ${leftChartData.map((d, i) => `
                    <circle cx="${60 + (i / Math.max(leftChartData.length - 1, 1)) * 320}" cy="${scaleLeftY(d.value)}" r="4" fill="#2563eb" opacity="0.8">
                        <title>Iteration ${d.iteration}: ${formatYLabel(d.value)}</title>
                    </circle>
                    `).join('')}
                ` : ''}
                
                <!-- Axes -->
                <line x1="60" y1="250" x2="380" y2="250" stroke="#64748b" stroke-width="2"/>
                <line x1="60" y1="30" x2="60" y2="250" stroke="#64748b" stroke-width="2"/>
                <text x="220" y="270" text-anchor="middle" fill="#64748b" font-size="12">Iteration</text>
                </svg>
            </div>
            </div>
            
            <!-- Right Chart: Accuracy Only -->
            <div class="chart-section" style="background: transparent; border: none; padding: 0; padding-left: 20px;">
            <div class="chart-header" style="margin-top: 16px;">
                <h4 class="chart-title">Accuracy</h4>
            </div>
            <div class="line-chart" id="accuracyChart">
                <svg viewBox="0 0 400 280" class="chart-svg" preserveAspectRatio="xMidYMid meet">
                <!-- Grid lines -->
                ${Array.from({length: 6}, (_, i) => {
                    const yPos = 30 + i * 44;
                    const yValue = yMax - (i / 5) * yRange;
                    return `
                    <line x1="60" y1="${yPos}" x2="380" y2="${yPos}" stroke="#e5e7eb" stroke-width="1" opacity="0.5"/>
                    <text x="55" y="${yPos + 4}" text-anchor="end" fill="#64748b" font-size="11">${formatYLabel(yValue)}</text>
                    `;
                }).join('')}
                
                <!-- Accuracy line -->
                <polyline fill="none" stroke="#10b981" stroke-width="3"
                    points="${accuracyData.map((d, i) => 
                    `${60 + (i / Math.max(accuracyData.length - 1, 1)) * 320},${scaleY(d.value)}`
                    ).join(' ')}" />
                ${accuracyData.map((d, i) => `
                    <circle cx="${60 + (i / Math.max(accuracyData.length - 1, 1)) * 320}" cy="${scaleY(d.value)}" r="4" fill="#10b981" opacity="0.8">
                    <title>Iteration ${d.iteration} - Accuracy: ${formatYLabel(d.value)}</title>
                    </circle>
                `).join('')}
                
                <!-- Axes -->
                <line x1="60" y1="250" x2="380" y2="250" stroke="#64748b" stroke-width="2"/>
                <line x1="60" y1="30" x2="60" y2="250" stroke="#64748b" stroke-width="2"/>
                <text x="220" y="270" text-anchor="middle" fill="#64748b" font-size="12">Iteration</text>
                </svg>
            </div>
            </div>
            
        </div>
        </div>
        <div class="step-metrics">
        <div class="metric-card">
            <span class="metric-label">${leftChartLabel}</span>
            <span class="metric-value">${leftChartData.length > 0 ? formatYLabel(leftChartData[leftChartData.length - 1].value) : 'N/A'}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">Accuracy</span>
            <span class="metric-value">${accuracyScore}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">Total Time</span>
            <span class="metric-value">${totalTime}s</span>
        </div>
        </div>
    </div>
    
    <!-- History Section (Outside main window) -->
    <div style="margin-top: 16px;">
        <h4 style="margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: var(--text); text-transform: uppercase; letter-spacing: 0.5px;">Iteration History</h4>
        <div class="history-panel-external">
        <div class="history-list-external">
            ${state.history.length > 0 ? state.history.filter(h => h && h.metrics).map((h, i) => `
            <div class="history-item-external" onclick="showHistoryDetail(${i})">
                <span class="history-data">Iteration ${h.iteration !== undefined ? h.iteration : i}</span>
                <span class="history-divider">|</span>
                <span class="history-data">${config.useBiasMitigation && config.useAccuracyEnhancement ? 'BM+AE' : config.useBiasMitigation ? 'BM' : 'AE'}</span>
                <span class="history-divider">|</span>
                <span class="history-data">Fairness: ${h.metrics.Overall_Fairness !== undefined ? h.metrics.Overall_Fairness.toFixed(4) : 'N/A'}</span>
                <span class="history-divider">|</span>
                <span class="history-data">Acc: ${h.metrics.ACC !== undefined ? h.metrics.ACC.toFixed(4) : 'N/A'}</span>
            </div>
            `).join('') : '<div class="history-empty">No history yet</div>'}
        </div>
        </div>
    </div>
    </div>
    `;
    
    // 绑定 metric selector 事件监听器
    // 使用 setTimeout 确保 DOM 已完全渲染
    setTimeout(() => {
    const metricSelector = $('#metricSelector');
    if (metricSelector) {
        // 设置当前选中的 metric
        if (state.selectedMetric) {
        metricSelector.value = state.selectedMetric;
        } else {
        state.selectedMetric = metricSelector.value || 'Max_Epsilon';
        }
        
        // 监听变化（使用 onchange 避免重复绑定）
        metricSelector.onchange = function() {
        state.selectedMetric = this.value;
        console.log(`[DEBUG] Metric changed to: ${state.selectedMetric}`);
        // 重新渲染图表（保持当前step和realData）
        showProcessStep(stepIndex, true);
        };
    }
    }, 0);
}

function updateProcessStatus(status) {
    const statusEl = $('#processStatus');
    statusEl.textContent = status;
    statusEl.className = 'process-status status-' + status.toLowerCase().replace(/\s+/g, '-');
}

// 显示 history 详情
function showHistoryDetail(index) {
    if (index < 0 || index >= state.history.length) return;
    
    const h = state.history[index];
    if (!h || !h.metrics) {
    alert('No data available for this iteration.');
    return;
    }
    
    const mode = config.useBiasMitigation && config.useAccuracyEnhancement ? 'BM+AE' : config.useBiasMitigation ? 'BM' : 'AE';
    const modeFull = config.useBiasMitigation && config.useAccuracyEnhancement ? 'Bias Mitigation + Accuracy Enhancement' : config.useBiasMitigation ? 'Bias Mitigation' : 'Accuracy Enhancement';
    
    // Update modal title
    $('#historyDetailTitle').textContent = `Iteration ${h.iteration !== undefined ? h.iteration : index} Details`;
    
    // Build modal content
    const modalBody = $('#historyDetailBody');
    modalBody.innerHTML = `
    <!-- Main Metrics Cards -->
    <div class="history-detail-metrics-grid">
        <div class="history-detail-metric-card">
        <div class="history-detail-metric-header">
            <div class="history-detail-metric-icon history-detail-metric-icon--primary">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="4" x2="12" y2="20"></line>
                <line x1="5" y1="7" x2="19" y2="7"></line>
                <line x1="6" y1="11" x2="10" y2="11"></line>
                <line x1="14" y1="11" x2="18" y2="11"></line>
            </svg>
            </div>
            <div class="history-detail-metric-title">Overall Fairness</div>
        </div>
        <div class="history-detail-metric-value">${h.metrics?.Overall_Fairness?.toFixed(6) || 'N/A'}</div>
        </div>

        <div class="history-detail-metric-card">
        <div class="history-detail-metric-header">
            <div class="history-detail-metric-icon history-detail-metric-icon--success">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M9 12l2 2 4-4"></path>
            </svg>
            </div>
            <div class="history-detail-metric-title">Accuracy</div>
        </div>
        <div class="history-detail-metric-value">${h.metrics?.ACC?.toFixed(6) || 'N/A'}</div>
        </div>

        <div class="history-detail-metric-card">
        <div class="history-detail-metric-header">
            <div class="history-detail-metric-icon history-detail-metric-icon--warning">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            </div>
            <div class="history-detail-metric-title">Max Epsilon</div>
        </div>
        <div class="history-detail-metric-value">${h.current_max_epsilon !== undefined ? h.current_max_epsilon.toExponential(3) : 'N/A'}</div>
        </div>
    </div>

    <!-- Additional Metrics Section -->
    <div class="history-detail-section">
        <h4 class="history-detail-section-title">Performance Metrics</h4>
        <div class="history-detail-info-row">
        <span class="history-detail-info-label">F1 Score</span>
        <span class="history-detail-info-value">${h.metrics?.F1?.toFixed(6) || 'N/A'}</span>
        </div>
        <div class="history-detail-info-row">
        <span class="history-detail-info-label">Recall</span>
        <span class="history-detail-info-value">${h.metrics?.Recall?.toFixed(6) || 'N/A'}</span>
        </div>
        <div class="history-detail-info-row">
        <span class="history-detail-info-label">Precision</span>
        <span class="history-detail-info-value">${h.metrics?.Precision?.toFixed(6) || 'N/A'}</span>
        </div>
    </div>

    ${h.selected_attribute || h.selected_label_O ? `
    <!-- Iteration Details Section -->
    <div class="history-detail-section">
        <h4 class="history-detail-section-title">Iteration Details</h4>
        ${h.selected_attribute ? `
        <div class="history-detail-info-row">
        <span class="history-detail-info-label">Selected Attribute</span>
        <span class="history-detail-info-value">${h.selected_attribute}</span>
        </div>
        ` : ''}
        ${h.selected_label_O ? `
        <div class="history-detail-info-row">
        <span class="history-detail-info-label">Protected Attribute</span>
        <span class="history-detail-info-value">${h.selected_label_O}</span>
        </div>
        ` : ''}
    </div>
    ` : ''}
    `;
    
    // Show modal
    const modal = $('#historyDetailModal');
    modal.style.display = 'flex';
}

// Close history detail modal
function closeHistoryDetailModal() {
    const modal = $('#historyDetailModal');
    modal.style.display = 'none';
}

// Setup history detail modal events
$('#btnCloseHistoryDetail').addEventListener('click', closeHistoryDetailModal);
$('#btnCloseHistoryDetailFooter').addEventListener('click', closeHistoryDetailModal);
$('#historyDetailModalOverlay').addEventListener('click', closeHistoryDetailModal);
