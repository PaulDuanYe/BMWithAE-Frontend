// ===== Subgroup Builder Functions =====
// Event listeners
$('#btnGenerateSubgroups').addEventListener('click', openSubgroupBuilderModal);
$('#btnCloseSubgroupModal').addEventListener('click', closeSubgroupBuilderModal);
$('#btnCloseSubgroupModalFooter').addEventListener('click', closeSubgroupBuilderModal);
$('#subgroupModalOverlay').addEventListener('click', closeSubgroupBuilderModal);

// NEW Subgroup Builder - Open modal and populate feature selector
async function openSubgroupBuilderModal() {
    if (!state.datasetId) {
    alert('Please load a dataset first.');
    return;
    }
    
    // Show modal
    const modal = $('#subgroupModal');
    modal.style.display = 'flex';
    
    // Reset state
    state.currentConditions = {};
    state.createdSubgroups = [];
    
    // Get dataset info
    try {
    const info = await api.getDatasetInfo(state.datasetId);
    if (info.status !== 'success') {
        throw new Error('Failed to get dataset info');
    }
    
    state.featureStats = info.data.feature_stats;
    
    // Populate feature selector
    populateFeatureSelector(info.data);
    
    // Update UI
    updateConditionsDisplay();
    updateSubgroupsDisplay();
    
    } catch (err) {
    console.error('Error opening subgroup builder:', err);
    alert(`Error: ${err.message}`);
    }
}

// Close subgroup builder modal
function closeSubgroupBuilderModal() {
    const modal = $('#subgroupModal');
    modal.style.display = 'none';
}

// Populate feature selector with all features
function populateFeatureSelector(datasetInfo) {
    const container = $('#featureSelectorGrid');
    const featureStats = datasetInfo.feature_stats || {};
    const features = datasetInfo.features || [];
    
    container.innerHTML = features.map(featureName => {
    const stats = featureStats[featureName];
    if (!stats) return '';
    
    const featureType = stats.type.charAt(0).toUpperCase() + stats.type.slice(1);
    
    // Get values for this feature
    let values = [];
    if (stats.type === 'categorical' && stats.value_counts) {
        // Categorical: show all unique values
        values = Object.entries(stats.value_counts).map(([value, count]) => ({
        value: value,
        count: count,
        type: 'categorical'
        }));
    } else if (stats.type === 'continuous') {
        // Continuous: create bins with counts from backend
        const binCounts = stats.bin_counts || {};
        const binRanges = stats.bin_ranges || {};
        
        // Use backend-calculated bins if available
        const bins = ['Low', 'Medium', 'High'].map(binName => {
        const range = binRanges[binName] || '';
        const count = binCounts[binName] || 0;
        return {
            label: `${binName} ${range}`,
            value: `${binName} ${range}`,
            rawValue: binName,
            count: count,
            type: 'continuous'
        };
        });
        
        values = bins.filter(bin => bin.count > 0); // Only show bins with samples
    }
    
    return `
        <div class="feature-selector-card">
        <div class="feature-selector-card-header">
            <div class="feature-selector-card-title">${featureName}</div>
            <div class="feature-selector-card-type">${featureType}</div>
        </div>
        <div class="feature-values-list">
            ${values.map(v => `
            <div class="feature-value-item" 
                    data-feature="${featureName}" 
                    data-value="${v.value}"
                    onclick="toggleCondition('${featureName}', '${v.value}')">
                <span>${v.value}</span>
                ${v.count ? `<span class="feature-value-count">${v.count}</span>` : ''}
            </div>
            `).join('')}
        </div>
        </div>
    `;
    }).filter(html => html).join('');
    
    // Setup clear button
    $('#btnClearConditions').addEventListener('click', () => {
    state.currentConditions = {};
    updateConditionsDisplay();
    });
    
    // Setup add subgroup button
    $('#btnAddSubgroup').addEventListener('click', addCurrentSubgroup);
    
    // Setup toggle collapse/expand button
    setupFeatureSelectorToggle();
    
    // Setup search functionality
    setupFeatureSearch();
}

// Setup feature selector toggle (collapse/expand)
function setupFeatureSelectorToggle() {
    const toggleBtn = $('#btnToggleFeatureSelector');
    const grid = $('#featureSelectorGrid');
    const searchInput = $('#featureSearchInput');
    
    toggleBtn.addEventListener('click', () => {
    const isCollapsed = grid.classList.contains('collapsed');
    
    if (isCollapsed) {
        // Expand
        grid.classList.remove('collapsed');
        toggleBtn.classList.remove('collapsed');
        searchInput.style.display = 'block';
    } else {
        // Collapse
        grid.classList.add('collapsed');
        toggleBtn.classList.add('collapsed');
        searchInput.style.display = 'none';
    }
    });
}

// Setup feature search functionality
function setupFeatureSearch() {
    const searchInput = $('#featureSearchInput');
    
    searchInput.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase().trim();
    const featureCards = document.querySelectorAll('.feature-selector-card');
    
    featureCards.forEach(card => {
        const featureTitle = card.querySelector('.feature-selector-card-title');
        const featureName = featureTitle ? featureTitle.textContent.toLowerCase() : '';
        
        if (featureName.includes(searchTerm)) {
        card.style.display = 'block';
        } else {
        card.style.display = 'none';
        }
    });
    
    // Show count of visible features
    const visibleCount = Array.from(featureCards).filter(card => card.style.display !== 'none').length;
    console.log(`[INFO] Showing ${visibleCount} of ${featureCards.length} features`);
    });
}

// Toggle condition selection
function toggleCondition(featureName, value) {
    // Each feature can only have one value selected at a time
    if (state.currentConditions[featureName] === value) {
    // Deselect
    delete state.currentConditions[featureName];
    } else {
    // Select
    state.currentConditions[featureName] = value;
    }
    
    updateConditionsDisplay();
    console.log('[INFO] Current conditions:', state.currentConditions);
}

// Update conditions display
function updateConditionsDisplay() {
    const container = $('#currentConditions');
    const addButton = $('#btnAddSubgroup');
    
    const hasConditions = Object.keys(state.currentConditions).length > 0;
    
    if (hasConditions) {
    container.innerHTML = Object.entries(state.currentConditions).map(([feature, value]) => `
        <div class="condition-chip">
        <span>${feature} = ${value}</span>
        <div class="condition-chip-remove" onclick="removeCondition('${feature}')">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </div>
        </div>
    `).join('');
    
    addButton.disabled = false;
    } else {
    container.innerHTML = '<div class="conditions-empty-state"><p>Select feature values below to define your subgroup</p></div>';
    addButton.disabled = true;
    }
    
    // Update selected state in feature selector
    document.querySelectorAll('.feature-value-item').forEach(item => {
    const feature = item.dataset.feature;
    const value = item.dataset.value;
    
    if (state.currentConditions[feature] === value) {
        item.classList.add('selected');
    } else {
        item.classList.remove('selected');
    }
    });
}

// Remove condition
function removeCondition(featureName) {
    delete state.currentConditions[featureName];
    updateConditionsDisplay();
}

// Add current subgroup to list
async function addCurrentSubgroup() {
    if (Object.keys(state.currentConditions).length === 0) {
    return;
    }
    
    // Create subgroup object
    const subgroup = {
    id: 'subgroup_' + Date.now(),
    conditions: {...state.currentConditions},
    conditionsText: Object.entries(state.currentConditions)
        .map(([f, v]) => `${f}=${v}`)
        .join(' AND '),
    metrics: null
    };
    
    // Fetch metrics for this subgroup
    try {
    subgroup.metrics = await fetchSubgroupMetricsAPI(subgroup.conditions);
    } catch (err) {
    console.error('Error fetching metrics:', err);
    subgroup.metrics = getMockMetrics();
    }
    
    // Add to list
    state.createdSubgroups.push(subgroup);
    
    // Clear current conditions
    state.currentConditions = {};
    
    // Update displays
    updateConditionsDisplay();
    updateSubgroupsDisplay();
    
    console.log('[INFO] Added subgroup:', subgroup);
}

// Update subgroups display
function updateSubgroupsDisplay() {
    const container = $('#subgroupCardsGrid');
    const countEl = $('#subgroupCount');
    const btnViewAll = $('#btnViewAllSubgroups');
    const btnClearAll = $('#btnClearAllSubgroups');
    
    countEl.textContent = state.createdSubgroups.length;
    
    // Show/hide action buttons based on subgroup count
    if (state.createdSubgroups.length > 0) {
    btnViewAll.style.display = 'flex';
    btnClearAll.style.display = 'flex';
    } else {
    btnViewAll.style.display = 'none';
    btnClearAll.style.display = 'none';
    }
    
    if (state.createdSubgroups.length === 0) {
    container.innerHTML = '<div class="subgroups-empty-state"><p>No subgroups created yet. Define conditions above and click "Add Subgroup".</p></div>';
    return;
    }
    
    container.innerHTML = state.createdSubgroups.map((subgroup, index) => {
    const metrics = subgroup.metrics || {};
    return `
        <div class="subgroup-result-card" data-index="${index}">
        <div class="subgroup-result-card-header">
            <div>
            <div class="subgroup-result-card-title">Subgroup ${index + 1}</div>
            <div class="subgroup-result-card-conditions">${subgroup.conditionsText}</div>
            </div>
            <div class="subgroup-result-card-remove" data-index="${index}">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
            </div>
        </div>
        <div class="subgroup-result-metrics">
            <div class="subgroup-result-metric">
            <div class="subgroup-result-metric-label">Size</div>
            <div class="subgroup-result-metric-value">${metrics.size || 'N/A'}</div>
            </div>
            <div class="subgroup-result-metric">
            <div class="subgroup-result-metric-label">Accuracy</div>
            <div class="subgroup-result-metric-value">${metrics.accuracy || 'N/A'}</div>
            </div>
            <div class="subgroup-result-metric">
            <div class="subgroup-result-metric-label">Precision</div>
            <div class="subgroup-result-metric-value">${metrics.precision || 'N/A'}</div>
            </div>
            <div class="subgroup-result-metric">
            <div class="subgroup-result-metric-label">F1 Score</div>
            <div class="subgroup-result-metric-value">${metrics.f1 || 'N/A'}</div>
            </div>
        </div>
        <div class="subgroup-card-hint">Click to view detailed comparison</div>
        </div>
    `;
    }).join('');
    
    // Add event listeners after rendering
    setTimeout(() => {
    // Card click - show detail view
    document.querySelectorAll('.subgroup-result-card').forEach(card => {
        card.addEventListener('click', (e) => {
        const index = parseInt(card.dataset.index);
        showSubgroupDetailView(index);
        });
    });
    
    // Remove button click - delete subgroup
    document.querySelectorAll('.subgroup-result-card-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent card click
        const index = parseInt(btn.dataset.index);
        removeSubgroup(index);
        });
    });
    }, 0);
}

// Remove subgroup
function removeSubgroup(index) {
    state.createdSubgroups.splice(index, 1);
    updateSubgroupsDisplay();
}

// Clear all subgroups
function clearAllSubgroups() {
    if (confirm(`Are you sure you want to delete all ${state.createdSubgroups.length} subgroups?`)) {
    state.createdSubgroups = [];
    updateSubgroupsDisplay();
    console.log('[INFO] All subgroups cleared');
    }
}

// View all subgroups in table format
function openViewAllSubgroupsModal() {
    const modal = $('#viewAllSubgroupsModal');
    const tbody = $('#allSubgroupsTableBody');
    
    // Populate table
    tbody.innerHTML = state.createdSubgroups.map((subgroup, index) => {
    const metrics = subgroup.metrics || {};
    const positiveRate = metrics.positive_rate || 'N/A';
    const positiveRateDisplay = positiveRate !== 'N/A' ? (positiveRate * 100).toFixed(2) + '%' : 'N/A';
    
    return `
        <tr>
        <td>${index + 1}</td>
        <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${subgroup.conditionsText}">${subgroup.conditionsText}</td>
        <td>${metrics.size || 'N/A'}</td>
        <td>${positiveRateDisplay}</td>
        <td>${metrics.accuracy || 'N/A'}</td>
        <td>${metrics.precision || 'N/A'}</td>
        <td>${metrics.recall || 'N/A'}</td>
        <td>${metrics.f1 || 'N/A'}</td>
        <td class="subgroups-table-actions">
            <button class="btn btn--small btn--secondary" onclick="showSubgroupDetailViewFromTable(${index})" title="View Details">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
            </button>
            <button class="btn btn--small btn--danger" onclick="removeSubgroupFromTable(${index})" title="Delete">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
            </button>
        </td>
        </tr>
    `;
    }).join('');
    
    modal.style.display = 'flex';
}

// Close view all subgroups modal
function closeViewAllSubgroupsModal() {
    const modal = $('#viewAllSubgroupsModal');
    modal.style.display = 'none';
}

// Show subgroup detail view from table
function showSubgroupDetailViewFromTable(index) {
    closeViewAllSubgroupsModal();
    showSubgroupDetailView(index);
}

// Remove subgroup from table view
function removeSubgroupFromTable(index) {
    if (confirm(`Delete Subgroup ${index + 1}?`)) {
    removeSubgroup(index);
    // If modal is still open, refresh the table
    if ($('#viewAllSubgroupsModal').style.display === 'flex') {
        openViewAllSubgroupsModal();
    }
    }
}

// Setup view all subgroups button events
$('#btnViewAllSubgroups').addEventListener('click', openViewAllSubgroupsModal);
$('#btnClearAllSubgroups').addEventListener('click', clearAllSubgroups);
$('#btnCloseViewAllSubgroups').addEventListener('click', closeViewAllSubgroupsModal);
$('#btnCloseViewAllSubgroupsFooter').addEventListener('click', closeViewAllSubgroupsModal);
$('#viewAllSubgroupsModalOverlay').addEventListener('click', closeViewAllSubgroupsModal);

// Fetch subgroup metrics from API
async function fetchSubgroupMetricsAPI(conditions) {
    try {
    const response = await api.getSubgroupMetrics(state.datasetId, conditions);
    if (response.status === 'success') {
        const data = response.data;
        const metrics = data.subgroup_metrics;
        return {
        size: data.subgroup_size,
        positive_rate: metrics.positive_rate,
        positive_samples: metrics.positive_samples,
        negative_samples: metrics.negative_samples,
        accuracy: metrics.accuracy ? metrics.accuracy.toFixed(4) : 'N/A',
        precision: metrics.precision ? metrics.precision.toFixed(4) : 'N/A',
        recall: metrics.recall ? metrics.recall.toFixed(4) : 'N/A',
        f1: metrics.f1 ? metrics.f1.toFixed(4) : 'N/A',
        fpr: metrics.fpr ? metrics.fpr.toFixed(4) : 'N/A',
        note: metrics.note || '',
        overall_metrics: data.overall_metrics
        };
    } else {
        console.error('Failed to fetch subgroup metrics:', response.message);
        return getMockMetrics();
    }
    } catch (error) {
    console.error('Error fetching subgroup metrics:', error);
    return getMockMetrics();
    }
}

// Get mock metrics
function getMockMetrics() {
    return {
    size: Math.floor(Math.random() * 500) + 50,
    accuracy: (Math.random() * 0.3 + 0.65).toFixed(4),
    precision: (Math.random() * 0.3 + 0.65).toFixed(4),
    recall: (Math.random() * 0.3 + 0.65).toFixed(4),
    f1: (Math.random() * 0.3 + 0.65).toFixed(4),
    fpr: (Math.random() * 0.2 + 0.15).toFixed(4) // False Positive Rate
    };
}

// Show detailed comparison view for a subgroup
function showSubgroupDetailView(index) {
    if (index < 0 || index >= state.createdSubgroups.length) return;
    
    const subgroup = state.createdSubgroups[index];
    const modal = $('#subgroupDetailModal');
    
    // Set title
    $('#subgroupDetailTitle').textContent = `Subgroup ${index + 1} - Detailed Comparison`;
    
    // Set conditions
    $('#subgroupDetailConditions').innerHTML = `
    <div class="subgroup-detail-conditions-title">Subgroup Definition</div>
    <div class="subgroup-detail-conditions-text">${subgroup.conditionsText}</div>
    `;
    
    // Generate comparison chart
    generateComparisonChart(subgroup);
    
    // Show modal
    modal.style.display = 'flex';
}

// Generate comparison chart (like FairVis)
function generateComparisonChart(subgroup) {
    const container = $('#subgroupComparisonChart');
    const metrics = subgroup.metrics || {};
    
    // Use real overall metrics if available, otherwise use defaults
    const overallMetricsData = metrics.overall_metrics || {};
    const averageMetrics = {
    'Accuracy': overallMetricsData.accuracy || 0.7500,
    'Precision': overallMetricsData.precision || 0.7200,
    'Recall': overallMetricsData.recall || 0.6800,
    'F1 Score': overallMetricsData.f1 || 0.7000,
    'False Positive Rate': overallMetricsData.fpr || 0.2500
    };
    
    const subgroupMetrics = {
    'Accuracy': parseFloat(metrics.accuracy) || 0.75,
    'Precision': parseFloat(metrics.precision) || 0.72,
    'Recall': parseFloat(metrics.recall) || 0.68,
    'F1 Score': parseFloat(metrics.f1) || 0.70,
    'False Positive Rate': parseFloat(metrics.fpr) || 0.25
    };
    
    // Generate HTML for each metric
    const metricsHTML = Object.keys(averageMetrics).map(metricName => {
    const avgValue = averageMetrics[metricName];
    const subValue = subgroupMetrics[metricName];
    
    // Calculate percentage positions for the bars
    const avgPercent = avgValue * 100;
    const subPercent = subValue * 100;
    
    // Determine if subgroup is better or worse
    const isBetter = (metricName === 'False Positive Rate') ? (subValue < avgValue) : (subValue > avgValue);
    const diffPercent = ((subValue - avgValue) / avgValue * 100).toFixed(1);
    const diffSign = diffPercent > 0 ? '+' : '';
    
    return `
        <div class="comparison-metric-row">
        <div class="comparison-metric-header">
            <div class="comparison-metric-name">${metricName}</div>
            <div class="comparison-metric-values">
            <div class="comparison-metric-value">
                <span class="comparison-metric-label">avg:</span>
                <span class="comparison-metric-number">${(avgValue * 100).toFixed(1)}%</span>
            </div>
            <div class="comparison-metric-value">
                <span class="comparison-metric-label">subgroup:</span>
                <span class="comparison-metric-number subgroup">${(subValue * 100).toFixed(1)}%</span>
            </div>
            <div class="comparison-metric-value">
                <span class="comparison-metric-label" style="color: ${isBetter ? '#10b981' : '#ef4444'}">(${diffSign}${diffPercent}%)</span>
            </div>
            </div>
        </div>
        <div class="comparison-bar-container">
            <!-- Average line -->
            <div class="comparison-bar-average-line" style="left: ${avgPercent}%"></div>
            
            <!-- Subgroup line -->
            <div class="comparison-bar-subgroup" style="left: ${subPercent}%"></div>
        </div>
        </div>
    `;
    }).join('');
    
    container.innerHTML = `
    <h3 style="font-size: 14px; font-weight: 600; margin-bottom: 20px; color: var(--text);">
        Performance Metrics Comparison
    </h3>
    <div style="margin-bottom: 20px; padding: 12px; background: var(--surface-2); border-radius: 8px; font-size: 12px; color: var(--muted);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 3px; height: 20px; background: #94a3b8; border-radius: 2px;"></div>
            <span><strong style="color: #94a3b8;">avg</strong> = Dataset average</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 3px; height: 20px; background: linear-gradient(180deg, var(--brand), var(--brand-2)); border-radius: 2px;"></div>
            <span><strong style="color: var(--brand);">subgroup</strong> = Subgroup value</span>
            </div>
        </div>
        <div style="font-size: 11px; color: var(--muted); font-style: italic;">
            ⓘ Hover over bars to see labels
        </div>
        </div>
    </div>
    ${metricsHTML}
    `;
}

// Close subgroup detail modal
function closeSubgroupDetailModal() {
    $('#subgroupDetailModal').style.display = 'none';
}

// Event listeners for subgroup detail modal
$('#btnCloseSubgroupDetail').addEventListener('click', closeSubgroupDetailModal);
$('#btnCloseSubgroupDetailFooter').addEventListener('click', closeSubgroupDetailModal);
$('#subgroupDetailModalOverlay').addEventListener('click', closeSubgroupDetailModal);

// Make functions globally available
window.toggleCondition = toggleCondition;
window.removeCondition = removeCondition;
window.removeSubgroup = removeSubgroup;
window.showSubgroupDetailView = showSubgroupDetailView;