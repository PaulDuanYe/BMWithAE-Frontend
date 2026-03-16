function showDataExplorer() {
    const loadPanel = $('#loadDatasetPanel');
    const explorerPanel = $('#dataExplorerPanel');
    
    // Slide out load panel
    loadPanel.classList.add('slide-out');
    
    // Show and slide in explorer panel
    explorerPanel.style.display = 'block';
    setTimeout(() => {
    explorerPanel.classList.add('slide-in');
    
    // Initialize explorer content
    initializeDataExplorer();
    }, 50);
}

function hideDataExplorer() {
    const loadPanel = $('#loadDatasetPanel');
    const explorerPanel = $('#dataExplorerPanel');
    
    // Slide out explorer panel
    explorerPanel.classList.remove('slide-in');
    
    // Slide in load panel
    loadPanel.classList.remove('slide-out');
    
    // Hide explorer panel after animation
    setTimeout(() => {
    explorerPanel.style.display = 'none';
    }, 400);
}

async function initializeDataExplorer() {
    if (!state.datasetId) return;
    
    console.log('[DEBUG] Initializing data explorer for dataset:', state.datasetId);
    
    try {
    // Get dataset metadata from backend
    const response = await api.getDatasetInfo(state.datasetId);
    
    if (response.status === 'success' && response.data) {
        const datasetInfo = response.data;
        
        // Store dataset info for later use
        state.currentDatasetInfo = datasetInfo;
        
        // Populate protected attributes list (checkboxes)
        populateProtectedAttributesList(datasetInfo.features || []);
        
        // Get the selected protected attributes (priority ones will be auto-selected)
        const selectedAttrs = getSelectedProtectedAttributes();
        
        // Update the display bar with selected attributes
        updateProtectedAttrDisplay();
        
        // Display bias metrics overview for the selected attributes
        if (selectedAttrs.length > 0) {
        await displayBiasOverview(selectedAttrs);
        }
        
        // Render feature distributions
        renderFeatureDistributions(datasetInfo);
        
        // Setup modal event listeners
        $('#btnOpenAttrModal').addEventListener('click', openAttrModal);
        $('#btnCloseAttrModal').addEventListener('click', closeAttrModal);
        $('#btnCancelAttrModal').addEventListener('click', closeAttrModal);
        $('#btnConfirmAttrModal').addEventListener('click', confirmAttrSelection);
        
        // Close modal on overlay click
        $('#attrModalOverlay').addEventListener('click', (e) => {
        if (e.target === $('#attrModalOverlay')) {
            closeAttrModal();
        }
        });
    }
    } catch (err) {
    console.error('[ERROR] Failed to initialize data explorer:', err);
    }
}

function populateProtectedAttributesList(features) {
    const container = $('#protectedAttributesList');
    container.innerHTML = '';
    
    // Common protected attributes
    const protectedAttrs = ['sex', 'race', 'age', 'gender', 'ethnicity', 'marriage'];
    
    // Separate priority and other attributes
    const priorityFeatures = [];
    const otherFeatures = [];
    
    features.forEach(feature => {
    const featureLower = feature.toLowerCase();
    if (protectedAttrs.some(attr => featureLower.includes(attr))) {
        priorityFeatures.push(feature);
    } else {
        otherFeatures.push(feature);
    }
    });
    
    // Combine with priority first
    const sortedFeatures = [...priorityFeatures, ...otherFeatures];
    
    // Create checkbox for each attribute
    sortedFeatures.forEach((attr, index) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'protected-attr-item';
    itemDiv.dataset.attribute = attr;
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `protected-attr-${index}`;
    checkbox.value = attr;
    checkbox.className = 'protected-attr-checkbox';
    
    // Auto-select only SEX and MARRIAGE by default (not AGE)
    const defaultSelected = ['SEX', 'MARRIAGE'];
    if (defaultSelected.includes(attr.toUpperCase())) {
        checkbox.checked = true;
        itemDiv.classList.add('checked');
    }
    
    const label = document.createElement('label');
    label.htmlFor = `protected-attr-${index}`;
    label.textContent = attr;
    
    // Toggle on click
    itemDiv.addEventListener('click', (e) => {
        if (e.target !== checkbox) {
        checkbox.checked = !checkbox.checked;
        }
        if (checkbox.checked) {
        itemDiv.classList.add('checked');
        } else {
        itemDiv.classList.remove('checked');
        }
    });
    
    checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
        itemDiv.classList.add('checked');
        } else {
        itemDiv.classList.remove('checked');
        }
    });
    
    itemDiv.appendChild(checkbox);
    itemDiv.appendChild(label);
    container.appendChild(itemDiv);
    });
}

function calculateBiasScore(metrics) {
    /**
     * Calculate a composite bias score from 0-100
     * Higher score = more bias
     * 
     * Formula combines:
     * - Statistical Parity (0-1, lower is better)
     * - Equal Opportunity (0-1, lower is better)  
     * - Equalized Odds (0-1, lower is better)
     * - Disparate Impact (0-1, closer to 1 is better, convert to bias measure)
     */
    
    const sp = metrics.statistical_parity;
    const eo = metrics.equal_opportunity;
    const eodds = metrics.equalized_odds;
    const di = metrics.disparate_impact;
    
    // Convert disparate impact to bias measure (distance from perfect fairness at 1.0)
    const diBias = Math.abs(1.0 - di);
    
    // Weighted average (all metrics are 0-1 range, lower is better)
    // Weights: SP=30%, EO=25%, EOdds=25%, DI=20% (sum=100%)
    // Scale to 0-100
    const score = (sp * 0.3 + eo * 0.25 + eodds * 0.25 + diBias * 0.2) * 100;
    
    return Math.min(100, Math.max(0, score));
}

function getBiasScoreColor(score) {
    /**
     * Return color based on bias score (0-100)
     * Green (low bias) -> Yellow -> Orange -> Red (high bias)
     */
    if (score < 20) {
    return '#10b981'; // Green - Excellent
    } else if (score < 40) {
    return '#84cc16'; // Light Green - Good
    } else if (score < 60) {
    return '#eab308'; // Yellow - Moderate
    } else if (score < 80) {
    return '#f97316'; // Orange - High
    } else {
    return '#ef4444'; // Red - Very High
    }
}

function getBiasScoreDescription(score) {
    if (score < 20) {
    return 'Excellent - Very low bias detected';
    } else if (score < 40) {
    return 'Good - Low bias detected';
    } else if (score < 60) {
    return 'Moderate - Some bias present';
    } else if (score < 80) {
    return 'High - Significant bias detected';
    } else {
    return 'Very High - Severe bias detected';
    }
}

function updateBiasScoreCircle(score) {
    const circle = $('#biasScoreProgress');
    const text = $('#biasScoreText');
    const description = $('#biasScoreDescription');
    
    // Calculate stroke-dashoffset (534.07 is circumference of r=85 circle)
    const circumference = 534.07;
    const offset = circumference - (score / 100) * circumference;
    
    // Update circle
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = getBiasScoreColor(score);
    
    // Update text
    text.textContent = score.toFixed(0);
    text.style.fill = getBiasScoreColor(score);
    
    // Update description
    description.textContent = getBiasScoreDescription(score);
}

async function displayBiasOverview(protectedAttrs = null) {
    // Get all metric cards and score container
    const metricCards = document.querySelectorAll('.metric-card-small');
    const scoreContainer = $('.bias-score-container');
    
    // Handle both array and single value for backward compatibility
    if (!protectedAttrs || !state.datasetId) {
    // Display placeholder values if no protected attributes selected
    $('#metricSP').textContent = '--';
    $('#metricEO').textContent = '--';
    $('#metricEOdds').textContent = '--';
    $('#metricPP').textContent = '--';
    $('#biasScoreText').textContent = '--';
    $('#biasScoreDescription').textContent = 'Select protected attributes to calculate bias';
    
    // Reset circle
    const circle = $('#biasScoreProgress');
    circle.style.strokeDashoffset = '534.07';
    circle.style.stroke = '#10b981';
    
    return;
    }
    
    // Ensure protectedAttrs is an array
    const attrsArray = Array.isArray(protectedAttrs) ? protectedAttrs : [protectedAttrs];
    
    if (attrsArray.length === 0) {
    return;
    }
    
    try {
    // Add loading state
    metricCards.forEach(card => card.classList.add('loading'));
    scoreContainer.classList.add('loading');
    
    // Call backend with array of protected attributes
    const response = await api.getBiasMetrics(state.datasetId, attrsArray);
    
    if (response.status === 'success' && response.data && response.data.metrics) {
        const metrics = response.data.metrics;
        
        // Small delay for visual feedback
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Calculate composite bias score
        const biasScore = calculateBiasScore(metrics);
        
        console.log('[INFO] Bias score calculation:', {
        protectedAttributes: attrsArray,
        metrics: metrics,
        biasScore: biasScore
        });
        
        // Update bias score circle
        updateBiasScoreCircle(biasScore);
        
        // Update metric displays
        $('#metricSP').textContent = metrics.statistical_parity.toFixed(3);
        $('#metricEO').textContent = metrics.equal_opportunity.toFixed(3);
        $('#metricEOdds').textContent = metrics.equalized_odds.toFixed(3);
        $('#metricPP').textContent = metrics.disparate_impact.toFixed(3);
        
        console.log('[INFO] Updated bias metrics for protected attributes:', attrsArray, {
        biasScore,
        metrics
        });
    }
    } catch (err) {
    console.error('[ERROR] Failed to fetch bias metrics:', err);
    $('#metricSP').textContent = 'Error';
    $('#metricEO').textContent = 'Error';
    $('#metricEOdds').textContent = 'Error';
    $('#metricPP').textContent = 'Error';
    $('#biasScoreText').textContent = 'N/A';
    $('#biasScoreDescription').textContent = 'Failed to calculate bias score';
    } finally {
    // Remove loading state
    metricCards.forEach(card => card.classList.remove('loading'));
    scoreContainer.classList.remove('loading');
    }
}

function renderFeatureDistributions(datasetInfo) {
    const container = $('#featureDistributions');
    container.innerHTML = '';
    
    const featureStats = datasetInfo.feature_stats || {};
    const features = datasetInfo.features || [];
    
    features.forEach(featureName => {
    const stats = featureStats[featureName];
    if (stats) {
        const featureItem = createFeatureItem(featureName, stats);
        container.appendChild(featureItem);
    }
    });
}

function createFeatureItem(featureName, stats) {
    const item = document.createElement('div');
    item.className = 'feature-item feature-card';
    item.dataset.featureName = featureName;
    item.dataset.featureType = stats.type;
    
    // Generate distribution bars from real data
    let bars = [];
    let maxVal = 1;
    
    if (stats.type === 'categorical' && stats.value_counts) {
    // Use actual value counts for categorical features
    const values = Object.values(stats.value_counts);
    maxVal = Math.max(...values, 1);
    bars = values.slice(0, 10); // Limit to 10 bars
    } else {
    // For continuous features, generate representative bars
    bars = Array.from({length: 10}, () => Math.random() * 100);
    maxVal = Math.max(...bars);
    }
    
    // Capitalize feature type
    const featureType = stats.type.charAt(0).toUpperCase() + stats.type.slice(1);
    
    // Create the basic structure first
    item.innerHTML = `
    <div class="feature-item-header">
        <span class="feature-name">${featureName}</span>
        <span class="feature-type">${featureType}</span>
    </div>
    <div class="feature-chart">
        <div class="feature-bar-container">
        ${bars.map(val => 
            `<div class="feature-bar" style="height: ${(val / maxVal * 100)}%"></div>`
        ).join('')}
        </div>
    </div>
    <div class="feature-tooltip">
        <div class="tooltip-header">Feature: ${featureName}</div>
        <div class="tooltip-section">
        <div class="tooltip-section-title">Basic Stats</div>
        <div class="tooltip-metric">
            <span class="tooltip-metric-label">Type:</span>
            <span class="tooltip-metric-value">${featureType}</span>
        </div>
        <div class="tooltip-metric">
            <span class="tooltip-metric-label">Unique Values:</span>
            <span class="tooltip-metric-value">${stats.unique_values}</span>
        </div>
        <div class="tooltip-metric">
            <span class="tooltip-metric-label">Missing:</span>
            <span class="tooltip-metric-value">${stats.missing_percentage.toFixed(1)}%</span>
        </div>
        ${stats.mean !== undefined ? `
        <div class="tooltip-metric">
            <span class="tooltip-metric-label">Mean:</span>
            <span class="tooltip-metric-value">${stats.mean.toFixed(2)}</span>
        </div>
        ` : ''}
        </div>
        <div class="tooltip-section" id="bias-section-${featureName}">
        <div class="tooltip-section-title">Bias Metrics</div>
        <div class="tooltip-loading">Loading bias metrics...</div>
        </div>
    </div>
    `;
    
    // Position tooltip dynamically and load bias metrics on click
    const tooltip = item.querySelector('.feature-tooltip');
    let isTooltipVisible = false;
    
    // Create backdrop for blur effect
    let backdrop = document.getElementById('tooltip-backdrop');
    if (!backdrop) {
    backdrop = document.createElement('div');
    backdrop.id = 'tooltip-backdrop';
    backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        z-index: 99998;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
    `;
    document.body.appendChild(backdrop);
    }
    
    // Add visual feedback for clickability
    item.style.cursor = 'pointer';
    item.setAttribute('title', 'Click to view detailed bias metrics');
    
    // Handle click: show tooltip
    item.addEventListener('click', async (e) => {
    e.stopPropagation();
    
    // Toggle visibility
    if (isTooltipVisible) {
        // Hide tooltip and backdrop
        tooltip.style.opacity = '0';
        tooltip.style.pointerEvents = 'none';
        backdrop.style.opacity = '0';
        backdrop.style.pointerEvents = 'none';
        isTooltipVisible = false;
        return;
    }
    
    // Show tooltip
    isTooltipVisible = true;
    
    // Position tooltip centered on screen with larger size
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    
    // Calculate centered position using transform for perfect centering
    const tooltipWidth = 500;  // Larger width
    const tooltipMaxHeight = 700;  // Larger max height
    
    // Apply beautiful styles with centered positioning
    tooltip.style.position = 'fixed';
    tooltip.style.left = '50%';
    tooltip.style.top = '50%';
    tooltip.style.transform = 'translate(-50%, -50%)';
    tooltip.style.opacity = '1';
    tooltip.style.pointerEvents = 'auto';
    tooltip.style.display = 'block';
    tooltip.style.visibility = 'visible';
    tooltip.style.zIndex = '99999';
    tooltip.style.width = `${tooltipWidth}px`;
    tooltip.style.minHeight = '300px';
    tooltip.style.maxHeight = `${tooltipMaxHeight}px`;
    tooltip.style.overflowY = 'auto';
    tooltip.style.backgroundColor = 'rgba(255, 255, 255, 0.98)';
    tooltip.style.backdropFilter = 'blur(20px)';
    tooltip.style.webkitBackdropFilter = 'blur(20px)';
    tooltip.style.border = '1px solid rgba(200, 200, 200, 0.3)';
    tooltip.style.borderRadius = '12px';
    tooltip.style.padding = '20px';
    tooltip.style.boxShadow = '0 20px 60px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.5) inset';
    tooltip.style.color = '#1a1a1a';
    tooltip.style.fontSize = '14px';
    
    // Move to body to bypass container restrictions
    if (tooltip.parentElement !== document.body) {
        document.body.appendChild(tooltip);
    }
    
    // Show backdrop
    backdrop.style.opacity = '1';
    backdrop.style.pointerEvents = 'auto';
    
    // Load bias metrics
    // Note: tooltip might have been moved to body, so search in both places
    let biasSection = tooltip.querySelector(`#bias-section-${featureName}`);
    if (!biasSection) {
        biasSection = item.querySelector(`#bias-section-${featureName}`);
    }
    
    if (!biasSection) {
        console.error('[ERROR] Bias section not found for:', featureName);
        return;
    }
    
    const loadingEl = biasSection.querySelector('.tooltip-loading');
    
    // Only load once
    if (loadingEl && !biasSection.dataset.loaded) {
        try {
        const response = await api.getBiasMetrics(state.datasetId, [featureName]);
        if (response.status === 'success' && response.data) {
            // Backend returns: {data: {metrics: {...}, protected_attributes: [...]}}
            const metrics = response.data.metrics || response.data;
            biasSection.dataset.loaded = 'true';
            
            console.log('[DEBUG] Bias metrics for', featureName, ':', metrics);
            
            // Display bias metrics summary
            let biasHTML = `
            <div class="tooltip-metric">
                <span class="tooltip-metric-label">Statistical Parity:</span>
                <span class="tooltip-metric-value">${metrics.statistical_parity?.toFixed(4) || 'N/A'}</span>
            </div>
            <div class="tooltip-metric">
                <span class="tooltip-metric-label">Equal Opportunity:</span>
                <span class="tooltip-metric-value">${metrics.equal_opportunity?.toFixed(4) || 'N/A'}</span>
            </div>
            <div class="tooltip-metric">
                <span class="tooltip-metric-label">Equalized Odds:</span>
                <span class="tooltip-metric-value">${metrics.equalized_odds?.toFixed(4) || 'N/A'}</span>
            </div>
            <div class="tooltip-metric">
                <span class="tooltip-metric-label">Disparate Impact:</span>
                <span class="tooltip-metric-value">${metrics.disparate_impact?.toFixed(4) || 'N/A'}</span>
            </div>
            `;
            
            // Get group_positive_rates from individual_metrics if available
            let groupRates = metrics.group_positive_rates;
            if (!groupRates && metrics.individual_metrics && metrics.individual_metrics.length > 0) {
            groupRates = metrics.individual_metrics[0].group_positive_rates;
            }
            
            console.log('[DEBUG] Group rates:', groupRates);
            
            // Display subgroup positive rates if available
            if (groupRates && Object.keys(groupRates).length > 0) {
            biasHTML += `
                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(0,0,0,0.1);">
                <div style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
                    Subgroup Analysis (Top 10)
                </div>
            `;
            
            const rates = groupRates;
            const maxRate = Math.max(...Object.values(rates));
            
            // Sort subgroups by positive rate (descending) and take top 10
            const sortedGroups = Object.entries(rates)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);  // Only show top 10
            
            console.log('[DEBUG] Rendering', sortedGroups.length, 'subgroups');
            
            sortedGroups.forEach(([group, rate]) => {
                const percentage = (rate * 100).toFixed(1);
                const barWidth = maxRate > 0 ? (rate / maxRate * 100) : 0;
                
                biasHTML += `
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 13px; color: #1a1a1a; font-weight: 500;">${group}</span>
                    <span style="font-size: 12px; color: #64748b;">${percentage}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                    <div style="width: ${barWidth}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #2563eb); border-radius: 4px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                `;
            });
            
            biasHTML += `</div>`;
            }
            
            console.log('[DEBUG] Setting HTML, length:', biasHTML.length);
            loadingEl.innerHTML = biasHTML;
            console.log('[DEBUG] HTML set successfully');
        } else {
            loadingEl.textContent = 'Failed to load bias metrics';
        }
        } catch (err) {
        console.error('Error loading bias metrics:', err);
        loadingEl.textContent = 'Error loading metrics';
        }
    }
    });
    
    // Close tooltip when clicking backdrop
    backdrop.addEventListener('click', () => {
    tooltip.style.opacity = '0';
    tooltip.style.pointerEvents = 'none';
    backdrop.style.opacity = '0';
    backdrop.style.pointerEvents = 'none';
    isTooltipVisible = false;
    });
    
    return item;
}

// Event Listeners for Data Explorer
$('#btnBackToLoad').addEventListener('click', hideDataExplorer);

// Scroll hint click handler
$('#scrollHint').addEventListener('click', () => {
    const featureSection = document.querySelector('.feature-list');
    if (featureSection) {
    featureSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});