
metricSelector.innerHTML = `
  <option value="Max_Epsilon" ${currentSelectedMetric === 'Max_Epsilon' ? 'selected' : ''}>Max Epsilon</option>
  <option value="Overall_Fairness" ${currentSelectedMetric === 'Overall_Fairness' ? 'selected' : ''}>Overall Fairness Score</option>
  <optgroup label="Individual Metrics">
    ${config.evalMetricFairness.map(metric => 
    `<option value="${metric}" ${currentSelectedMetric === metric ? 'selected' : ''}>${metric} - ${metricDescriptions[metric]}</option>`
    ).join('')}
  </optgroup>`;