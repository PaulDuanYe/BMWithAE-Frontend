const fairnessMetricMap = {
  ACC: "Accuracy",
  F1: "F1 Score",
  Recall: "Recall",
  Precision: "Precision",
  BNC: "Between Negative Classes",
  BPC: "Between Positive Classes",
  CUAE: "Conditional Use Accuracy Equality",
  EOpp: "Equal Opportunity",
  EO: "Equalized Odds",
  FDRP: "False Discovery Rate Parity",
  FORP: "False Omission Rate Parity",
  FNRB: "False Negative Rate Balance",
  FPRB: "False Positive Rate Balance",
  NPVP: "Negative Predictive Value Parity",
  OAE: "Overall Accuracy Equality",
  PPVP: "Positive Predictive Value Parity",
  SP: "Statistical Parity"
};

function getFairnessMetrics() {
  const metrics = state?.data?.iterations?.[0]?.metrics;
  if (!metrics || typeof metrics !== "object") return [];
  return Object.keys(metrics);
}

function getFairnessMetricOptions() {
  return getFairnessMetrics().map(key => {
    const label = fairnessMetricMap[key] || key;
    return `${key} - ${label}`;
  });
}

function getSelectedFairnessMetric() {
  const selector = document.getElementById("fairnessMetricSelector");
  if (!selector || !selector.value) return null;
  return selector.value.split(" - ")[0];
}

function renderFairnessGraph() {
  const selectedMetric = getSelectedFairnessMetric();

  if (!selectedMetric) {
    renderLineGraph({
      error: "Please select a metric."
    });
    return;
  }

  const iterations = state?.data?.iterations;
  if (!Array.isArray(iterations) || iterations.length === 0) {
    renderLineGraph({
      error: "No iteration data available."
    });
    return;
  }

  const seriesMap = {};
  const xValues = iterations.map((_, index) => index);

  iterations.forEach((iterationItem, index) => {
    const metricValue = iterationItem?.metrics?.[selectedMetric];

    if (metricValue == null) return;

    // single-value metric: ACC / F1 / Recall / Precision ...
    if (typeof metricValue === "number") {
      if (!seriesMap[selectedMetric]) {
        seriesMap[selectedMetric] = [];
      }

      seriesMap[selectedMetric].push({
        x: index,
        y: Number(metricValue)
      });

      return;
    }

    // multi-series metric: BNC / BPC / EO / SP ...
    if (typeof metricValue === "object" && !Array.isArray(metricValue)) {
      Object.entries(metricValue).forEach(([seriesName, value]) => {
        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) return;

        if (!seriesMap[seriesName]) {
          seriesMap[seriesName] = [];
        }

        seriesMap[seriesName].push({
          x: index,
          y: numericValue
        });
      });
    }
  });

  const seriesNames = Object.keys(seriesMap);

  if (seriesNames.length === 0) {
    renderLineGraph({
      error: `No data found for metric "${selectedMetric}".`
    });
    return;
  }

  const graphData = {
    title: `${selectedMetric} across iterations`,
    xAxisLabel: "Iteration",
    yAxisLabel: fairnessMetricMap[selectedMetric] || selectedMetric,
    xValues,
    seriesNames,
    seriesMap
  };

  renderLineGraph(graphData);
}

function generateFairnessSelectorData() {
  const metricOptions = getFairnessMetricOptions();

  const selectors = [
    {
      id: "fairnessMetricSelector",
      name: "Metric",
      options: metricOptions
    }
  ];

  const header = document.getElementById("graphHeaderTitle");
  if (header) {
    header.textContent = "Fairness Metrics";
  }

  createSelectors(selectors);
  showParameterSelectionPanel();

  document
    .getElementById("fairnessMetricSelector")
    ?.addEventListener("change", renderFairnessGraph);

  renderFairnessGraph();
}

$('#btnSelectFairnessMatrix').addEventListener('click', generateFairnessSelectorData);