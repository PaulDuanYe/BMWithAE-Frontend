let epsilonChartInstance = null;

const EPSILON_MODE_LABELS = {
  max_epsilon: "Max Epsilon",
  iteration_compare: "Iteration Compare"
};

function generateEpsilonSelectorData() {
  const protectedAttrOptions = Array.isArray(state.protectedAttrs) ? state.protectedAttrs : [];

  const selectors = [
    {
      id: "epsilonModeSelector",
      name: "Mode",
      options: [
        "max_epsilon - Max Epsilon",
        "iteration_compare - Compare Iterations"
      ]
    },
    {
      id: "epsilonProtectedAttrSelector",
      name: "Protected Attribute - Compare Iterations",
      options: protectedAttrOptions
    }
  ];

  const header = document.getElementById("graphHeaderTitle");
  if (header) {
    header.textContent = "Epsilon Analysis";
  }

  createSelectors(selectors);
  attachEpsilonListeners();
  showParameterSelectionPanel();
  renderEpsilonGraph();
}

function attachEpsilonListeners() {
  document
    .getElementById("epsilonModeSelector")
    ?.addEventListener("change", renderEpsilonGraph);

  document
    .getElementById("epsilonProtectedAttrSelector")
    ?.addEventListener("change", renderEpsilonGraph);
}

function getSelectedEpsilonMode() {
  const selector = document.getElementById("epsilonModeSelector");
  if (!selector || !selector.value) return "max_epsilon";
  return selector.value.split(" - ")[0];
}

function getSelectedEpsilonProtectedAttr() {
  const selector = document.getElementById("epsilonProtectedAttrSelector");
  if (!selector || !selector.value) return null;
  return selector.value;
}

function renderEpsilonGraph() {
  const mode = getSelectedEpsilonMode();

  if (mode === "iteration_compare") {
    renderEpsilonIterationCompareMode();
    return;
  }

  renderEpsilonMaxMode();
}

function destroyEpsilonChartIfExists() {
  if (epsilonChartInstance) {
    epsilonChartInstance.destroy();
    epsilonChartInstance = null;
  }
}

function getIterationArray() {
  const iterations = state?.data?.iterations;
  return Array.isArray(iterations) ? iterations : [];
}

function getProtectedAttrBlock(iterationItem, protectedAttr) {
  return iterationItem?.epsilon_values
    ? getCaseInsensitive(iterationItem.epsilon_values, protectedAttr)
    : null;
}

function getEpsilonValuesObject(iterationItem, protectedAttr) {
  const block = getProtectedAttrBlock(iterationItem, protectedAttr);
  const epsilonObj = block?.epsilon_values;
  return epsilonObj && typeof epsilonObj === "object" && !Array.isArray(epsilonObj)
    ? epsilonObj
    : null;
}

function makeChartShell({ titleText = "", sidePanel = false }) {
  const container = document.getElementById("graphContent");
  if (!container) {
    throw new Error("graphContent container is required.");
  }

  container.innerHTML = "";
  destroyEpsilonChartIfExists();

  const shell = document.createElement("div");
  shell.style.width = "100%";
  shell.style.height = "100%";
  shell.style.display = "flex";
  shell.style.flexDirection = "column";
  shell.style.boxSizing = "border-box";
  shell.style.padding = "20px 24px 24px 24px";
  shell.style.gap = "16px";

  const title = document.createElement("div");
  title.textContent = titleText;
  title.style.fontSize = "18px";
  title.style.fontWeight = "700";
  title.style.color = "var(--text)";
  shell.appendChild(title);

  const body = document.createElement("div");
  body.style.flex = "1";
  body.style.minHeight = "0";
  body.style.display = "flex";
  body.style.gap = "16px";
  body.style.alignItems = "stretch";

  const chartWrap = document.createElement("div");
  chartWrap.style.flex = "1";
  chartWrap.style.minWidth = "0";
  chartWrap.style.minHeight = "0";
  chartWrap.style.background = "var(--surface)";
  chartWrap.style.border = "1px solid var(--border)";
  chartWrap.style.borderRadius = "12px";
  chartWrap.style.padding = "16px";
  chartWrap.style.boxSizing = "border-box";

  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  chartWrap.appendChild(canvas);

  body.appendChild(chartWrap);

  let rightPanel = null;

  if (sidePanel) {
    rightPanel = document.createElement("div");
    rightPanel.style.width = "220px";
    rightPanel.style.flex = "0 0 220px";
    rightPanel.style.background = "var(--surface)";
    rightPanel.style.border = "1px solid var(--border)";
    rightPanel.style.borderRadius = "12px";
    rightPanel.style.padding = "12px";
    rightPanel.style.boxSizing = "border-box";
    rightPanel.style.overflowY = "auto";
    body.appendChild(rightPanel);
  }

  shell.appendChild(body);
  container.appendChild(shell);

  return { container, shell, body, chartWrap, canvas, rightPanel };
}

function makeBluePalette(count) {
  const palette = [];
  const total = Math.max(count, 2);

  for (let i = 0; i < count; i++) {
    const hue = 210 + (i / Math.max(total - 1, 1)) * 35;
    const lightness = 42 + (i % 2) * 8;
    palette.push(`hsl(${hue}, 78%, ${lightness}%)`);
  }

  return palette;
}

function renderEpsilonMaxMode() {
  const iterations = state?.data?.iterations;
  if (!Array.isArray(iterations) || iterations.length === 0) {
    renderLineGraph({
      error: "No iteration data available."
    });
    return;
  }

  const firstIteration = iterations[0];
  if (!firstIteration || !firstIteration.epsilon_values) {
    renderLineGraph({
      error: 'iterations[0].epsilon_values is missing.'
    });
    return;
  }

  const protectedAttrNames = Object.keys(firstIteration.epsilon_values);
  if (!protectedAttrNames.length) {
    renderLineGraph({
      error: "No protected attributes found in epsilon_values."
    });
    return;
  }

  const xValues = iterations.map((_, i) => i.toString());
  const seriesMap = {};

  protectedAttrNames.forEach(protectedAttr => {
    seriesMap[protectedAttr] = iterations
      .map((iterationItem, iterationIndex) => {
        const protectedBlock = getCaseInsensitive(
          iterationItem.epsilon_values,
          protectedAttr
        );

        if (!protectedBlock || !protectedBlock.epsilon_values) {
          return null;
        }

        const epsilonObj = protectedBlock.epsilon_values;
        const values = Object.values(epsilonObj)
          .map(Number)
          .filter(Number.isFinite);

        if (!values.length) {
          return null;
        }

        return {
          x: iterationIndex,
          y: Math.max(...values)
        };
      })
      .filter(Boolean);
  });

  const seriesNames = Object.keys(seriesMap).filter(
    seriesName => Array.isArray(seriesMap[seriesName]) && seriesMap[seriesName].length > 0
  );

  if (!seriesNames.length) {
    renderLineGraph({
      error: "No valid epsilon data found."
    });
    return;
  }

  const graphData = {
    title: "Max epsilon across iterations",
    xAxisLabel: "Iteration",
    yAxisLabel: "Max Epsilon",
    xValues,
    seriesNames,
    seriesMap
  };

  renderLineGraph(graphData);
}

/*
Mode 2:
- no extra selector
- x-axis = feature names inside epsilon_values
- one line per iteration
- right-side button list
- hover/click button highlights that iteration
*/
function renderEpsilonIterationCompareMode() {
  const protectedAttr = getSelectedEpsilonProtectedAttr();
  const iterations = state?.data?.iterations;

  if (!protectedAttr) {
    throw new Error("[epsilon] Please select a protected attribute for Compare Iterations mode.");
  }

  if (!Array.isArray(iterations) || iterations.length === 0) {
    throw new Error("[epsilon] No iteration data available.");
  }

  if (typeof Chart === "undefined") {
    throw new Error("Chart.js is not loaded. Please include Chart.js before epsilon.js");
  }

  const featureSet = new Set();

  iterations.forEach((iterationItem, iterationIndex) => {
    const protectedBlock = getCaseInsensitive(iterationItem.epsilon_values, protectedAttr);

    if (!protectedBlock || !protectedBlock.epsilon_values) {
      throw new Error(
        `[epsilon] iteration ${iterationIndex}: missing epsilon_values for protected attribute "${protectedAttr}".`
      );
    }

    Object.keys(protectedBlock.epsilon_values).forEach(key => featureSet.add(key));
  });

  const featureNames = Array.from(featureSet);
  if (!featureNames.length) {
    throw new Error(`[epsilon] No feature names found for protected attribute "${protectedAttr}".`);
  }

  const { canvas, rightPanel } = makeChartShell({
    titleText: `Compare iterations across features (${protectedAttr})`,
    sidePanel: true
  });

  const labelsTitle = document.createElement("div");
  labelsTitle.textContent = "Iterations";
  labelsTitle.style.fontWeight = "700";
  labelsTitle.style.fontSize = "14px";
  labelsTitle.style.marginBottom = "6px";
  rightPanel.appendChild(labelsTitle);

  const labelsHint = document.createElement("div");
  labelsHint.textContent = "Hover or click to highlight";
  labelsHint.style.fontSize = "12px";
  labelsHint.style.color = "var(--muted)";
  labelsHint.style.marginBottom = "10px";
  rightPanel.appendChild(labelsHint);

  const buttonList = document.createElement("div");
  buttonList.style.display = "flex";
  buttonList.style.flexDirection = "column";
  buttonList.style.gap = "0";                 // removed gaps
  buttonList.style.margin = "0";
  buttonList.style.padding = "0";
  buttonList.style.borderRadius = "10px";
  buttonList.style.overflow = "hidden";       // smooth hover transition across items
  rightPanel.appendChild(buttonList);

  function makeIterationGradient(count) {
    const colors = [];
    const total = Math.max(count - 1, 1);

    for (let i = 0; i < count; i++) {
      const t = i / total;

      // red -> blue
      const r = Math.round(220 + (37 - 220) * t);
      const g = Math.round(38 + (99 - 38) * t);
      const b = Math.round(38 + (235 - 38) * t);

      colors.push(`rgb(${r}, ${g}, ${b})`);
    }

    return colors;
  }

  const colors = makeIterationGradient(iterations.length);

  const datasets = iterations.map((iterationItem, index) => {
    const protectedBlock = getCaseInsensitive(iterationItem.epsilon_values, protectedAttr);
    const epsilonObj = protectedBlock.epsilon_values;

    const data = featureNames.map(featureName => {
      const value = Number(epsilonObj[featureName]);
      return Number.isFinite(value) ? value : null;
    });

    return {
      label: index === 0 ? "raw" : `#${index}`,
      data,
      _i: index,
      _c: colors[index],
      borderColor: colors[index],
      backgroundColor: colors[index],
      borderWidth: 2,
      pointRadius: 3.5,
      pointHoverRadius: 5.5,
      pointBorderWidth: 2,
      pointBackgroundColor: "#fff",
      pointBorderColor: colors[index],
      tension: 0,
      spanGaps: false
    };
  });

  let pinnedIndex = null;
  let hoveredIndex = null;

  function focusIndex() {
    return hoveredIndex ?? pinnedIndex;
  }

  function fadedColor(color, alpha) {
    const m = color.match(/\d+/g);
    if (!m || m.length < 3) return color;

    const [r, g, b] = m.map(Number);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  const ctx = canvas.getContext("2d");

  epsilonChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: featureNames,
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: "nearest",
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          filter(context) {
            if (pinnedIndex == null) return true;
            return context.datasetIndex === pinnedIndex;
          },
          callbacks: {
            label(context) {
              const v = context.raw;
              const valueText = Number.isFinite(v) ? Number(v).toFixed(4) : "N/A";
              return `${context.dataset.label}: ${valueText}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            maxRotation: 35,
            minRotation: 35
          },
          title: {
            display: true,
            text: "Feature"
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Epsilon"
          }
        }
      },
      onHover(event, activeElements, chart) {
        if (pinnedIndex != null) {
          const onlyPinned = activeElements.filter(el => el.datasetIndex === pinnedIndex);
          chart.canvas.style.cursor = onlyPinned.length ? "pointer" : "default";
          return;
        }
        chart.canvas.style.cursor = activeElements.length ? "pointer" : "default";
      }
    }
  });

  function applyHighlightState() {
    const focused = focusIndex();

    epsilonChartInstance.data.datasets.forEach((ds, i) => {
      const base = ds._c;

      if (focused == null) {
        ds.order = 10;
        ds.borderWidth = 2;
        ds.pointRadius = 3.5;
        ds.pointHoverRadius = 5.5;
        ds.pointBorderWidth = 2;
        ds.borderColor = base;
        ds.pointBorderColor = base;
        ds.pointBackgroundColor = "#fff";
      } else if (i === focused) {
        ds.order = -100;
        ds.borderWidth = 4;
        ds.pointRadius = 7;
        ds.pointHoverRadius = 8;
        ds.pointBorderWidth = 3;
        ds.borderColor = base;
        ds.pointBorderColor = base;
        ds.pointBackgroundColor = "#fff";
      } else {
        ds.order = 10;
        ds.borderWidth = 1.1;
        ds.pointRadius = 2;
        ds.pointHoverRadius = 0;     // deactivate hover effect
        ds.pointBorderWidth = 1;
        ds.borderColor = fadedColor(base, 0.12);
        ds.pointBorderColor = fadedColor(base, 0.18);
        ds.pointBackgroundColor = "rgba(255,255,255,0.35)";
      }
    });

    Array.from(buttonList.children).forEach((btn, i) => {
      const isPinned = i === pinnedIndex;
      const isFocused = i === focused;

      btn.style.background = isPinned ? "#e8f1ff" : (isFocused ? "#f5f9ff" : "transparent");
      btn.style.borderColor = isPinned ? "#93c5fd" : "transparent";
      btn.style.transform = "scale(1)";
      btn.setAttribute("aria-pressed", isPinned ? "true" : "false");
    });

    epsilonChartInstance.update("none");
  }

  function makeButtonIcon(color) {
    const wrap = document.createElement("span");
    wrap.style.width = "42px";
    wrap.style.flex = "0 0 42px";
    wrap.style.display = "inline-flex";
    wrap.style.justifyContent = "center";

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 34 14");
    svg.style.width = "34px";
    svg.style.height = "14px";

    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", "1");
    line.setAttribute("y1", "7");
    line.setAttribute("x2", "33");
    line.setAttribute("y2", "7");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2");

    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("cx", "17");
    circle.setAttribute("cy", "7");
    circle.setAttribute("r", "4");
    circle.setAttribute("fill", "#fff");
    circle.setAttribute("stroke", color);
    circle.setAttribute("stroke-width", "2");

    svg.appendChild(line);
    svg.appendChild(circle);
    wrap.appendChild(svg);

    return wrap;
  }

  datasets.forEach((ds, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-pressed", "false");
    btn.style.width = "100%";
    btn.style.border = "1px solid transparent";
    btn.style.borderLeft = "0";
    btn.style.borderRight = "0";
    btn.style.borderTop = i === 0 ? "0" : "1px solid rgba(148,163,184,0.16)";
    btn.style.background = "transparent";
    btn.style.borderRadius = "0";          // no item gap look
    btn.style.padding = "10px 10px";
    btn.style.cursor = "pointer";
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.gap = "10px";
    btn.style.textAlign = "left";
    btn.style.transition = "background .12s ease, border-color .12s ease";

    btn.appendChild(makeButtonIcon(ds._c));

    const text = document.createElement("span");
    text.textContent = ds.label;
    btn.appendChild(text);

    btn.addEventListener("mouseenter", () => {
      hoveredIndex = i;
      applyHighlightState();
    });

    btn.addEventListener("mouseleave", () => {
      hoveredIndex = null;
      applyHighlightState();
    });

    btn.addEventListener("focus", () => {
      hoveredIndex = i;
      applyHighlightState();
    });

    btn.addEventListener("blur", () => {
      hoveredIndex = null;
      applyHighlightState();
    });

    btn.addEventListener("click", () => {
      pinnedIndex = pinnedIndex === i ? null : i;
      hoveredIndex = null;
      applyHighlightState();
    });

    buttonList.appendChild(btn);
  });

  applyHighlightState();
}

$('#btnSelectEpsilon').addEventListener('click', generateEpsilonSelectorData);