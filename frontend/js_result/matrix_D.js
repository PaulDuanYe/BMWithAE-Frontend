function getSelectedIterationMatirxD() {
    const iterationSelect = document.getElementById("iterationSelector");

    if (!iterationSelect) {
        console.error('Selector "iterationSelector" not found.');
        return null;
    }

    return Number(iterationSelect.value);
}

function getSelectedProtectedAttributeMatirxD() {
    const protectedAttrSelect = document.getElementById("protectedAttrSelector");

    if (!protectedAttrSelect) {
        return null;
    }

    return protectedAttrSelect.value;
}

function renderMatirxDGraph() {
  const container = document.getElementById("graphContent");
  if (!container) {
    throw new Error("container is required.");
  }

  container.innerHTML = "";

  const selectedIteration = getSelectedIterationMatirxD();
  const selectedProtectedAttr = getSelectedProtectedAttributeMatirxD();

  if (selectedIteration === null || selectedProtectedAttr === null) {
    return;
  }

  const matrix =
    getCaseInsensitive(
      state.data.iterations[selectedIteration].epsilon_values,
      selectedProtectedAttr
    )?.distance_matrix;

  if (!matrix || typeof matrix !== "object" || Array.isArray(matrix)) {
    console.log("matrix:", matrix);
    throw new Error("matrix must be a valid object.");
  }

  const labels = Object.keys(matrix);

  if (labels.length === 0) {
    console.log("matrix is empty:", matrix);
    throw new Error("matrix is empty.");
  }

  const rows = labels.length;
  const cols = labels.length;

  const values = [];
  let longestLabelLength = 0;

  for (const rowKey of labels) {
    const rowObj = matrix[rowKey];
    if (!rowObj || typeof rowObj !== "object") {
      throw new Error(`Row "${rowKey}" is invalid.`);
    }

    longestLabelLength = Math.max(longestLabelLength, String(rowKey).length);

    for (const colKey of labels) {
      values.push(Number(rowObj[colKey] ?? 0));
    }
  }

  const maxValue = Math.max(...values, 0);

  const containerWidth = Math.max(container.clientWidth || 1000, 700);
  const containerHeight = Math.max(container.clientHeight || 520, 420);

  // ---- sizing strategy ----
  // keep the whole graph visually centered and balanced
  const availableWidth = Math.max(containerWidth - 120, 520);
  const availableHeight = Math.max(containerHeight - 40, 320);

  const legendWidth = 54; // bar + labels + gap
  const estimatedLabelWidth = Math.max(
    120,
    Math.min(220, longestLabelLength * 8.6 + 22)
  );

  const maxCellWidthFromLayout = Math.floor(
    (availableWidth - estimatedLabelWidth - legendWidth) / cols
  );

  const maxCellHeightFromLayout = Math.floor(
    availableHeight / rows
  );

  let cellWidth = Math.max(26, Math.min(56, maxCellWidthFromLayout));
  let cellHeight = Math.max(24, Math.min(42, maxCellHeightFromLayout));

  // keep cells not too skinny compared to height
  if (cellWidth < cellHeight * 1.05) {
    cellWidth = Math.min(56, Math.round(cellHeight * 1.08));
  }

  // for very dense matrices, prefer compact square-ish blocks
  if (rows >= 30) {
    cellWidth = Math.min(cellWidth, 34);
    cellHeight = Math.min(cellHeight, 28);
  }

  if (rows >= 45) {
    cellWidth = Math.min(cellWidth, 28);
    cellHeight = Math.min(cellHeight, 24);
  }

  const labelWidth = Math.max(
    130,
    Math.min(240, longestLabelLength * 8.2 + 18)
  );

  const fullHeight = rows * cellHeight;
  const fullWidth = cols * cellWidth;

  let cellFontSize = 12;
  if (cellWidth <= 36 || cellHeight <= 28) cellFontSize = 10;
  if (cellWidth <= 30 || cellHeight <= 24) cellFontSize = 9;

  let labelFontSize = 14;
  if (cellHeight <= 28) labelFontSize = 13;
  if (cellHeight <= 24) labelFontSize = 12;

  let legendFontSize = 12;
  if (fullHeight < 320) legendFontSize = 11;

  const showCellText = cellWidth >= 28 && cellHeight >= 24 && rows <= 40;
  const tickCount = Math.min(11, Math.max(6, Math.round(fullHeight / 48)));

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function getColor(value) {
    const t = maxValue === 0 ? 0 : Math.max(0, Math.min(1, value / maxValue));

    let r, g, b;

    if (t < 0.25) {
      const k = t / 0.25;
      r = Math.round(lerp(234, 198, k));
      g = Math.round(lerp(244, 224, k));
      b = Math.round(lerp(255, 255, k));
    } else if (t < 0.5) {
      const k = (t - 0.25) / 0.25;
      r = Math.round(lerp(198, 144, k));
      g = Math.round(lerp(224, 194, k));
      b = Math.round(lerp(255, 255, k));
    } else if (t < 0.75) {
      const k = (t - 0.5) / 0.25;
      r = Math.round(lerp(144, 91, k));
      g = Math.round(lerp(194, 157, k));
      b = Math.round(lerp(255, 255, k));
    } else {
      const k = (t - 0.75) / 0.25;
      r = Math.round(lerp(91, 16, k));
      g = Math.round(lerp(157, 58, k));
      b = Math.round(lerp(255, 120, k));
    }

    return `rgb(${r}, ${g}, ${b})`;
  }

  function getTextColor(value) {
    if (maxValue === 0) return "#102a43";
    const t = value / maxValue;
    return t > 0.58 ? "#f7fbff" : "#102a43";
  }

  function formatValue(value) {
    if (value === 0) return "0";

    if (cellWidth >= 48) {
      if (value < 0.01) return value.toFixed(4);
      if (value < 1) return value.toFixed(3);
      return value.toFixed(3);
    }

    if (cellWidth >= 38) {
      if (value < 0.01) return value.toFixed(3);
      return value.toFixed(2);
    }

    if (value < 0.01) return value.toFixed(2);
    return value.toFixed(2);
  }

  const figure = document.createElement("div");
  figure.className = "matrixD-figure";

  const wrap = document.createElement("div");
  wrap.className = "matrixD-wrap";

  const composition = document.createElement("div");
  composition.className = "matrixD-composition";

  const main = document.createElement("div");
  main.className = "matrixD-main";

  const yLabelsEl = document.createElement("div");
  yLabelsEl.className = "matrixD-y-labels";
  yLabelsEl.style.gridTemplateRows = `repeat(${rows}, ${cellHeight}px)`;
  yLabelsEl.style.width = `${labelWidth}px`;

  const heatmapEl = document.createElement("div");
  heatmapEl.className = "heatmap";
  heatmapEl.style.gridTemplateColumns = `repeat(${cols}, ${cellWidth}px)`;
  heatmapEl.style.gridTemplateRows = `repeat(${rows}, ${cellHeight}px)`;

  const legendWrap = document.createElement("div");
  legendWrap.className = "legend-wrap";

  const legendBar = document.createElement("div");
  legendBar.className = "legend";
  legendBar.style.height = `${fullHeight}px`;

  const legendLabelsEl = document.createElement("div");
  legendLabelsEl.className = "legend-labels";
  legendLabelsEl.style.height = `${fullHeight}px`;
  legendLabelsEl.style.fontSize = `${legendFontSize}px`;

  const yLabelFragment = document.createDocumentFragment();
  const heatmapFragment = document.createDocumentFragment();
  const legendFragment = document.createDocumentFragment();

  for (const label of labels) {
    const div = document.createElement("div");
    div.className = "matrixD-y-label";
    div.textContent = label;
    div.title = label;
    div.style.width = `${labelWidth}px`;
    div.style.height = `${cellHeight}px`;
    div.style.fontSize = `${labelFontSize}px`;
    yLabelFragment.appendChild(div);
  }

  for (const rowKey of labels) {
    const rowObj = matrix[rowKey];

    for (const colKey of labels) {
      const value = Number(rowObj[colKey] ?? 0);

      const cell = document.createElement("div");
      cell.className = "cell";
      if (rowKey === colKey) {
        cell.classList.add("cell-diagonal");
      }

      cell.style.width = `${cellWidth}px`;
      cell.style.height = `${cellHeight}px`;
      cell.style.fontSize = `${cellFontSize}px`;
      cell.style.backgroundColor = getColor(value);
      cell.style.color = getTextColor(value);

      if (showCellText) {
        cell.textContent = formatValue(value);
      } else {
        cell.title = `${rowKey} × ${colKey}: ${value.toFixed(4)}`;
      }

      heatmapFragment.appendChild(cell);
    }
  }

  for (let i = 0; i < tickCount; i++) {
    const v = maxValue * (1 - i / (tickCount - 1));
    const d = document.createElement("div");
    d.textContent = Number(v.toFixed(3)).toString();
    legendFragment.appendChild(d);
  }

  yLabelsEl.appendChild(yLabelFragment);
  heatmapEl.appendChild(heatmapFragment);
  legendLabelsEl.appendChild(legendFragment);

  main.appendChild(yLabelsEl);
  main.appendChild(heatmapEl);

  legendWrap.appendChild(legendBar);
  legendWrap.appendChild(legendLabelsEl);

  composition.appendChild(main);
  composition.appendChild(legendWrap);

  wrap.appendChild(composition);
  figure.appendChild(wrap);
  container.appendChild(figure);
}

function generateMatrixDSelectorData() {
  const iterationOptions = Array.from(
    { length: state.data.iterations.length },
    (_, i) => i.toString()
  );
        
  const selectors =  [
    {
      id: "protectedAttrSelector",
      name: "Protected Attribute",
      options: state.protectedAttrs
    },
    {
      id: "iterationSelector",
      name: "Iteration",
      options: iterationOptions
    }
  ];

  const header = document.getElementById("graphHeaderTitle");
  header.textContent = 'Distance Matrix';
  createSelectors(selectors);
  attachMatrixDListeners();
  showParameterSelectionPanel();
  renderMatirxDGraph();
}

function attachMatrixDListeners() {
  const protectedSelector = document.getElementById("protectedAttrSelector");
  const iterationSelector = document.getElementById("iterationSelector");

  if (protectedSelector) {
    protectedSelector.addEventListener("change", renderMatirxDGraph);
  }

  if (iterationSelector) {
    iterationSelector.addEventListener("change", renderMatirxDGraph);
  }
}

$('#btnSelectDistanceMatrix').addEventListener('click', generateMatrixDSelectorData);