// example graphData structure:
/* const graphData = {
  title: "Model performance across iterations",
  xAxisLabel: "Iteration",
  yAxisLabel: "Score",
  xValues: [0, 1, 2, 3],
  seriesNames: ["ACC", "F1"],
  seriesMap: {
    ACC: [
      { x: 0, y: 0.81 },
      { x: 1, y: 0.84 },
      { x: 2, y: 0.86 },
      { x: 3, y: 0.88 }
    ],
    F1: [
      { x: 0, y: 0.76 },
      { x: 1, y: 0.80 },
      { x: 2, y: 0.82 },
      { x: 3, y: 0.85 }
    ]
  }
}; */

function renderLineGraph(graphData) {
  const container = document.getElementById("graphContent");
  if (!container) {
    throw new Error("graphContent container is required.");
  }

  container.innerHTML = "";

  if (!graphData || graphData.error) {
    container.textContent = graphData?.error || "No graph data available.";
    return;
  }

  const {
    title = "Trend graph",
    xAxisLabel = "X",
    yAxisLabel = "Y",
    xValues = [],
    seriesNames = [],
    seriesMap = {}
  } = graphData;

  const normalizedXValues = Array.isArray(xValues)
    ? xValues
    : Object.values(xValues || {});

  if (!Array.isArray(normalizedXValues) || normalizedXValues.length === 0) {
    container.textContent = "No x-axis data available.";
    return;
  }

  if (!Array.isArray(seriesNames) || seriesNames.length === 0) {
    container.textContent = "No series data available.";
    return;
  }

  const allValues = [];

  seriesNames.forEach(seriesName => {
    const data = seriesMap[seriesName];
    if (!Array.isArray(data)) return;

    data.forEach(point => {
      const numericY = Number(point?.y);
      if (Number.isFinite(numericY)) {
        allValues.push(numericY);
      }
    });
  });

  if (allValues.length === 0) {
    container.textContent = "No valid y-values available.";
    return;
  }

  const containerWidth = Math.max(container.clientWidth || 900, 700);
  const containerHeight = Math.max(container.clientHeight || 520, 420);

  const width = containerWidth;
  const height = Math.max(420, containerHeight);

  const hasMultipleLines = seriesNames.length > 1;

  const padding = {
    top: 40,
    right: hasMultipleLines ? 180 : 40,
    bottom: 56,
    left: 72
  };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const minY = 0;
  const maxRaw = Math.max(...allValues, 0);
  const maxY = maxRaw === 0 ? 1 : maxRaw * 1.08;

  const tickCountY = 6;
  const tickCountX = normalizedXValues.length;

  const svgNS = "http://www.w3.org/2000/svg";

  const wrapper = document.createElement("div");
  wrapper.style.width = "100%";
  wrapper.style.height = "100%";
  wrapper.style.display = "flex";
  wrapper.style.justifyContent = "center";
  wrapper.style.alignItems = "stretch";
  wrapper.style.padding = "20px 24px 24px 24px";
  wrapper.style.boxSizing = "border-box";

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.width = "100%";
  svg.style.height = "100%";
  svg.style.display = "block";
  svg.style.background = "var(--surface, #fff)";

  function createSvgEl(tag, attrs = {}) {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => {
      el.setAttribute(key, value);
    });
    return el;
  }

  function xScale(xIndex) {
    if (normalizedXValues.length <= 1) {
      return padding.left + chartWidth / 2;
    }
    return padding.left + (xIndex / (normalizedXValues.length - 1)) * chartWidth;
  }

  function yScale(value) {
    const t = (value - minY) / (maxY - minY);
    return padding.top + chartHeight - t * chartHeight;
  }

  function formatY(value) {
    if (value === 0) return "0";
    if (Math.abs(value) < 0.01) return value.toFixed(3);
    return value.toFixed(2);
  }

  const palette = [
    "#2563eb",
    "#0ea5e9",
    "#7c3aed",
    "#14b8a6",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4"
  ];

  // horizontal grid + y-axis labels
  for (let i = 0; i < tickCountY; i++) {
    const value = minY + (i / (tickCountY - 1)) * (maxY - minY);
    const y = yScale(value);

    svg.appendChild(createSvgEl("line", {
      x1: padding.left,
      y1: y,
      x2: padding.left + chartWidth,
      y2: y,
      stroke: "#e5e7eb",
      "stroke-width": "1"
    }));

    const label = createSvgEl("text", {
      x: padding.left - 10,
      y: y + 4,
      "text-anchor": "end",
      "font-size": "12",
      fill: "#64748b"
    });
    label.textContent = formatY(value);
    svg.appendChild(label);
  }

  // x-axis ticks + labels
  for (let i = 0; i < tickCountX; i++) {
    const x = xScale(i);

    svg.appendChild(createSvgEl("line", {
      x1: x,
      y1: padding.top + chartHeight,
      x2: x,
      y2: padding.top + chartHeight + 6,
      stroke: "#94a3b8",
      "stroke-width": "1"
    }));

    const label = createSvgEl("text", {
      x,
      y: padding.top + chartHeight + 22,
      "text-anchor": "middle",
      "font-size": "12",
      fill: "#64748b"
    });
    label.textContent = String(normalizedXValues[i]);
    svg.appendChild(label);
  }

  // axes
  svg.appendChild(createSvgEl("line", {
    x1: padding.left,
    y1: padding.top,
    x2: padding.left,
    y2: padding.top + chartHeight,
    stroke: "#94a3b8",
    "stroke-width": "1.5"
  }));

  svg.appendChild(createSvgEl("line", {
    x1: padding.left,
    y1: padding.top + chartHeight,
    x2: padding.left + chartWidth,
    y2: padding.top + chartHeight,
    stroke: "#94a3b8",
    "stroke-width": "1.5"
  }));

  // axis labels
  const xAxisText = createSvgEl("text", {
    x: padding.left + chartWidth / 2,
    y: height - 14,
    "text-anchor": "middle",
    "font-size": "13",
    fill: "#334155",
    "font-weight": "600"
  });
  xAxisText.textContent = xAxisLabel;
  svg.appendChild(xAxisText);

  const yAxisText = createSvgEl("text", {
    x: 20,
    y: padding.top + chartHeight / 2,
    transform: `rotate(-90 20 ${padding.top + chartHeight / 2})`,
    "text-anchor": "middle",
    "font-size": "13",
    fill: "#334155",
    "font-weight": "600"
  });
  yAxisText.textContent = yAxisLabel;
  svg.appendChild(yAxisText);

  const titleText = createSvgEl("text", {
    x: padding.left,
    y: 22,
    "text-anchor": "start",
    "font-size": "16",
    fill: "#0f172a",
    "font-weight": "700"
  });
  titleText.textContent = title;
  svg.appendChild(titleText);

  // draw lines
  seriesNames.forEach((seriesName, index) => {
    const color = palette[index % palette.length];
    const data = seriesMap[seriesName];

    if (!Array.isArray(data) || data.length === 0) return;

    const sortedData = data
      .filter(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)))
      .sort((a, b) => Number(a.x) - Number(b.x));

    if (sortedData.length === 0) return;

    const pathD = sortedData
      .map((point, i) => {
        const x = xScale(Number(point.x));
        const y = yScale(Number(point.y));
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ");

    const path = createSvgEl("path", {
      d: pathD,
      fill: "none",
      stroke: color,
      "stroke-width": "2.5",
      "stroke-linejoin": "round",
      "stroke-linecap": "round"
    });
    svg.appendChild(path);

    sortedData.forEach(point => {
      const cx = xScale(Number(point.x));
      const cy = yScale(Number(point.y));

const circle = createSvgEl("circle", {
  cx,
  cy,
  r: "5",
  fill: color,
  stroke: "#ffffff",
  "stroke-width": "1.5",
  style: "cursor: pointer"
});

// create tooltip div ONCE (reuse)
let tooltipEl = document.getElementById("graphTooltip");
if (!tooltipEl) {
  tooltipEl = document.createElement("div");
  tooltipEl.id = "graphTooltip";
  tooltipEl.style.position = "fixed";
  tooltipEl.style.pointerEvents = "none";
  tooltipEl.style.background = "rgba(15, 23, 42, 0.9)";
  tooltipEl.style.color = "#fff";
  tooltipEl.style.padding = "6px 10px";
  tooltipEl.style.borderRadius = "6px";
  tooltipEl.style.fontSize = "12px";
  tooltipEl.style.whiteSpace = "nowrap";
  tooltipEl.style.zIndex = "9999";
  tooltipEl.style.display = "none";
  document.body.appendChild(tooltipEl);
}

    // hover events
    circle.addEventListener("mousemove", (e) => {
    tooltipEl.style.display = "block";
    tooltipEl.style.left = e.clientX + 12 + "px";
    tooltipEl.style.top = e.clientY + 12 + "px";

    tooltipEl.innerHTML = `
        <strong>${seriesName}</strong><br/>
        ${xAxisLabel}: ${normalizedXValues[Number(point.x)]}<br/>
        ${yAxisLabel}: ${Number(point.y).toFixed(4)}
    `;
    });

    circle.addEventListener("mouseleave", () => {
    tooltipEl.style.display = "none";
    });

    svg.appendChild(circle);
    });
  });

  // legend only if multiple series
  if (hasMultipleLines) {
    const legendX = padding.left + chartWidth + 24;
    let legendY = padding.top + 8;

    const legendTitle = createSvgEl("text", {
      x: legendX,
      y: legendY,
      "font-size": "12",
      fill: "#64748b",
      "font-weight": "700"
    });
    legendTitle.textContent = "Series";
    svg.appendChild(legendTitle);

    legendY += 18;

    seriesNames.forEach((seriesName, index) => {
      const color = palette[index % palette.length];

      svg.appendChild(createSvgEl("line", {
        x1: legendX,
        y1: legendY,
        x2: legendX + 18,
        y2: legendY,
        stroke: color,
        "stroke-width": "3",
        "stroke-linecap": "round"
      }));

      svg.appendChild(createSvgEl("circle", {
        cx: legendX + 9,
        cy: legendY,
        r: "3.5",
        fill: color,
        stroke: "#fff",
        "stroke-width": "1"
      }));

      const text = createSvgEl("text", {
        x: legendX + 28,
        y: legendY + 4,
        "font-size": "12.5",
        fill: "#334155"
      });
      text.textContent = seriesName;
      svg.appendChild(text);

      legendY += 24;
    });
  }

  wrapper.appendChild(svg);
  container.appendChild(wrapper);
}