function renderEmpty() {
// clear plot
pts2DG.selectAll('*').remove();
lbl2DG.selectAll('*').remove();
zeroG.selectAll('*').remove();
xAxisG.selectAll('*').remove();
yAxisG.selectAll('*').remove();

axes3DG.selectAll('*').remove();
cubeG.selectAll('*').remove();
pts3DG.selectAll('*').remove();
lbl3DG.selectAll('*').remove();

// header
el.chartTitle.textContent = '未選擇 iteration';
el.chartSubtitle.textContent = '請喺左邊撳一個 iteration 以顯示 scatterplot。';

// table
d3.select(el.tableHead).selectAll('th').remove();
d3.select(el.tableBody).selectAll('tr').remove();

renderAttrListState();
hideTooltip();
}

function renderChart() {
const model = state.model;
if (!model) return;

updateModeVisibility();

if (state.currentIteration == null) {
    renderEmpty();
    return;
}

const payload = buildActiveData();
if (!payload) {
    setStatus('Unable to render: fail to fetch iteration coordinates.', 'error');
    renderEmpty();
    return;
}

const { pts, labelsBefore, labelsAfter, iterI, beforeI, afterI } = payload;
const mode = getMode();

const axesText = state.selectedDims.map(d => dimLabel(d)).join(', ');
const modeText = (mode === '1d') ? `1D (${axesText})`
                : (mode === '2d') ? `2D (${axesText})`
                : `3D (${axesText})`;

el.chartTitle.textContent = `iteration ${iterI}`;
el.chartSubtitle.textContent =
    `before: iteration_${beforeI}  •  after: iteration_${afterI}`;

if (mode === '3d') render3D(payload, pts, labelsBefore, labelsAfter);
else render2D(payload, pts, labelsBefore, labelsAfter);

applyHighlight();
}

function render2D(payload, pts, labelsBefore, labelsAfter) {
const model = state.model;
const dims = state.selectedDims.slice();
const xDim = dims[0];
const yDim = dims[1] ?? null;

const xMeta = model.dims[xDim];
const yMeta = (yDim != null) ? model.dims[yDim] : null;

const baseX = d3.scaleLinear().domain(xMeta.extent).range([0, W]);
const baseY = (yDim != null)
    ? d3.scaleLinear().domain(yMeta.extent).range([H, 0])
    : d3.scaleLinear().domain([-1, 1]).range([H, 0]); // baseline for 1D

// zoom
const t = zoomAroundCenterTransform(state.zoomK);
const xScale = t.rescaleX(baseX);
const yScale = (yDim != null) ? t.rescaleY(baseY) : baseY;

const is1D = (getMode() === '1d');
const xAxisY = is1D ? yScale(0) : H;

xAxisG.attr('transform', `translate(0,${xAxisY})`);
yAxisG.attr('transform', `translate(0,0)`);

xLabelText.text(xMeta.label);
yLabelText.text(yDim != null ? yMeta.label : '');

// Exponent labels should follow current zoomed domain
const currXDomain = xScale.domain();
const xExp = expMultipleOf3FromDomain(currXDomain);
const xFactor = Math.pow(10, xExp);

let yExp = 0, yFactor = 1;
if (yDim != null) {
    const currYDomain = yScale.domain();
    yExp = expMultipleOf3FromDomain(currYDomain);
    yFactor = Math.pow(10, yExp);
}

const fmt = d3.format('~g');

xAxisG.call(
    d3.axisBottom(xScale)
    .ticks(6)
    .tickFormat(d => fmt(d / xFactor))
);

if (yDim != null) {
    yAxisG.style('display', null);
    yAxisG.call(
    d3.axisLeft(yScale)
        .ticks(6)
        .tickFormat(d => fmt(d / yFactor))
    );
} else {
    yAxisG.style('display', 'none');
}

xExpText
    .attr('x', W)
    .attr('y', xAxisY - 8)
    .text(xExp !== 0 ? `1e${xExp}` : '');

yExpText
    .attr('x', 0)
    .attr('y', -12)
    .text((yDim != null && yExp !== 0) ? `1e${yExp}` : '');

// Zero lines (2D only)
if (yDim != null) {
    const zeros = [];
    const xd = xScale.domain(), yd = yScale.domain();
    if (xd[0] <= 0 && xd[1] >= 0) zeros.push({type:'x0'});
    if (yd[0] <= 0 && yd[1] >= 0) zeros.push({type:'y0'});

    zeroG.style('display', null);

    zeroG.selectAll('line.zero-line')
    .data(zeros, d => d.type)
    .join(
        enter => enter.append('line').attr('class','zero-line'),
        update => update,
        exit => exit.remove()
    )
    .attr('x1', d => d.type === 'x0' ? xScale(0) : 0)
    .attr('x2', d => d.type === 'x0' ? xScale(0) : W)
    .attr('y1', d => d.type === 'y0' ? yScale(0) : 0)
    .attr('y2', d => d.type === 'y0' ? yScale(0) : H);
} else {
    zeroG.style('display', 'none');
}

function toXY(vec) {
    if (!vec || vec.length <= xDim) return null;
    const x = vec[xDim];
    if (!Number.isFinite(x)) return null;

    if (yDim == null) {
    return { x: xScale(x), y: yScale(0) };
    }

    if (vec.length <= yDim) return null;
    const y = vec[yDim];
    if (!Number.isFinite(y)) return null;

    return { x: xScale(x), y: yScale(y) };
}

// Points
const ptSel = pts2DG.selectAll('circle.pt')
    .data(pts, d => `${d.name}|${d.phase}`);

ptSel.join(
    enter => enter.append('circle')
    .attr('class', d => `pt ${d.phase}`)
    .attr('r', BASE_R)
    .on('mouseenter', (event, d) => {
        const dimsNow = state.selectedDims.slice();
        const row = payload.rowByName.get(d.name);
        showTooltip(tooltipHTML(d.name, d.phase, row, dimsNow), event.clientX, event.clientY);
    })
    .on('mousemove', (event, d) => {
        const dimsNow = state.selectedDims.slice();
        const row = payload.rowByName.get(d.name);
        showTooltip(tooltipHTML(d.name, d.phase, row, dimsNow), event.clientX, event.clientY);
    })
    .on('mouseleave', () => hideTooltip()),
    update => update,
    exit => exit.remove()
)
.attr('cx', d => (toXY(d.vec)?.x ?? -9999))
.attr('cy', d => (toXY(d.vec)?.y ?? -9999))
.style('display', d => toXY(d.vec) ? null : 'none');

// Labels (before + after, separately toggled)
const labelData = []
    .concat(state.showBeforeLabels ? labelsBefore : [])
    .concat(state.showAfterLabels ? labelsAfter : []);

const lblSel = lbl2DG.selectAll('text.label')
    .data(labelData, d => `${d.name}|${d.phase}`);

lblSel.join(
    enter => enter.append('text')
    .attr('class', d => `label ${d.phase}`),
    update => update,
    exit => exit.remove()
)
.text(d => d.name)
.attr('x', d => {
    const p = toXY(d.vec);
    if (!p) return -9999;
    return d.phase === 'before' ? (p.x - 8) : (p.x + 8);
})
.attr('y', d => {
    const p = toXY(d.vec);
    return p ? p.y : -9999;
})
.attr('text-anchor', d => d.phase === 'before' ? 'end' : 'start')
.style('display', d => toXY(d.vec) ? null : 'none');

applyHighlight();
}

function render3D(payload, pts, labelsBefore, labelsAfter) {
const model = state.model;
const dims = state.selectedDims.slice();
const xDim = dims[0], yDim = dims[1], zDim = dims[2];

const xMeta = model.dims[xDim];
const yMeta = model.dims[yDim];
const zMeta = model.dims[zDim];

function norm(v, meta) {
    const a = meta.extent[0], b = meta.extent[1];
    if (!Number.isFinite(v) || !Number.isFinite(a) || !Number.isFinite(b)) return 0;
    if (a === b) return 0;
    return ((v - a) / (b - a)) - 0.5; // [-0.5..0.5]
}

function rotate(p) {
    const cosY = Math.cos(state.rotY), sinY = Math.sin(state.rotY);
    const cosX = Math.cos(state.rotX), sinX = Math.sin(state.rotX);

    // yaw around Y
    const x1 = p.x * cosY + p.z * sinY;
    const z1 = -p.x * sinY + p.z * cosY;
    const y1 = p.y;

    // pitch around X
    const y2 = y1 * cosX - z1 * sinX;
    const z2 = y1 * sinX + z1 * cosX;

    return { x: x1, y: y2, z: z2 };
}

const centerX = W / 2;
const centerY = H / 2;
const plotScale = Math.min(W, H) * 0.92 * state.zoomK;
const cam = 2.5;

function project(p) {
    const r = rotate(p);
    const denom = (cam - r.z);
    const k = denom !== 0 ? (cam / denom) : 1;
    return {
    x: centerX + r.x * plotScale * k,
    y: centerY - r.y * plotScale * k,
    z: r.z
    };
}

function toXYZ(vec) {
    if (!vec) return null;
    if (vec.length <= Math.max(xDim, yDim, zDim)) return null;

    const x = vec[xDim], y = vec[yDim], z = vec[zDim];
    if (![x,y,z].every(Number.isFinite)) return null;

    return { x: norm(x, xMeta), y: norm(y, yMeta), z: norm(z, zMeta) };
}

// cube (unit cube centered at 0)
const c = [-0.5, 0.5];
const corners = [
    {x:c[0], y:c[0], z:c[0]}, {x:c[1], y:c[0], z:c[0]},
    {x:c[1], y:c[1], z:c[0]}, {x:c[0], y:c[1], z:c[0]},
    {x:c[0], y:c[0], z:c[1]}, {x:c[1], y:c[0], z:c[1]},
    {x:c[1], y:c[1], z:c[1]}, {x:c[0], y:c[1], z:c[1]}
];

const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
].map(([a,b]) => ({ a: corners[a], b: corners[b] }));

cubeG.selectAll('line.cube-edge')
    .data(edges)
    .join(
    enter => enter.append('line').attr('class','cube-edge'),
    update => update,
    exit => exit.remove()
    )
    .attr('x1', d => project(d.a).x)
    .attr('y1', d => project(d.a).y)
    .attr('x2', d => project(d.b).x)
    .attr('y2', d => project(d.b).y);

// 3D axes (origin + ticks + labels)
// Use cube corner (-0.5,-0.5,-0.5) as reference "origin" of bounding box.
const cubeOrigin = { x: -0.5, y: -0.5, z: -0.5 };
const xEnd = { x:  0.5, y: -0.5, z: -0.5 };
const yEnd = { x: -0.5, y:  0.5, z: -0.5 };
const zEnd = { x: -0.5, y: -0.5, z:  0.5 };

// tick generation in raw domain
function ticksFor(meta, count=5) {
    const d = meta.extent;
    return d3.ticks(d[0], d[1], count);
}

const xTicks = ticksFor(xMeta, 5).map(v => ({ axis:'x', v, p:{ x:norm(v,xMeta), y:-0.5, z:-0.5 } }));
const yTicks = ticksFor(yMeta, 5).map(v => ({ axis:'y', v, p:{ x:-0.5, y:norm(v,yMeta), z:-0.5 } }));
const zTicks = ticksFor(zMeta, 5).map(v => ({ axis:'z', v, p:{ x:-0.5, y:-0.5, z:norm(v,zMeta) } }));

const axisLines = [
    { axis:'x', a:cubeOrigin, b:xEnd, label:xMeta.label, meta:xMeta },
    { axis:'y', a:cubeOrigin, b:yEnd, label:yMeta.label, meta:yMeta },
    { axis:'z', a:cubeOrigin, b:zEnd, label:zMeta.label, meta:zMeta }
];

// Draw axis lines
axes3DG.selectAll('line.axis3d-line')
    .data(axisLines, d => d.axis)
    .join(
    enter => enter.append('line').attr('class','axis3d-line'),
    update => update,
    exit => exit.remove()
    )
    .attr('x1', d => project(d.a).x)
    .attr('y1', d => project(d.a).y)
    .attr('x2', d => project(d.b).x)
    .attr('y2', d => project(d.b).y);

// Origin dot
axes3DG.selectAll('circle.origin-dot')
    .data([cubeOrigin])
    .join(
    enter => enter.append('circle').attr('class','origin-dot'),
    update => update,
    exit => exit.remove()
    )
    .attr('r', 3.6)
    .attr('cx', d => project(d).x)
    .attr('cy', d => project(d).y);

// Axis label at end (with exponent hint)
axes3DG.selectAll('text.axis3d-label')
    .data(axisLines, d => d.axis)
    .join(
    enter => enter.append('text').attr('class','axis3d-label'),
    update => update,
    exit => exit.remove()
    )
    .text(d => {
    const exp = expMultipleOf3FromDomain(d.meta.extent);
    return exp !== 0 ? `${d.label} (1e${exp})` : d.label;
    })
    .attr('x', d => project(d.b).x + 6)
    .attr('y', d => project(d.b).y + 2);

// Tick marks + tick labels
const allTicks = xTicks.concat(yTicks, zTicks);

// tick line (short segment perpendicular-ish in screen space)
// We approximate by drawing tiny offset in y-direction in 3D local space.
function tickSegment(tp, axis) {
    const len = 0.03; // in cube units
    if (axis === 'x') return { a: tp, b: { x: tp.x, y: tp.y + len, z: tp.z } };
    if (axis === 'y') return { a: tp, b: { x: tp.x + len, y: tp.y, z: tp.z } };
    return { a: tp, b: { x: tp.x + len, y: tp.y, z: tp.z } };
}

axes3DG.selectAll('line.axis3d-tick')
    .data(allTicks, d => `${d.axis}|${d.v}`)
    .join(
    enter => enter.append('line').attr('class','axis3d-tick'),
    update => update,
    exit => exit.remove()
    )
    .attr('x1', d => project(tickSegment(d.p, d.axis).a).x)
    .attr('y1', d => project(tickSegment(d.p, d.axis).a).y)
    .attr('x2', d => project(tickSegment(d.p, d.axis).b).x)
    .attr('y2', d => project(tickSegment(d.p, d.axis).b).y);

function tickText(meta, v) {
    const exp = expMultipleOf3FromDomain(meta.extent);
    const factor = Math.pow(10, exp);
    return d3.format('~g')(v / factor);
}

axes3DG.selectAll('text.axis3d-ticktext')
    .data(allTicks, d => `${d.axis}|${d.v}`)
    .join(
    enter => enter.append('text').attr('class','axis3d-ticktext'),
    update => update,
    exit => exit.remove()
    )
    .text(d => {
    const meta = (d.axis === 'x') ? xMeta : (d.axis === 'y') ? yMeta : zMeta;
    return tickText(meta, d.v);
    })
    .attr('x', d => project(d.p).x + 6)
    .attr('y', d => project(d.p).y + 4);

// Points
const ptSel = pts3DG.selectAll('circle.pt')
    .data(pts, d => `${d.name}|${d.phase}`);

ptSel.join(
    enter => enter.append('circle')
    .attr('class', d => `pt ${d.phase}`)
    .attr('r', BASE_R)
    .on('mouseenter', (event, d) => {
        const dimsNow = state.selectedDims.slice();
        const row = payload.rowByName.get(d.name);
        showTooltip(tooltipHTML(d.name, d.phase, row, dimsNow), event.clientX, event.clientY);
    })
    .on('mousemove', (event, d) => {
        const dimsNow = state.selectedDims.slice();
        const row = payload.rowByName.get(d.name);
        showTooltip(tooltipHTML(d.name, d.phase, row, dimsNow), event.clientX, event.clientY);
    })
    .on('mouseleave', () => hideTooltip()),
    update => update,
    exit => exit.remove()
)
.attr('cx', d => {
    const p3 = toXYZ(d.vec);
    if (!p3) return -9999;
    return project(p3).x;
})
.attr('cy', d => {
    const p3 = toXYZ(d.vec);
    if (!p3) return -9999;
    return project(p3).y;
})
.style('display', d => toXYZ(d.vec) ? null : 'none');

// Labels (before + after, separately toggled)
const labelData = []
    .concat(state.showBeforeLabels ? labelsBefore : [])
    .concat(state.showAfterLabels ? labelsAfter : []);

const lblSel = lbl3DG.selectAll('text.label')
    .data(labelData, d => `${d.name}|${d.phase}`);

lblSel.join(
    enter => enter.append('text')
    .attr('class', d => `label ${d.phase}`),
    update => update,
    exit => exit.remove()
)
.text(d => d.name)
.attr('x', d => {
    const p3 = toXYZ(d.vec);
    if (!p3) return -9999;
    const p2 = project(p3);
    return d.phase === 'before' ? (p2.x - 8) : (p2.x + 8);
})
.attr('y', d => {
    const p3 = toXYZ(d.vec);
    if (!p3) return -9999;
    return project(p3).y;
})
.attr('text-anchor', d => d.phase === 'before' ? 'end' : 'start')
.style('display', d => toXYZ(d.vec) ? null : 'none');

applyHighlight();
}

function renderTable() {
const model = state.model;
if (!model) return;

if (state.currentIteration == null) {
    d3.select(el.tableHead).selectAll('th').remove();
    d3.select(el.tableBody).selectAll('tr').remove();
    return;
}

const payload = buildActiveData();
if (!payload) return;

const { rows } = payload;
const dims = state.selectedDims.slice();

const columns = [
    { key: 'name', label: 'attribute', cell: r => r.name },

    ...dims.map(dim => ({
    key: `b${dim}`,
    label: `before ${dimLabel(dim)}`,
    cell: r => {
        const v = r.before?.[dim];
        return (v == null) ? '' : fmtNumber(v);
    }
    })),

    ...dims.map(dim => ({
    key: `a${dim}`,
    label: `after ${dimLabel(dim)}`,
    cell: r => {
        const v = r.after?.[dim];
        return (v == null) ? '' : fmtNumber(v);
    }
    }))
];

d3.select(el.tableHead)
    .selectAll('th')
    .data(columns, d => d.key)
    .join(
    enter => enter.append('th'),
    update => update,
    exit => exit.remove()
    )
    .text(d => d.label);

const tr = d3.select(el.tableBody)
    .selectAll('tr')
    .data(rows, d => d.name);

tr.join(
    enter => enter.append('tr'),
    update => update,
    exit => exit.remove()
).each(function(row){
    const td = d3.select(this)
    .selectAll('td')
    .data(columns, c => c.key);

    td.join(
    enter => enter.append('td'),
    update => update,
    exit => exit.remove()
    ).text(c => c.cell(row));
});

applyTableHighlight();

// If attribute is pinned, keep it visible after re-render
if (state.selectedAttrName) scrollTableToAttribute(state.selectedAttrName);
}

function renderAll() {
renderAttrListState();
renderChart();
renderTable();
applyHighlight();
applyTableHighlight();
}
