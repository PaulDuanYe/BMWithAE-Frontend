function parseModel(raw) {
if (!raw || typeof raw !== 'object') throw new Error('JSON root must be an object.');

// attributes
let attributes = [];
if (Array.isArray(raw.attributes)) {
    attributes = raw.attributes.map((a, idx) => {
    if (typeof a === 'string') return { index: idx, name: a };
    if (a && typeof a === 'object') {
        const name = a.name ?? a.label ?? a.id ?? `attr.${idx+1}`;
        return { index: idx, name: String(name) };
    }
    return { index: idx, name: `attr.${idx+1}` };
    });
}

// iterations
const iterKeyRegex = /^iteration_(\d+)_coordinates$/;
const iterEntries = Object.keys(raw)
    .filter(k => iterKeyRegex.test(k))
    .map(k => ({ i: Number(k.match(iterKeyRegex)[1]), rawCoords: raw[k] }))
    .sort((a, b) => a.i - b.i);

if (iterEntries.length === 0) {
    throw new Error('Cannot find iteration_*_coordinates (e.g. iteration_0_coordinates).');
}

const iterations = iterEntries.map(entry => {
    const coordsMap = normalizeCoords(entry.rawCoords, attributes);
    const transformedAttrName = getTransformedAttrName(raw, entry.i, attributes);
    return { i: entry.i, coords: coordsMap, transformedAttrName };
});

// If attributes not provided, infer from keys
if (attributes.length === 0) {
    const allNames = new Set();
    for (const it of iterations) for (const name of it.coords.keys()) allNames.add(name);
    attributes = Array.from(allNames).sort().map((name, idx) => ({ index: idx, name }));
}

const attrNames = attributes.map(d => d.name);

// Ensure each iteration has all keys (missing => null)
for (const it of iterations) {
    for (const name of attrNames) if (!it.coords.has(name)) it.coords.set(name, null);
}

// Determine dimension count
let dimCount = 1;
for (const it of iterations) {
    for (const vec of it.coords.values()) {
    if (vec && vec.length > dimCount) dimCount = vec.length;
    }
}

// Global extents per dim
const dimMin = Array(dimCount).fill(Number.POSITIVE_INFINITY);
const dimMax = Array(dimCount).fill(Number.NEGATIVE_INFINITY);

for (const it of iterations) {
    for (const vec of it.coords.values()) {
    if (!vec) continue;
    for (let d = 0; d < Math.min(dimCount, vec.length); d++) {
        const v = vec[d];
        if (!Number.isFinite(v)) continue;
        if (v < dimMin[d]) dimMin[d] = v;
        if (v > dimMax[d]) dimMax[d] = v;
    }
    }
}

const dims = [];
for (let d = 0; d < dimCount; d++) {
    let ext = [dimMin[d], dimMax[d]];
    if (!Number.isFinite(ext[0]) || !Number.isFinite(ext[1])) ext = [-1, 1];
    const nice = d3.scaleLinear().domain(padExtent(ext)).nice(6).domain();

    const exp = expMultipleOf3FromDomain(nice);
    const factor = Math.pow(10, exp);

    dims.push({
    index: d,
    label: dimLabel(d),
    extent: nice,
    exp,
    factor
    });
}

const maxI = d3.max(iterations, d => d.i);
const minI = d3.min(iterations, d => d.i);
const iterMap = new Map(iterations.map(it => [it.i, it]));

return { attributes, attrNames, iterations, iterMap, maxI, minI, dimCount, dims };
}

function parseVector(v) {
if (v == null) return null;

if (Array.isArray(v)) {
    if (v.length === 0) return null;
    const arr = v.map(Number);
    if (arr.some(n => !Number.isFinite(n))) return null;
    return arr;
}

if (typeof v === 'object') {
    if (Array.isArray(v.values)) return parseVector(v.values);
    const keys = ['x','y','z','w'];
    if (keys.some(k => k in v)) {
    const arr = keys.filter(k => k in v).map(k => Number(v[k]));
    if (arr.length === 0 || arr.some(n => !Number.isFinite(n))) return null;
    return arr;
    }
}

return null;
}

function normalizeCoords(rawCoords, attributes) {
const map = new Map();

if (rawCoords && !Array.isArray(rawCoords) && typeof rawCoords === 'object') {
    for (const [name, v] of Object.entries(rawCoords)) {
    const vec = parseVector(v);
    if (vec) map.set(String(name), vec);
    }
    return map;
}

if (Array.isArray(rawCoords)) {
    for (let idx = 0; idx < rawCoords.length; idx++) {
    const row = rawCoords[idx];

    if (Array.isArray(row)) {
        const name = attributes[idx]?.name ?? String(idx);
        const vec = parseVector(row);
        if (vec) map.set(name, vec);
        continue;
    }

    if (row && typeof row === 'object') {
        const name = row.name ?? row.label ?? row.id ?? attributes[idx]?.name ?? String(idx);
        const vec = parseVector(row.values ?? row.coords ?? row.vector ?? row);
        if (vec) map.set(String(name), vec);
        continue;
    }
    }
    return map;
}

return map;
}

function dimLabel(index) {
return `bias axis ${index + 1}`;
}

function fmtNumber(x) {
if (!Number.isFinite(x)) return '';
return d3.format('.6~g')(x);
}
  function fmtCoord(vec, dims) {
    if (!vec) return '—';
    const parts = dims.map(dim => {
      const v = vec?.[dim];
      return (v == null || !Number.isFinite(v)) ? '—' : fmtNumber(v);
    });
    return '(' + parts.join(', ') + ')';
  }
