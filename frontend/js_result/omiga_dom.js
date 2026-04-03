export function createViewerDOM(container) {
  container.innerHTML = `
    <div class="viz">
      <div class="viz-top">
        <div>
          <div class="title" data-role="title">Loading…</div>
          <div class="subtitle" data-role="subtitle"></div>
        </div>
        <div class="controls" data-role="controls">
          <button data-action="toggle-after">After Label: On</button>
          <button data-action="toggle-before">Before Label: Off</button>
          <button data-action="zoom-in">Zoom In</button>
          <button data-action="zoom-out">Zoom Out</button>
          <button data-action="zoom-reset">Reset Zoom</button>
          <button data-action="reset-3d" class="hidden">Reset 3D</button>
        </div>
      </div>

      <svg data-role="chart"></svg>
      <div class="status" data-role="status"></div>
      <div class="table-wrap hidden" data-role="tableWrap">
        <table>
          <thead><tr data-role="tableHead"></tr></thead>
          <tbody data-role="tableBody"></tbody>
        </table>
      </div>
      <div class="tooltip hidden" data-role="tooltip"></div>
    </div>
  `;

  return {
    root: container,
    title: container.querySelector('[data-role="title"]'),
    subtitle: container.querySelector('[data-role="subtitle"]'),
    chart: container.querySelector('[data-role="chart"]'),
    status: container.querySelector('[data-role="status"]'),
    tooltip: container.querySelector('[data-role="tooltip"]'),
    tableWrap: container.querySelector('[data-role="tableWrap"]'),
    tableHead: container.querySelector('[data-role="tableHead"]'),
    tableBody: container.querySelector('[data-role="tableBody"]'),
    controls: {
      toggleAfter: container.querySelector('[data-action="toggle-after"]'),
      toggleBefore: container.querySelector('[data-action="toggle-before"]'),
      zoomIn: container.querySelector('[data-action="zoom-in"]'),
      zoomOut: container.querySelector('[data-action="zoom-out"]'),
      zoomReset: container.querySelector('[data-action="zoom-reset"]'),
      reset3D: container.querySelector('[data-action="reset-3d"]')
    }
  };
}