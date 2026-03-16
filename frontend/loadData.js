function displayFileName(file){
  const fileList = $('#fileList');
  const fileSize = (file.size / 1024).toFixed(2) + ' KB';
  fileList.innerHTML = `
    <div class="file-item">
      <span class="file-item__icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </span>
      <span class="file-item__name">${file.name}</span>
      <span class="file-item__size">${fileSize}</span>
    </div>
  `;
}

function handleFile(file){
  if(!file || !file.name.endsWith('.csv')){
    alert('Please select a valid CSV file.');
    return;
  }
  state.currentFile = file;
  displayFileName(file);
  $('#dropzone').classList.add('has-file');
}

function showLoadStatus(elementId, message, type = 'success') {
  const statusEl = $(elementId);
  statusEl.style.display = 'flex';
  statusEl.className = `load-status load-status--${type}`;
  statusEl.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      ${type === 'success' ? 
        '<polyline points="20 6 9 17 4 12"/>' : 
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
      }
    </svg>
    <span style="flex: 1;">${message}</span>
    <button class="load-status-close" onclick="this.parentElement.style.display='none'" title="Dismiss">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
    </button>
  `;
  
  // Status remains visible until user dismisses it
}

async function importData(){
  if(!state.currentFile){
    alert('Please select a file first.');
    return;
  }
  
  const btn = $('#btnImport');
  const btnText = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');
  
  // Show loading state
  btn.disabled = true;
  btnText.style.display = 'none';
  btnSpinner.style.display = 'inline-block';
  
  try {
    // Need to prompt for target and protected columns
    // For demo, use default columns for credit dataset
    const result = await api.uploadDataset(
      state.currentFile,
      'default payment next month',  // target column
      ['SEX', 'MARRIAGE']  // protected columns
    );
    
    if (result.status === 'success') {
      state.datasetId = result.data.dataset_id;
      state.currentData = result.data.filename;
      
      showLoadStatus('#loadStatus', 
        `Successfully loaded ${result.data.rows.toLocaleString()} rows and ${result.data.columns} columns from ${result.data.filename}`,
        'success'
      );
    } else {
      throw new Error(result.message);
    }
  } catch(err) {
    showLoadStatus('#loadStatus', 
      `Error loading file: ${err.message}`,
      'error'
    );
  } finally {
    // Reset button state
    btn.disabled = false;
    btnText.style.display = 'inline';
    btnSpinner.style.display = 'none';
  }
}

async function loadDemoData(demoKey){
  const btn = $('#btnLoadDemo');
  const btnText = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');
  
  // Show loading state
  btn.disabled = true;
  btnText.style.display = 'none';
  btnSpinner.style.display = 'inline-block';
  
  try {
    const result = await api.loadDemo(demoKey);
    
    if (result.status === 'success') {
      state.datasetId = result.data.dataset_id;
      state.currentData = result.data.filename;
      state.datasetInfo = result.data; // Store dataset info
      
      // Trigger transition to data explorer immediately
      setTimeout(() => {
        showDataExplorer();
      }, 300);
    } else {
      throw new Error(result.message);
    }
  } catch(err) {
    showLoadStatus('#demoLoadStatus', 
      `Error loading demo: ${err.message}`,
      'error'
    );
  } finally {
    // Reset button state
    btn.disabled = false;
    btnText.style.display = 'inline';
    btnSpinner.style.display = 'none';
  }
}

// Drag and drop handlers
const dropzone = $('#dropzone');
const fileInput = $('#fileInput');

dropzone.addEventListener('click', ()=> fileInput.click());

fileInput.addEventListener('change', (e)=> {
  if(e.target.files[0]) handleFile(e.target.files[0]);
});

dropzone.addEventListener('dragover', (e)=> {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', ()=> {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e)=> {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

// Demo dataset loader
$('#btnLoadDemo').addEventListener('click', ()=> {
  const selectBox = $('#demoSelectBox');
  const demoKey = selectBox.value;
  if(!demoKey){
    alert('Please select a demo dataset first.');
    return;
  }
  loadDemoData(demoKey);
});

$('#btnImport').addEventListener('click', importData);