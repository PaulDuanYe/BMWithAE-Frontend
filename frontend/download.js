// Show/hide log download button
    function showLogDownloadButton() {
      const btn = $('#btnDownloadLog');
      if (btn) {
        btn.style.display = 'block';
      }
    }
    
    function hideLogDownloadButton() {
      const btn = $('#btnDownloadLog');
      if (btn) {
        btn.style.display = 'none';
      }
    }
    
    // View experiment log
    function viewExperimentLog() {
      if (!state.logPath) {
        alert('No experiment log available.');
        return;
      }
      
      // Open log file in new window or show in modal
      alert(`Experiment log saved at:\n${state.logPath}\n\nYou can find this file in the backend/logs folder.`);
    }
    
    // Download transformation rules
    async function downloadTransformationRules() {
      if (!state.jobId) {
        alert('No job available. Please run debiasing first.');
        return;
      }
      
      try {
        const url = `${API_BASE_URL}/debias/${state.jobId}/download_transforms`;
        
        // Create a temporary anchor element to trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = `transforms_${state.jobId.substring(0, 8)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        console.log('[INFO] Transformation rules download initiated');
      } catch (err) {
        console.error('[ERROR] Failed to download transformation rules:', err);
        alert(`Failed to download transformation rules: ${err.message}`);
      }
    }

    // Event Listeners
    $('#btnDocs').addEventListener('click', ()=> alert('Documentation (coming soon)'));
    
    // Config Modal
    $('#btnDownloadLog').addEventListener('click', viewExperimentLog);
    