/**
 * BMWithAE Frontend API Client
 */

// 动态获取 API 地址：如果是本地开发使用 localhost，否则使用当前主机名
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:5001/api'
  : `http://${window.location.hostname}:5001/api`;

class BMWithAEAPI {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
    this.currentJobId = null;
    this.currentDatasetId = null;
  }

  /**
   * Upload custom dataset
   */
  async uploadDataset(file, targetColumn, protectedColumns) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_column', targetColumn);
    protectedColumns.forEach(col => {
      formData.append('protected_columns[]', col);
    });

    const response = await fetch(`${this.baseURL}/data/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (data.status === 'success') {
      this.currentDatasetId = data.data.dataset_id;
    }
    return data;
  }

  /**
   * Load demo dataset
   */
  async loadDemo(datasetName) {
    const response = await fetch(`${this.baseURL}/data/demo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ dataset_name: datasetName })
    });

    const data = await response.json();
    if (data.status === 'success') {
      this.currentDatasetId = data.data.dataset_id;
    }
    return data;
  }

  /**
   * Get current configuration
   */
  async getConfig() {
    const response = await fetch(`${this.baseURL}/config`);
    return await response.json();
  }

  /**
   * Update configuration
   */
  async updateConfig(configParams) {
    const response = await fetch(`${this.baseURL}/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(configParams)
    });
    return await response.json();
  }

  /**
   * Initialize debiasing job
   * @param {string} datasetId - Dataset ID
   * @param {string|Array} protectedAttributes - Protected attribute(s) to use for debiasing (optional)
   */
  async initDebias(datasetId = null, protectedAttributes = null) {
    const requestBody = { 
      dataset_id: datasetId || this.currentDatasetId 
    };
    
    // Add protected attributes if specified (support both string and array)
    if (protectedAttributes) {
      if (Array.isArray(protectedAttributes)) {
        requestBody.protected_attributes = protectedAttributes;
      } else {
        // Single string - convert to array for backend
        requestBody.protected_attributes = [protectedAttributes];
      }
    }
    
    const response = await fetch(`${this.baseURL}/debias/init`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    const data = await response.json();
    if (data.status === 'success') {
      this.currentJobId = data.data.job_id;
    }
    return data;
  }

  /**
   * Execute one complete iteration (BM + AE + evaluate)
   */
  async stepIteration(jobId = null) {
    const response = await fetch(
      `${this.baseURL}/debias/${jobId || this.currentJobId}/step`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    return await response.json();
  }

  /**
   * Run full debiasing process
   */
  async runFullProcess(jobId = null) {
    const response = await fetch(
      `${this.baseURL}/debias/${jobId || this.currentJobId}/run-full`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    return await response.json();
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId = null) {
    const response = await fetch(
      `${this.baseURL}/debias/${jobId || this.currentJobId}/status`
    );
    return await response.json();
  }

  /**
   * Get dataset information (features, statistics, etc.)
   */
  async getDatasetInfo(datasetId = null) {
    const response = await fetch(
      `${this.baseURL}/data/${datasetId || this.currentDatasetId}/info`
    );
    return await response.json();
  }

  /**
   * Get bias metrics for protected attribute(s)
   * @param {string} datasetId - Dataset ID
   * @param {string|Array} protectedAttributes - Protected attribute(s)
   */
  async getBiasMetrics(datasetId, protectedAttributes) {
    const requestBody = {};
    
    // Support both string and array
    if (Array.isArray(protectedAttributes)) {
      requestBody.protected_attributes = protectedAttributes;
    } else {
      // Single string - send as array
      requestBody.protected_attributes = [protectedAttributes];
    }
    
    const response = await fetch(
      `${this.baseURL}/data/${datasetId || this.currentDatasetId}/bias-metrics`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      }
    );
    return await response.json();
  }

  /**
   * Get subgroup performance metrics
   * @param {string} datasetId - Dataset ID
   * @param {Object} conditions - Subgroup conditions (e.g., {'AGE': 25, 'SEX': 'M'})
   */
  async getSubgroupMetrics(datasetId, conditions) {
    const response = await fetch(
      `${this.baseURL}/data/${datasetId || this.currentDatasetId}/subgroup-metrics`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ conditions })
      }
    );
    return await response.json();
  }
}

// Export for use in index.html
window.BMWithAEAPI = BMWithAEAPI;

