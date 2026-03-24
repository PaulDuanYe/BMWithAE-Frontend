const state = { 
  currentFile: null,
  currentData: null,
  currentStep: 0,
  isRunning: false,
  runMode: 'all', // 'all' or 'step'
  datasetId: null,
  jobId: null,
  history: [],
  selectedMetric: 'Max_Epsilon',  // 默认选择 Max Epsilon
  /* startTime: null, */  // 开始时间，用于计算总时长
  logPath: null,  // 实验日志路径
  maxEpsilonSeries: [],  // 存储最大 epsilon 曲线数据（绝对值）
  epsilonThreshold: 0,  // epsilon 阈值（绝对值）
  currentConditions: {},  // 当前子群体定义的条件 {feature: value}
  createdSubgroups: [],  // 已创建的子群体列表
  featureStats: {},  // 特征统计信息
  canDownload: false,  // 是否可以下载transformation规则
  runDemo: true,  // 是否正在运行演示模式
  data: null  // 数据是否已经经过变换
};

// Configuration Parameters
const config = {
  numToCatMethod: 'quartile',
  numToCatCuts: 4,
  seed: 0,
  useBiasMitigation: true,
  useAccuracyEnhancement: false,
  mainStep: 'd3B',
  mainClassifier: 'LR',
  mainMaxIteration: 20,  // 增加到20轮，让epsilon有机会达到阈值
  mainTrainingRate: 0.5,
  mainThresholdEpsilon: 0.9,
  mainThresholdAccuracy: 0.01,
  mainAeImportanceMeasure: 'a1',
  mainAeRebinMethod: 'r1',
  mainAlphaO: 0.8,
  evalHOrder: 'default',
  evalSum: 'd1A',
  evalCat: 'cat-a',
  evalNum: 'num-a',
  evalScale: 'zscore',
  evalDistMetric: 'euclidean',
  evalMetricFairness: ['BNC', 'BPC', 'CUAE', 'EOpp', 'EO', 'FDRP', 'FORP', 'FNRB', 'FPRB', 'NPVP', 'OAE', 'PPVP', 'SP'],
  evalMetricAccuracy: ['ACC', 'F1', 'Recall', 'Precision'],
  transformMethod: 'poly',
  transformMulti: 't1',
  transformStream: 'd4A',
  streamConfig: {
    p: 102,
    q: 173,
    emin: 0.5,
    emax: 2,
    order: 0,
    length: 10
  }
};
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);