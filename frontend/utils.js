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
    startTime: null,  // 开始时间，用于计算总时长
    logPath: null,  // 实验日志路径
    maxEpsilonSeries: [],  // 存储最大 epsilon 曲线数据（绝对值）
    epsilonThreshold: 0,  // epsilon 阈值（绝对值）
    currentConditions: {},  // 当前子群体定义的条件 {feature: value}
    createdSubgroups: [],  // 已创建的子群体列表
    featureStats: {},  // 特征统计信息
    canDownload: false  // 是否可以下载transformation规则
};

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);